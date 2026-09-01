from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
MODEL = "models/gemini-3.5-flash"
TRANSIENT_STATUS = {500, 502, 503, 504}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temp.replace(path)


def append_usage(hcc_root: Path, entry: dict[str, Any]) -> None:
    ledger_path = hcc_root / "run_state" / "cost_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"requests": [], "phases": []}
    requests = ledger.setdefault("requests", [])
    requests.append(entry)
    atomic_write_json(ledger_path, ledger)


def classify_provider_error(status: int | None, body: str) -> str:
    text = body.lower()
    if status in TRANSIENT_STATUS:
        return "transient_server_error"
    if status in {401, 403}:
        return "credential_or_permission_error"
    if status == 429:
        if any(term in text for term in ("quota", "billing", "usage", "limit")):
            return "provider_quota_or_usage_limit"
        return "rate_limit"
    if any(term in text for term in ("billing", "quota", "usage limit", "insufficient")):
        return "provider_billing_or_quota_error"
    return "provider_non_transient_error"


def gemini_generate(api_key: str, request_body: dict[str, Any], hcc_root: Path) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key="
    url += urllib.parse.quote(api_key)
    payload = json.dumps(request_body).encode("utf-8")
    retry_count = 0
    while True:
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = utc_now()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                append_usage(
                    hcc_root,
                    {
                        "provider": "gemini",
                        "phase": "source_pdf_extraction",
                        "model": MODEL,
                        "request_timestamp": started,
                        "response_timestamp": utc_now(),
                        "retry_count": retry_count,
                        "usage_metadata": data.get("usageMetadata", {}),
                        "status": "succeeded",
                    },
                )
                return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            category = classify_provider_error(exc.code, body)
            append_usage(
                hcc_root,
                {
                    "provider": "gemini",
                    "phase": "source_pdf_extraction",
                    "model": MODEL,
                    "request_timestamp": started,
                    "response_timestamp": utc_now(),
                    "retry_count": retry_count,
                    "http_status": exc.code,
                    "error_category": category,
                    "status": "failed" if category != "transient_server_error" else "retrying",
                },
            )
            if exc.code in TRANSIENT_STATUS:
                retry_count += 1
                time.sleep(120)
                continue
            raise RuntimeError(f"Gemini extraction failed: {category} HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            append_usage(
                hcc_root,
                {
                    "provider": "gemini",
                    "phase": "source_pdf_extraction",
                    "model": MODEL,
                    "request_timestamp": started,
                    "response_timestamp": utc_now(),
                    "retry_count": retry_count,
                    "error_category": "transient_network_error",
                    "status": "retrying",
                },
            )
            retry_count += 1
            time.sleep(120)


def extraction_prompt(protocol: dict[str, Any]) -> str:
    return f"""
You are extracting a source clinical-practice guideline PDF for a blinded
scientific reconstruction study. Extract only the attached PDF. Do not use web
knowledge or any later HCC guideline.

Return one valid JSON object with these top-level keys:
- document_map: object with title, authors, publication, pages, chapters,
  subsections, tables, figures, and page spans. Preserve source order.
- source_chronology: array of chronological source-native items. Include
  narrative sections, formal recommendations, table entries, figure entries,
  note, conflicts, and references. Each item needs stable id, order_index,
  item_type, heading_path, page, text, citation_numbers, grades_or_levels.
- formal_items: array containing every formal recommendation, key point,
  statement, graded statement, table item, figure item, and source-native formal
  item type. Include original wording, grade/level, source citations, page,
  linked_context_ids.
- original_references: array for references 1 through 38 with number, full_text,
  first_author, year, title, journal, page.
- grading_systems: object describing levels of evidence, grades of
  recommendation, statement-without-grading policy, and any source wording.
- extraction_notes: array of concrete extraction uncertainties, if any.

Locked scientific facts that the extraction must preserve:
- source SHA-256: {protocol['source_pdf']['sha256']}
- documented last update: {protocol['source_pdf']['documented_last_update']}
- search_start: {protocol['dates']['search_start']}
- search_end: {protocol['dates']['search_end']}
- original reference range: 1-38

Keep original citation numbers unchanged. Do not modernize wording.
"""


def parse_candidate_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini response did not include candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if not texts:
        raise RuntimeError("Gemini response did not include text parts.")
    return "\n".join(texts)


def validate_extraction(extraction: dict[str, Any], protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    refs = extraction.get("original_references", [])
    ref_numbers = sorted(ref.get("number") for ref in refs if isinstance(ref.get("number"), int))
    chapters = extraction.get("document_map", {}).get("chapters", [])
    chronology = extraction.get("source_chronology", [])
    formal_items = extraction.get("formal_items", [])

    if ref_numbers != list(range(1, 39)):
        issues.append(
            {
                "severity": "mandatory",
                "issue": "original reference list is not exactly 1-38",
                "observed": ref_numbers,
            }
        )
    if len(chapters) < 6:
        issues.append({"severity": "mandatory", "issue": "fewer than six source chapters extracted"})
    if not chronology:
        issues.append({"severity": "mandatory", "issue": "source chronology is empty"})
    if not formal_items:
        issues.append({"severity": "mandatory", "issue": "formal item extraction is empty"})

    text_blob = json.dumps(extraction, ensure_ascii=False)
    for required in (
        protocol["source_pdf"]["documented_last_update"],
        "Incidence and epidemiology",
        "Diagnosis",
        "Staging",
        "Management of local disease",
        "Management of locally advanced/metastatic disease",
        "Response evaluation and follow-up",
    ):
        if required.lower() not in text_blob.lower():
            issues.append({"severity": "mandatory", "issue": f"missing required source text marker: {required}"})

    qc = {
        "created_at": utc_now(),
        "gemini_model": MODEL,
        "mandatory_issue_count": sum(1 for issue in issues if issue["severity"] == "mandatory"),
        "issues": issues,
        "counts": {
            "document_map_chapters": len(chapters),
            "source_chronology_items": len(chronology),
            "formal_items": len(formal_items),
            "original_references": len(refs),
        },
    }
    return qc, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical Gemini native-PDF extraction for ESMO HCC 2012.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    hcc_root = Path(args.hcc_root)
    protocol = read_json(hcc_root / "config" / "protocol_lock.json")
    if protocol.get("paid_phase_blocked"):
        raise RuntimeError("Protocol lock still blocks paid phases.")
    if protocol.get("source_pdf", {}).get("sha256") != "b65f49ba27e4640cb63976476818c79e97fc78cf111333f1ceed17e66e4b8482":
        raise RuntimeError("Source PDF hash no longer matches the locked HCC source.")

    output_dir = hcc_root / "data" / "source_extraction"
    final_outputs = [
        output_dir / "document_map.json",
        output_dir / "source_chronology.jsonl",
        output_dir / "formal_items.jsonl",
        output_dir / "original_references.json",
        output_dir / "grading_systems.json",
        output_dir / "extraction_qc.json",
        output_dir / "unresolved_extraction_issues.jsonl",
    ]
    if not args.force and all(path.exists() for path in final_outputs):
        print(json.dumps({"status": "already_complete", "output_dir": str(output_dir)}, indent=2))
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is not set.")

    pdf_path = Path(protocol["source_pdf"]["path"])
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": extraction_prompt(protocol)},
                    {"inlineData": {"mimeType": "application/pdf", "data": pdf_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 65536,
        },
    }
    raw_response = gemini_generate(api_key, request_body, hcc_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "gemini_raw_response.json", raw_response)

    extraction_text = parse_candidate_text(raw_response)
    try:
        extraction = json.loads(extraction_text)
    except json.JSONDecodeError as exc:
        (output_dir / "gemini_unparsed_text.txt").write_text(extraction_text, encoding="utf-8")
        raise RuntimeError(f"Gemini extraction did not return valid JSON: {exc}") from exc

    qc, issues = validate_extraction(extraction, protocol)
    atomic_write_json(output_dir / "document_map.json", extraction.get("document_map", {}))
    append_jsonl(output_dir / "source_chronology.jsonl", extraction.get("source_chronology", []))
    append_jsonl(output_dir / "formal_items.jsonl", extraction.get("formal_items", []))
    atomic_write_json(output_dir / "original_references.json", extraction.get("original_references", []))
    atomic_write_json(output_dir / "grading_systems.json", extraction.get("grading_systems", {}))
    atomic_write_json(output_dir / "extraction_qc.json", qc)
    append_jsonl(output_dir / "unresolved_extraction_issues.jsonl", issues)

    summary = {"status": "complete" if not issues else "qc_issues", "output_dir": str(output_dir), "qc": qc}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
