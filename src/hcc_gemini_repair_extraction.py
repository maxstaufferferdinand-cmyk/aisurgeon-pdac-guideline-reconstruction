from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temp.replace(path)


def append_usage(hcc_root: Path, entry: dict[str, Any]) -> None:
    ledger_path = hcc_root / "run_state" / "cost_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"requests": []}
    ledger.setdefault("requests", []).append(entry)
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
        started = utc_now()
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
                append_usage(
                    hcc_root,
                    {
                        "provider": "gemini",
                        "phase": "source_pdf_extraction_targeted_repair",
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
                    "phase": "source_pdf_extraction_targeted_repair",
                    "model": MODEL,
                    "request_timestamp": started,
                    "response_timestamp": utc_now(),
                    "retry_count": retry_count,
                    "http_status": exc.code,
                    "error_category": category,
                    "status": "failed" if exc.code not in TRANSIENT_STATUS else "retrying",
                },
            )
            if exc.code in TRANSIENT_STATUS:
                retry_count += 1
                time.sleep(120)
                continue
            raise RuntimeError(f"Gemini targeted repair failed: {category} HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError):
            append_usage(
                hcc_root,
                {
                    "provider": "gemini",
                    "phase": "source_pdf_extraction_targeted_repair",
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


def prompt(protocol: dict[str, Any], document_map: dict[str, Any]) -> str:
    return f"""
The previous canonical Gemini extraction of the attached PDF had a concrete
defect: source_chronology narrative entries were abbreviated with ellipses.
Perform a targeted repair only for that defect and for stable formal-item IDs.

Use only the attached ESMO HCC 2012 PDF. Do not use web knowledge or later HCC
guidelines.

Return one valid JSON object with:
- source_sections_verbatim: array in source order. Include every narrative
  chapter/subsection and source-native section: incidence and epidemiology,
  diagnosis and pathology, staging, management of local disease: radical
  therapies, management of locally advanced/metastatic disease: palliative
  treatments, transcatheter devices, systemic therapy, external beam
  radiotherapy, response evaluation and follow-up, note, conflict of interest.
  Each object: id, order_index, heading_path, page_start, page_end, full_text,
  citation_numbers, grades_or_levels. full_text must be verbatim continuous
  source text and must not contain ellipses used as abbreviation.
- formal_items_normalized: every item from Table 4 Summary of recommendations
  in source order. Each object: id, order_index, heading_path, original_wording,
  grade_or_level, source_citations, page, linked_context_ids. Preserve original
  wording and grading exactly.
- repair_notes: concrete unresolved extraction issues, if any.

Locked facts:
- source SHA-256: {protocol['source_pdf']['sha256']}
- documented last update: {protocol['source_pdf']['documented_last_update']}
- original reference range: 1-38
- source chapters from document_map: {json.dumps(document_map.get('chapters', []), ensure_ascii=False)}
"""


def parse_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini repair response did not include candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text:
        raise RuntimeError("Gemini repair response did not include text.")
    return text


def no_text_issue(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates") or []
    first = candidates[0] if candidates else {}
    return {
        "severity": "review",
        "issue": "targeted Gemini repair returned no text",
        "finish_reason": first.get("finishReason"),
        "finish_message": first.get("finishMessage"),
        "status": "canonical extraction retained; deterministic QC will audit downstream source coverage",
    }


def qc(repair: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    sections = repair.get("source_sections_verbatim", [])
    formal = repair.get("formal_items_normalized", [])
    if len(sections) < 9:
        issues.append({"severity": "mandatory", "issue": "too few verbatim source sections"})
    if len(formal) != 31:
        issues.append({"severity": "mandatory", "issue": "formal item count is not 31", "observed": len(formal)})
    for section in sections:
        text = section.get("full_text", "")
        if "..." in text or "\u2026" in text:
            issues.append({"severity": "mandatory", "issue": "section contains ellipsis abbreviation", "id": section.get("id")})
        if len(text) < 100 and section.get("heading_path") not in {"note", "conflict of interest"}:
            issues.append({"severity": "review", "issue": "section text is unusually short", "id": section.get("id")})
    ids = [item.get("id") for item in formal]
    if len(ids) != len(set(ids)):
        issues.append({"severity": "mandatory", "issue": "formal item IDs are not unique"})
    for item in formal:
        if not item.get("original_wording"):
            issues.append({"severity": "mandatory", "issue": "empty formal item wording", "id": item.get("id")})
    return (
        {
            "created_at": utc_now(),
            "gemini_model": MODEL,
            "mandatory_issue_count": sum(1 for issue in issues if issue["severity"] == "mandatory"),
            "issues": issues,
            "counts": {"sections": len(sections), "formal_items": len(formal)},
        },
        issues,
    )


def merge_chronology(existing: list[dict[str, Any]], repair: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in repair.get("source_sections_verbatim", []):
        rows.append(
            {
                "id": section.get("id"),
                "order_index": section.get("order_index"),
                "item_type": "verbatim_section",
                "heading_path": section.get("heading_path"),
                "page": section.get("page_start"),
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "text": section.get("full_text"),
                "citation_numbers": section.get("citation_numbers", []),
                "grades_or_levels": section.get("grades_or_levels", []),
            }
        )
    table_and_figure = [
        row
        for row in existing
        if row.get("item_type") in {"table_entry", "figure_entry"}
    ]
    start = max((row.get("order_index") or 0 for row in rows), default=0)
    for offset, row in enumerate(table_and_figure, start=1):
        row = dict(row)
        row["order_index"] = start + offset
        rows.append(row)
    return sorted(rows, key=lambda row: row.get("order_index") or 9999)


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted Gemini repair for source extraction chronology.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    hcc_root = Path(args.hcc_root)
    output_dir = hcc_root / "data" / "source_extraction"
    repair_path = output_dir / "source_extraction_targeted_repair.json"
    qc_path = output_dir / "extraction_repair_qc.json"
    if repair_path.exists() and qc_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "repair": str(repair_path)}, indent=2))
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is not set.")
    protocol = read_json(hcc_root / "config" / "protocol_lock.json")
    document_map = read_json(output_dir / "document_map.json")
    existing_chronology = read_jsonl(output_dir / "source_chronology.jsonl")
    pdf_b64 = base64.b64encode(Path(protocol["source_pdf"]["path"]).read_bytes()).decode("ascii")
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt(protocol, document_map)},
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
    response = gemini_generate(api_key, request_body, hcc_root)
    atomic_write_json(output_dir / "gemini_targeted_repair_raw_response.json", response)
    try:
        repair_text = parse_text(response)
    except RuntimeError:
        issue = no_text_issue(response)
        atomic_write_json(
            qc_path,
            {
                "created_at": utc_now(),
                "gemini_model": MODEL,
                "mandatory_issue_count": 0,
                "issues": [issue],
                "counts": {"sections": 0, "formal_items": 0},
            },
        )
        write_jsonl(output_dir / "unresolved_extraction_issues.jsonl", [issue])
        print(json.dumps({"status": "repair_no_text", "issue": issue}, indent=2, sort_keys=True))
        return 0
    repair = json.loads(repair_text)
    qc_doc, issues = qc(repair)
    atomic_write_json(repair_path, repair)
    atomic_write_json(qc_path, qc_doc)
    write_jsonl(output_dir / "source_chronology.jsonl", merge_chronology(existing_chronology, repair))
    write_jsonl(output_dir / "formal_items.jsonl", repair.get("formal_items_normalized", []))
    write_jsonl(output_dir / "unresolved_extraction_issues.jsonl", issues)
    print(json.dumps({"status": "complete" if not issues else "qc_issues", "qc": qc_doc}, indent=2, sort_keys=True))
    return 0 if not any(issue["severity"] == "mandatory" for issue in issues) else 2


if __name__ == "__main__":
    sys.exit(main())
