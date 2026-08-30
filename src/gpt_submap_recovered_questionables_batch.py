#!/usr/bin/env python3
"""
STEP 1 — Precisely submap recovered QUESTIONABLE records to evidence units.

Workflow
--------
The previous recovery step re-assigned QUESTIONABLE paper-chapter mappings
across the 8 major ESMO-PDAC chapters.

This script:
1. Takes only recovery records with decision=mapped_to_existing_chapter.
2. Expands multi-chapter recovery decisions into one paper-chapter assignment
   per corrected chapter.
3. Deduplicates globally by (PMID, corrected chapter_id), while preserving
   provenance from all original questionable assignments.
4. Excludes any PMID listed in manual_evidence_exclusions.jsonl.
5. Uses GPT-5.6 Sol to map each corrected paper-chapter assignment to:
      - one or more EXISTING evidence units in that chapter, OR
      - NEW_SUBUNIT_CANDIDATE within that chapter, OR
      - OUT_OF_SCOPE if the corrected assignment is still unsupported.
6. Does NOT assess evidence quality or rewrite the guideline.

Inputs
------
data/recovery_questionable_major_chapters.jsonl
data/evidence_unit_ontology.json
data/manual_evidence_exclusions.jsonl   (optional but enforced if present)

Outputs
-------
data/recovered_questionable_chapter_assignments.jsonl
data/gpt_recovered_questionable_submapping_batch_input.jsonl
data/gpt_recovered_questionable_submapping_batch_output.jsonl
data/recovered_questionables_submapped.csv
data/recovered_questionables_by_evidence_unit.jsonl
data/recovered_questionables_new_subunit_candidates.jsonl
data/recovered_questionables_out_of_scope.jsonl
data/recovered_questionables_submapping_parse_failures.jsonl
data/recovered_questionables_submapping_manifest.json
data/gpt_recovered_questionable_submapping_state.json

Default:
    model = gpt-5.6-sol
    reasoning_effort = high
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"

RECOVERY_INPUT = DATA / "recovery_questionable_major_chapters.jsonl"
ONTOLOGY_PATH = DATA / "evidence_unit_ontology.json"
MANUAL_EXCLUSIONS = DATA / "manual_evidence_exclusions.jsonl"

EXPANDED_INPUT = DATA / "recovered_questionable_chapter_assignments.jsonl"
BATCH_INPUT = DATA / "gpt_recovered_questionable_submapping_batch_input.jsonl"
BATCH_OUTPUT = DATA / "gpt_recovered_questionable_submapping_batch_output.jsonl"
BATCH_ERRORS = DATA / "gpt_recovered_questionable_submapping_batch_errors.jsonl"
STATE_PATH = DATA / "gpt_recovered_questionable_submapping_state.json"

OUT_CSV = DATA / "recovered_questionables_submapped.csv"
OUT_EXISTING = DATA / "recovered_questionables_by_evidence_unit.jsonl"
OUT_NEW_SUBUNIT = DATA / "recovered_questionables_new_subunit_candidates.jsonl"
OUT_OOS = DATA / "recovered_questionables_out_of_scope.jsonl"
OUT_FAILURES = DATA / "recovered_questionables_submapping_parse_failures.jsonl"
OUT_MANIFEST = DATA / "recovered_questionables_submapping_manifest.json"
LOG_PATH = LOGS / "recovered_questionables_submapping_log.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

SYSTEM_PROMPT = """You are performing a precise semantic indexing task for a medical clinical-practice-guideline evidence-update workflow.

CONTEXT
This publication was previously flagged because its initial major-chapter mapping was questionable. A separate recovery step has now re-assigned it to the major ESMO pancreatic-cancer guideline chapter shown below.

YOUR CURRENT TASK
Map the publication ONLY WITHIN THAT FIXED RECOVERED CHAPTER to the most appropriate predefined evidence unit(s).

STRICT RULES
1. Do NOT change the recovered major chapter in this task.
2. Identify the smallest set of evidence units that captures the substantive scientific content relevant to this chapter.
3. The PRIMARY unit must represent the main research question/intervention/exposure/diagnostic question/biological domain within this chapter.
4. Add SECONDARY units only when substantively studied. Background mentions, introductory statements, incidental covariates, and cited prior work do not count.
5. Do NOT assess evidence quality, risk of bias, efficacy, recommendation strength, or whether the guideline should change.
6. Do NOT infer content absent from title/abstract.
7. Broad systematic reviews/meta-analyses/reviews may receive multiple units, but avoid indiscriminate overmapping.
8. If the paper genuinely belongs to this recovered chapter but no predefined unit adequately captures its substantive topic, return NEW_SUBUNIT_CANDIDATE with a precise medically meaningful candidate title and description.
9. If even the recovered chapter is not substantively supported by the title/abstract, return OUT_OF_SCOPE rather than forcing a unit.
10. A new major chapter is NOT allowed in this task; new-major-chapter candidates are handled separately.
11. Guidelines and consensus statements were excluded upstream and must not be reintroduced.
12. Use only the publication information and the evidence-unit ontology supplied in the request.

OUTPUT
Return only the requested structured JSON.
"""


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if optional:
            return []
        raise FileNotFoundError(f"Missing required file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path.name} line {line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_event(event: str, **payload: Any) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    obj = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ontology_maps(ontology: dict[str, Any]):
    chapter_units: dict[str, list[str]] = {}
    unit_meta: dict[str, dict[str, str]] = {}
    for cid, chapter in ontology["chapters"].items():
        ids: list[str] = []
        for unit in chapter["evidence_units"]:
            uid = unit["id"]
            ids.append(uid)
            unit_meta[uid] = {
                "chapter_id": cid,
                "chapter_title": chapter["title"],
                "unit_name": unit["name"],
                "definition": unit["definition"],
                "boundary": unit["boundary"],
            }
        chapter_units[cid] = ids
    return chapter_units, unit_meta


def chapter_ontology_prompt(ontology: dict[str, Any], cid: str) -> str:
    chapter = ontology["chapters"][cid]
    lines = [
        f"FIXED RECOVERED MAJOR CHAPTER: {cid} — {chapter['title']}",
        "",
        "ALLOWED EXISTING EVIDENCE UNITS IN THIS CHAPTER:",
    ]
    for unit in chapter["evidence_units"]:
        lines += [
            "",
            f"{unit['id']} — {unit['name']}",
            f"Definition: {unit['definition']}",
            f"Boundary: {unit['boundary']}",
        ]
    return "\n".join(lines)


def response_schema(allowed_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "pdac_recovered_questionable_submapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mapping_status": {
                    "type": "string",
                    "enum": ["mapped", "new_subunit_candidate", "out_of_scope"],
                },
                "primary_evidence_unit_id": {
                    "type": "string",
                    "enum": allowed_ids + ["NEW_SUBUNIT_CANDIDATE", "OUT_OF_SCOPE"],
                },
                "secondary_evidence_unit_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_ids},
                    "minItems": 0,
                    "maxItems": 6,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "broad_multidomain_review": {"type": "boolean"},
                "candidate_title": {"type": "string"},
                "candidate_description": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": [
                "mapping_status",
                "primary_evidence_unit_id",
                "secondary_evidence_unit_ids",
                "confidence",
                "broad_multidomain_review",
                "candidate_title",
                "candidate_description",
                "rationale",
            ],
            "additionalProperties": False,
        },
    }


def article_text(rec: dict[str, Any]) -> str:
    return (
        f"PMID: {clean(rec.get('pmid'))}\n"
        f"Title: {clean(rec.get('title')) or '[missing]'}\n"
        f"Abstract: {clean(rec.get('abstract')) or '[no abstract available]'}\n"
        f"Publication types: {clean(rec.get('publication_types')) or '[not available]'}\n"
        f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}\n"
        f"MeSH terms: {clean(rec.get('mesh_terms')) or '[not available]'}\n"
        f"Keywords: {clean(rec.get('keywords')) or '[not available]'}\n"
        f"Publication year: {clean(rec.get('publication_year')) or '[not available]'}"
    )


def manual_excluded_pmids() -> set[str]:
    return {
        clean(row.get("pmid"))
        for row in load_jsonl(MANUAL_EXCLUSIONS, optional=True)
        if clean(row.get("pmid"))
    }


def build_recovered_assignments() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recovery_rows = load_jsonl(RECOVERY_INPUT)
    excluded = manual_excluded_pmids()

    # Aggregate by corrected (PMID, chapter) because the same PMID can have
    # multiple original questionable assignments that recover to the same chapter.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    provenance_chapters: dict[tuple[str, str], set[str]] = defaultdict(set)
    recovery_rationales: dict[tuple[str, str], set[str]] = defaultdict(set)
    recovery_confidences: dict[tuple[str, str], set[str]] = defaultdict(set)

    mapped_recovery_records = 0
    skipped_oos = 0
    skipped_manual = 0

    for rec in recovery_rows:
        pmid = clean(rec.get("pmid"))
        if not pmid:
            continue

        if pmid in excluded:
            skipped_manual += 1
            continue

        decision = clean(rec.get("decision"))
        if decision == "out_of_scope":
            skipped_oos += 1
            continue
        if decision != "mapped_to_existing_chapter":
            raise RuntimeError(
                f"Unexpected recovery decision for PMID {pmid}: {decision}"
            )

        mapped_recovery_records += 1
        recovered_chapters = rec.get("chapter_ids") or []
        if not recovered_chapters:
            raise RuntimeError(
                f"Recovered mapping has no chapter_ids for PMID {pmid}"
            )

        original_chapter = clean(
            rec.get("original_questionable_chapter_id")
            or rec.get("chapter_id")
        )

        for cid in recovered_chapters:
            cid = clean(cid)
            key = (pmid, cid)
            if key not in grouped:
                grouped[key] = dict(rec)
                grouped[key]["chapter_id"] = cid
                grouped[key]["chapter_title"] = ""
            if original_chapter:
                provenance_chapters[key].add(original_chapter)
            if clean(rec.get("rationale")):
                recovery_rationales[key].add(clean(rec.get("rationale")))
            if clean(rec.get("confidence")):
                recovery_confidences[key].add(clean(rec.get("confidence")))

    ontology = load_json(ONTOLOGY_PATH)
    valid_chapters = set(ontology["chapters"])

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda x: (x[1], int(x[0]) if x[0].isdigit() else x[0])):
        pmid, cid = key
        if cid not in valid_chapters:
            raise RuntimeError(f"Recovered invalid chapter {cid} for PMID {pmid}")

        rec = grouped[key]
        rec["chapter_id"] = cid
        rec["chapter_title"] = ontology["chapters"][cid]["title"]
        rec["recovery_original_questionable_chapter_ids"] = sorted(
            provenance_chapters[key]
        )
        rec["recovery_rationales"] = sorted(recovery_rationales[key])
        rec["recovery_confidences"] = sorted(recovery_confidences[key])
        rows.append(rec)

    stats = {
        "recovery_rows_total": len(recovery_rows),
        "recovery_records_mapped_to_existing_chapter": mapped_recovery_records,
        "recovery_records_out_of_scope_skipped": skipped_oos,
        "manual_exclusion_rows_skipped": skipped_manual,
        "unique_recovered_paper_chapter_assignments": len(rows),
        "unique_pmids_in_recovered_assignments": len(
            {clean(r.get("pmid")) for r in rows}
        ),
    }
    return rows, stats


def prepare(model: str, reasoning_effort: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY_PATH)
    chapter_units, _ = ontology_maps(ontology)
    assignments, stats = build_recovered_assignments()

    write_jsonl(EXPANDED_INPUT, assignments)

    chapter_counts = Counter()
    custom_ids: list[str] = []

    with BATCH_INPUT.open("w", encoding="utf-8", newline="\n") as f:
        for rec in assignments:
            pmid = clean(rec.get("pmid"))
            cid = clean(rec.get("chapter_id"))
            custom_id = f"rq-{pmid}-{cid.replace('.', '_')}"
            custom_ids.append(custom_id)

            user = (
                chapter_ontology_prompt(ontology, cid)
                + "\n\nPUBLICATION TO SUBMAP:\n"
                + article_text(rec)
                + "\n\nMap this publication only within the fixed recovered chapter."
            )

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema(chapter_units[cid]),
                },
                "max_completion_tokens": 1800,
                "reasoning_effort": reasoning_effort,
            }

            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            chapter_counts[cid] += 1

    if len(custom_ids) != len(set(custom_ids)):
        duplicates = [
            x for x, n in Counter(custom_ids).items() if n > 1
        ]
        raise RuntimeError(
            f"HARD FAIL: duplicate custom_id values: {duplicates[:10]}"
        )

    size_mb = BATCH_INPUT.stat().st_size / 1024 / 1024
    if len(custom_ids) > 50_000:
        raise RuntimeError("Batch exceeds 50,000 requests.")
    if size_mb > 190:
        raise RuntimeError(
            f"Batch is {size_mb:.2f} MB; split before upload."
        )

    summary = {
        **stats,
        "batch_requests": len(custom_ids),
        "batch_jsonl_mb": round(size_mb, 2),
        "chapter_request_counts": dict(chapter_counts),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    print("\nRecovered-questionable submapping prepared.")
    for key in [
        "recovery_rows_total",
        "recovery_records_mapped_to_existing_chapter",
        "recovery_records_out_of_scope_skipped",
        "manual_exclusion_rows_skipped",
        "unique_recovered_paper_chapter_assignments",
        "unique_pmids_in_recovered_assignments",
        "batch_requests",
    ]:
        print(f"  {key}: {summary[key]:,}")
    print(f"  batch JSONL MB: {size_mb:.2f}")
    print(f"  unique custom IDs: {len(set(custom_ids)):,}/{len(custom_ids):,}")
    print(f"  model: {model}")
    print(f"  reasoning effort: {reasoning_effort}")
    print()
    for cid in ontology["chapters"]:
        print(f"  chapter {cid:>3}: {chapter_counts[cid]:,} requests")
    print(f"\n  expanded recovered input: {EXPANDED_INPUT}")
    print(f"  batch input:              {BATCH_INPUT}")

    log_event("prepared", **summary)
    return summary


class OpenAIHTTP:
    def __init__(self, api_key: str, retry_wait: int):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty.")
        self.session = requests.Session()
        self.auth = {"Authorization": f"Bearer {api_key}"}
        self.retry_wait = retry_wait

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 300,
        **kwargs: Any,
    ) -> requests.Response:
        url = path if path.startswith("http") else OPENAI_BASE_URL + path

        while True:
            headers = dict(self.auth)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.ChunkedEncodingError,
                requests.ContentDecodingError,
            ) as e:
                print(
                    f"WARN: transient network/stream error "
                    f"{type(e).__name__}: {e}; retry in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                print(
                    f"WARN: transient OpenAI HTTP {r.status_code}; "
                    f"retry in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {r.status_code}: {r.text[:5000]}"
                )
            return r


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return load_json(STATE_PATH)


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def submit(
    client: OpenAIHTTP, model: str, reasoning_effort: str
) -> dict[str, Any]:
    state = load_state()
    if state.get("batch_id"):
        print(f"Existing batch found: {state['batch_id']}")
        print("Resuming it; no duplicate paid batch submitted.")
        return state

    if not BATCH_INPUT.exists():
        raise RuntimeError("Batch input missing. Run --mode prepare first.")

    print("\nUploading Batch input...")
    with BATCH_INPUT.open("rb") as f:
        uploaded = client.request(
            "POST",
            "/files",
            files={
                "file": (
                    BATCH_INPUT.name,
                    f,
                    "application/jsonl",
                )
            },
            data={"purpose": "batch"},
        ).json()

    payload = {
        "input_file_id": uploaded["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "project": "ESMO_PDAC_2015_to_2023_PoC",
            "task": "submap_recovered_questionables",
            "model": model,
            "reasoning_effort": reasoning_effort,
        },
    }

    batch = client.request(
        "POST",
        "/batches",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    ).json()

    state = {
        "input_file_id": uploaded["id"],
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    save_state(state)

    print(f"  batch id: {batch['id']}")
    print(f"  status:   {batch.get('status')}")
    return state


def download_file(
    client: OpenAIHTTP, file_id: str, destination: Path
) -> None:
    destination.write_bytes(
        client.request(
            "GET", f"/files/{file_id}/content", timeout=600
        ).content
    )


def watch(client: OpenAIHTTP, poll_seconds: int) -> dict[str, Any]:
    state = load_state()
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No batch_id in state.")

    while True:
        batch = client.request(
            "GET", f"/batches/{batch_id}"
        ).json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}
        print(
            f"status={status}; total={counts.get('total')}; "
            f"completed={counts.get('completed')}; "
            f"failed={counts.get('failed')}"
        )

        state.update(
            {
                "status": status,
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "request_counts": counts,
            }
        )
        save_state(state)

        if status in TERMINAL_BATCH_STATUSES:
            if batch.get("output_file_id"):
                download_file(
                    client,
                    batch["output_file_id"],
                    BATCH_OUTPUT,
                )
            if batch.get("error_file_id"):
                download_file(
                    client,
                    batch["error_file_id"],
                    BATCH_ERRORS,
                )
            return batch

        time.sleep(poll_seconds)


def parse_custom_id(value: str) -> tuple[str, str]:
    m = re.fullmatch(r"rq-(\d+)-([0-9_]+)", value)
    if not m:
        raise ValueError(f"Invalid custom_id: {value}")
    return m.group(1), m.group(2).replace("_", ".")


def normalize_result(
    parsed: dict[str, Any],
    allowed_ids: list[str],
) -> dict[str, Any]:
    status = parsed["mapping_status"]
    primary = parsed["primary_evidence_unit_id"]
    secondary = list(parsed.get("secondary_evidence_unit_ids", []))
    confidence = parsed["confidence"]
    broad = bool(parsed["broad_multidomain_review"])
    candidate_title = clean(parsed.get("candidate_title"))
    candidate_description = clean(parsed.get("candidate_description"))
    rationale = clean(parsed.get("rationale"))

    allowed = set(allowed_ids)
    if any(x not in allowed for x in secondary):
        raise ValueError("Invalid secondary evidence unit.")

    normalization: list[str] = []

    if status == "mapped":
        if primary not in allowed:
            raise ValueError("Mapped status with invalid primary unit.")
        if candidate_title or candidate_description:
            normalization.append("cleared_irrelevant_candidate_fields")
        candidate_title = ""
        candidate_description = ""

        # Deduplicate and do not repeat primary among secondary.
        secondary_set = set(secondary)
        secondary_set.discard(primary)
        secondary = [
            uid for uid in allowed_ids if uid in secondary_set
        ]

    elif status == "new_subunit_candidate":
        if primary != "NEW_SUBUNIT_CANDIDATE":
            raise ValueError(
                "new_subunit_candidate status without matching sentinel."
            )
        if not candidate_title or not candidate_description:
            raise ValueError(
                "new_subunit_candidate missing title/description."
            )
        if secondary:
            normalization.append(
                "cleared_irrelevant_secondary_existing_units"
            )
        secondary = []

    elif status == "out_of_scope":
        if primary != "OUT_OF_SCOPE":
            raise ValueError(
                "out_of_scope status without OUT_OF_SCOPE sentinel."
            )
        if secondary or candidate_title or candidate_description:
            normalization.append(
                "cleared_irrelevant_assignment_fields"
            )
        secondary = []
        candidate_title = ""
        candidate_description = ""

    else:
        raise ValueError(f"Unknown mapping_status: {status}")

    return {
        "mapping_status": status,
        "primary_evidence_unit_id": primary,
        "secondary_evidence_unit_ids": secondary,
        "confidence": confidence,
        "broad_multidomain_review": broad,
        "candidate_title": candidate_title,
        "candidate_description": candidate_description,
        "rationale": rationale,
        "deterministic_normalization": "; ".join(normalization),
    }


def parse_output():
    ontology = load_json(ONTOLOGY_PATH)
    chapter_units, _ = ontology_maps(ontology)
    source_rows = {
        (clean(r.get("pmid")), clean(r.get("chapter_id"))): r
        for r in load_jsonl(EXPANDED_INPUT)
    }

    results: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    with BATCH_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            try:
                pmid, cid = parse_custom_id(custom_id)
                response = obj.get("response")
                if (
                    obj.get("error")
                    or not response
                    or response.get("status_code") != 200
                ):
                    raise ValueError(
                        obj.get("error")
                        or f"HTTP "
                        f"{response.get('status_code') if response else 'missing'}"
                    )

                content = (
                    response["body"]["choices"][0]["message"]["content"]
                )
                parsed = json.loads(content)
                normalized = normalize_result(
                    parsed, chapter_units[cid]
                )

                key = (pmid, cid)
                if key not in source_rows:
                    raise ValueError(
                        f"No recovered source assignment for {key}"
                    )
                if key in results:
                    raise ValueError(
                        f"Duplicate response for recovered assignment {key}"
                    )
                results[key] = normalized

            except Exception as e:
                failures.append(
                    {
                        "line": line_no,
                        "custom_id": custom_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    return source_rows, results, failures


def merge(model: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY_PATH)
    _, unit_meta = ontology_maps(ontology)
    source_rows, results, failures = parse_output()
    state = load_state()

    csv_rows: list[dict[str, str]] = []
    existing_expanded: list[dict[str, Any]] = []
    new_subunit: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    status_counts = Counter()
    confidence_counts = Counter()
    unit_counts = Counter()

    fields: set[str] = set()

    for key, source in source_rows.items():
        pmid, cid = key
        result = results.get(key)

        if not result:
            missing.append({"pmid": pmid, "chapter_id": cid})
            continue

        status = result["mapping_status"]
        status_counts[status] += 1
        confidence_counts[result["confidence"]] += 1

        base = dict(source)
        base.update(
            {
                "recovered_submapping_status": status,
                "recovered_primary_evidence_unit_id":
                    result["primary_evidence_unit_id"],
                "recovered_secondary_evidence_unit_ids":
                    result["secondary_evidence_unit_ids"],
                "recovered_submapping_confidence":
                    result["confidence"],
                "recovered_broad_multidomain_review":
                    result["broad_multidomain_review"],
                "recovered_candidate_title":
                    result["candidate_title"],
                "recovered_candidate_description":
                    result["candidate_description"],
                "recovered_submapping_rationale":
                    result["rationale"],
                "deterministic_normalization":
                    result["deterministic_normalization"],
                "recovered_submapping_model": model,
                "recovered_submapping_batch_id":
                    clean(state.get("batch_id")),
            }
        )

        if status == "mapped":
            all_units = [
                result["primary_evidence_unit_id"],
                *result["secondary_evidence_unit_ids"],
            ]
            base["recovered_all_evidence_unit_ids"] = all_units
            base["recovered_all_evidence_unit_names"] = [
                unit_meta[u]["unit_name"] for u in all_units
            ]

            for uid in all_units:
                unit_counts[uid] += 1
                expanded = dict(base)
                expanded.update(
                    {
                        "evidence_unit_id": uid,
                        "evidence_unit_name":
                            unit_meta[uid]["unit_name"],
                        "evidence_unit_definition":
                            unit_meta[uid]["definition"],
                        "submapping_role":
                            "primary"
                            if uid
                            == result["primary_evidence_unit_id"]
                            else "secondary",
                    }
                )
                existing_expanded.append(expanded)

        elif status == "new_subunit_candidate":
            base["recovered_all_evidence_unit_ids"] = []
            base["recovered_all_evidence_unit_names"] = []
            new_subunit.append(base)

        elif status == "out_of_scope":
            base["recovered_all_evidence_unit_ids"] = []
            base["recovered_all_evidence_unit_names"] = []
            out_of_scope.append(base)

        flat: dict[str, str] = {}
        for k, v in base.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = clean(v)
        fields.update(flat)
        csv_rows.append(flat)

    write_jsonl(OUT_EXISTING, existing_expanded)
    write_jsonl(OUT_NEW_SUBUNIT, new_subunit)
    write_jsonl(OUT_OOS, out_of_scope)
    write_jsonl(OUT_FAILURES, failures)

    preferred = [
        "pmid",
        "title",
        "abstract",
        "chapter_id",
        "chapter_title",
        "evidence_labels",
        "publication_types",
        "recovered_submapping_status",
        "recovered_primary_evidence_unit_id",
        "recovered_secondary_evidence_unit_ids",
        "recovered_all_evidence_unit_ids",
        "recovered_all_evidence_unit_names",
        "recovered_submapping_confidence",
        "recovered_submapping_rationale",
        "recovered_broad_multidomain_review",
        "recovered_candidate_title",
        "recovered_candidate_description",
        "recovery_original_questionable_chapter_ids",
        "recovery_rationales",
        "recovery_confidences",
        "recovered_submapping_model",
        "recovered_submapping_batch_id",
        "deterministic_normalization",
    ]
    fieldnames = [x for x in preferred if x in fields]
    fieldnames += sorted(fields - set(fieldnames), key=str.casefold)

    with OUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "input_unique_recovered_paper_chapter_assignments":
            len(source_rows),
        "successful_submappings": len(results),
        "missing_submappings": len(missing),
        "parse_failures": len(failures),
        "mapping_status_counts": dict(status_counts),
        "confidence_counts": dict(confidence_counts),
        "expanded_existing_unit_assignments":
            len(existing_expanded),
        "new_subunit_candidates": len(new_subunit),
        "out_of_scope_after_recovered_chapter_mapping":
            len(out_of_scope),
        "evidence_unit_counts": {
            uid: {
                "chapter_id": unit_meta[uid]["chapter_id"],
                "unit_name": unit_meta[uid]["unit_name"],
                "paper_assignments": unit_counts[uid],
            }
            for uid in unit_meta
        },
        "missing_records": missing,
        "manual_exclusion_file_used":
            str(MANUAL_EXCLUSIONS)
            if MANUAL_EXCLUSIONS.exists()
            else None,
        "next_step": (
            "Combine these new-subunit candidates with the prior "
            "recovery_new_subunit_candidates.jsonl and cluster within "
            "each major chapter."
        ),
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRecovered-questionable submapping merge completed.")
    print(
        f"  input recovered paper-chapter assignments: "
        f"{len(source_rows):,}"
    )
    print(f"  successful submappings:                    {len(results):,}")
    print(f"  missing submappings:                       {len(missing):,}")
    print(f"  parse failures:                            {len(failures):,}")
    print(f"  mapped to existing units:                  {status_counts['mapped']:,}")
    print(f"  new subunit candidates:                    {status_counts['new_subunit_candidate']:,}")
    print(f"  out of scope:                              {status_counts['out_of_scope']:,}")
    print(f"  expanded existing-unit assignments:        {len(existing_expanded):,}")
    print()
    print(f"  CSV:             {OUT_CSV}")
    print(f"  existing units:  {OUT_EXISTING}")
    print(f"  new subunits:    {OUT_NEW_SUBUNIT}")
    print(f"  out of scope:    {OUT_OOS}")
    print(f"  failures:        {OUT_FAILURES}")
    print(f"  manifest:        {OUT_MANIFEST}")

    if missing or failures:
        print(
            "\nWARNING: repair missing/failed mappings before Step 2."
        )

    return manifest


def synchronous_test(client: OpenAIHTTP) -> None:
    if not BATCH_INPUT.exists():
        raise RuntimeError(
            "Batch input missing. Run --mode prepare first."
        )
    with BATCH_INPUT.open("r", encoding="utf-8") as f:
        request_obj = json.loads(f.readline())

    print(f"Testing: {request_obj['custom_id']}")
    response = client.request(
        "POST",
        "/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_obj["body"]),
    )
    print(f"HTTP {response.status_code}")
    print(
        json.dumps(
            response.json(),
            ensure_ascii=False,
            indent=2,
        )[:7000]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: submap recovered questionable major-chapter "
            "assignments to precise evidence units."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "test", "submit", "watch", "merge", "all"],
        default="prepare",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()
        print(f"Deleted state: {STATE_PATH}")

    if args.mode in {"prepare", "all"}:
        prepare(args.model, args.reasoning_effort)
        if args.mode == "prepare":
            return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if (
        args.mode in {"test", "submit", "watch", "all"}
        and not api_key
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this PowerShell session."
        )

    client = (
        OpenAIHTTP(api_key, args.retry_wait)
        if api_key
        else None
    )

    if args.mode == "test":
        synchronous_test(client)
        return 0

    if args.mode in {"submit", "all"}:
        submit(
            client,
            args.model,
            args.reasoning_effort,
        )
        if args.mode == "submit":
            return 0

    if args.mode in {"watch", "all"}:
        batch = watch(client, args.poll_seconds)
        print(f"Batch terminal status: {batch.get('status')}")
        if batch.get("status") != "completed":
            return 2
        merge(args.model)
        return 0

    if args.mode == "merge":
        merge(args.model)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
