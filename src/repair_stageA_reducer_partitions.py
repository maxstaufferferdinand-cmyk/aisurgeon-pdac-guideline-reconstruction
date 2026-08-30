#!/usr/bin/env python3
r"""
Repair the four Stage-A reducer outputs that failed only because their PMID
status partition was incomplete or duplicated.

NO OpenAI/API call.

The Stage-A chunk appraisal is treated as canonical for per-paper status.
This script:
1. reads stageA_evidence_chunk_results.jsonl
2. reads the original reducer Batch output JSONL
3. for ONLY the four known failed reducer custom_ids, replaces:
      main_synthesis_pmids
      context_only_pmids
      appendix_pmids
      rejected_pmids
   with the exact partition derived from the already completed chunk decisions
4. preserves all other reducer synthesis fields exactly as GPT returned them
5. creates a backup + audit JSON
6. writes the patched reducer Batch output back to the canonical path

Afterwards rerun:
    uv run python .\src\final_stageA_evidence_synthesis_gpt56.py --mode merge-reducers --model gpt-5.6-sol
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CHUNK_RESULTS = DATA / "stageA_evidence_chunk_results.jsonl"
REDUCER_OUTPUT = DATA / "stageA_unit_reducer_batch_output.jsonl"

BACKUP = DATA / "stageA_unit_reducer_batch_output_before_partition_repair.jsonl"
AUDIT = DATA / "stageA_unit_reducer_partition_repair_audit.json"

TARGET_CUSTOM_IDS = {
    "stageA-reduce-2_DOT_2",
    "stageA-reduce-3_DOT_N05",
    "stageA-reduce-5_DOT_19",
    "stageA-reduce-5_DOT_N12",
}

STATUS_TO_FIELD = {
    "MAIN_SYNTHESIS": "main_synthesis_pmids",
    "CONTEXT_ONLY": "context_only_pmids",
    "APPENDIX": "appendix_pmids",
    "REJECT": "rejected_pmids",
}


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path.name} line {n}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reducer_uid(custom_id: str) -> str:
    prefix = "stageA-reduce-"
    if not custom_id.startswith(prefix):
        raise ValueError(custom_id)
    return custom_id[len(prefix):].replace("_DOT_", ".")


def canonical_status_by_unit():
    by_unit = defaultdict(dict)
    order_by_unit = defaultdict(list)

    for chunk in load_jsonl(CHUNK_RESULTS):
        uid = clean(chunk.get("evidence_unit_id"))
        decisions = (chunk.get("result") or {}).get("paper_decisions") or []

        for d in decisions:
            pmid = clean(d.get("pmid"))
            status = clean(d.get("paper_status"))

            if status not in STATUS_TO_FIELD:
                raise RuntimeError(
                    f"Unexpected chunk paper_status={status!r} in unit {uid}, PMID {pmid}"
                )

            if pmid in by_unit[uid]:
                raise RuntimeError(
                    f"Duplicate chunk decision for unit {uid}, PMID {pmid}"
                )

            by_unit[uid][pmid] = status
            order_by_unit[uid].append(pmid)

    return by_unit, order_by_unit


def extract_content(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    response = row.get("response")
    if not response or response.get("status_code") != 200:
        raise RuntimeError(
            f"{row.get('custom_id')}: missing/non-200 reducer response"
        )

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"{row.get('custom_id')}: no choices")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content.strip():
        raise RuntimeError(f"{row.get('custom_id')}: empty reducer content")

    parsed = json.loads(content)
    return parsed, message


def main():
    canonical, paper_order = canonical_status_by_unit()
    rows = load_jsonl(REDUCER_OUTPUT)

    found = {
        clean(r.get("custom_id"))
        for r in rows
        if clean(r.get("custom_id")) in TARGET_CUSTOM_IDS
    }
    if found != TARGET_CUSTOM_IDS:
        raise RuntimeError(
            f"Target reducer outputs mismatch.\n"
            f"Expected: {sorted(TARGET_CUSTOM_IDS)}\n"
            f"Found:    {sorted(found)}"
        )

    if not BACKUP.exists():
        shutil.copy2(REDUCER_OUTPUT, BACKUP)
        print(f"Backup created: {BACKUP}")
    else:
        print(f"Backup already exists: {BACKUP}")

    audit = []
    patched_count = 0

    for row in rows:
        custom_id = clean(row.get("custom_id"))
        if custom_id not in TARGET_CUSTOM_IDS:
            continue

        uid = reducer_uid(custom_id)
        if uid not in canonical:
            raise RuntimeError(f"No chunk decisions found for reducer unit {uid}")

        parsed, message = extract_content(row)

        before = {
            field: [clean(x) for x in parsed.get(field, [])]
            for field in STATUS_TO_FIELD.values()
        }

        after = {field: [] for field in STATUS_TO_FIELD.values()}

        # Preserve deterministic original chunk-paper order.
        for pmid in paper_order[uid]:
            status = canonical[uid][pmid]
            after[STATUS_TO_FIELD[status]].append(pmid)

        # Hard completeness / exclusivity QC.
        flattened = []
        for field in after:
            flattened.extend(after[field])

        expected = set(canonical[uid])
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(f"{uid}: repaired partition contains duplicate PMID")
        if set(flattened) != expected:
            raise RuntimeError(
                f"{uid}: repaired partition mismatch; "
                f"missing={sorted(expected - set(flattened))}; "
                f"extra={sorted(set(flattened) - expected)}"
            )

        # Replace ONLY the four status lists. Preserve all narrative synthesis,
        # findings, readiness, implication and limitations exactly as returned.
        for field, values in after.items():
            parsed[field] = values

        message["content"] = json.dumps(parsed, ensure_ascii=False)

        changes = {}
        for field in after:
            before_set = set(before[field])
            after_set = set(after[field])
            if before[field] != after[field]:
                changes[field] = {
                    "before_count": len(before[field]),
                    "after_count": len(after[field]),
                    "added_pmids": sorted(after_set - before_set),
                    "removed_pmids": sorted(before_set - after_set),
                }

        audit.append({
            "custom_id": custom_id,
            "evidence_unit_id": uid,
            "repair_basis": (
                "Canonical per-paper status from completed Stage-A chunk appraisal"
            ),
            "fields_modified": changes,
            "paper_count": len(expected),
            "all_other_reducer_fields_preserved": True,
            "openai_api_call": False,
        })
        patched_count += 1

    if patched_count != 4:
        raise RuntimeError(f"Expected to patch 4 reducers, patched {patched_count}")

    write_jsonl(REDUCER_OUTPUT, rows)
    AUDIT.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "repair_type": "DETERMINISTIC_REDUCER_PMID_PARTITION_REPAIR",
                "patched_reducers": patched_count,
                "targets": audit,
                "principle": (
                    "Chunk appraisal is canonical for per-paper MAIN_SYNTHESIS / "
                    "CONTEXT_ONLY / APPENDIX / REJECT status. Reducer narrative "
                    "content was not regenerated or modified."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Deterministic Stage-A reducer partition repair completed.")
    print(f"  patched reducers: {patched_count}/4")
    print("  OpenAI/API calls: 0")
    print()
    for item in audit:
        print(f"  {item['custom_id']} -> unit {item['evidence_unit_id']}")
        for field, ch in item["fields_modified"].items():
            print(
                f"      {field}: {ch['before_count']} -> {ch['after_count']} "
                f"| added={ch['added_pmids']} | removed={ch['removed_pmids']}"
            )
    print()
    print(f"  patched reducer output: {REDUCER_OUTPUT}")
    print(f"  backup:                 {BACKUP}")
    print(f"  audit:                  {AUDIT}")
    print()
    print("NEXT COMMAND:")
    print(
        r"uv run python .\src\final_stageA_evidence_synthesis_gpt56.py "
        r"--mode merge-reducers --model gpt-5.6-sol"
    )


if __name__ == "__main__":
    main()
