from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from hcc_gpt_map_appraise_batch import MODEL, ontology_units
from hcc_gpt_map_appraise_direct import (
    OpenAIResponses,
    atomic_write_json,
    completed_pmids_from_outputs,
    instructions,
    load_csv,
    make_user_content,
    merge_outputs,
    read_json,
    response_text,
    schema,
    utc_now,
)


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def salvage_coverage_errors(hcc_root: Path) -> dict[str, Any]:
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    state_path = hcc_root / "run_state" / "openai_mapping_appraisal_background_state.json"
    if not state_path.exists():
        return {"coverage_error_chunks": 0, "salvaged_pmids": 0}
    state = read_json(state_path)
    salvaged = 0
    chunks = 0
    for chunk in state.get("chunks", []):
        if chunk.get("status") != "coverage_error":
            continue
        chunks += 1
        chunk_id = chunk["chunk_id"]
        expected = {str(pmid) for pmid in chunk.get("pmids", [])}
        missing = {str(pmid) for pmid in chunk.get("coverage_error", {}).get("missing", [])}
        bad_path = out / f"{chunk_id}_parsed_with_coverage_error.json"
        salvage_path = out / f"{chunk_id}_salvaged_parsed.json"
        if not bad_path.exists() or salvage_path.exists():
            continue
        parsed = read_json(bad_path)
        good_results = [
            item for item in parsed.get("results", [])
            if str(item.get("pmid", "")) in expected and str(item.get("pmid", "")) not in missing
        ]
        if good_results:
            atomic_write_json(
                salvage_path,
                {
                    "results": good_results,
                    "chunk_notes": (
                        "Salvaged deterministic subset from a coverage-error response. "
                        "Missing PMIDs are repaired separately."
                    ),
                },
            )
            salvaged += len(good_results)
    return {"coverage_error_chunks": chunks, "salvaged_pmids": salvaged}


def missing_pmids(hcc_root: Path) -> list[str]:
    selected = load_csv(hcc_root / "data" / "selected_evidence_v2.csv")
    completed = completed_pmids_from_outputs(hcc_root / "data" / "gpt_mapping_appraisal_direct")
    return [row["pmid"] for row in selected if row["pmid"] not in completed]


def repair_missing(
    hcc_root: Path,
    model: str,
    pmids: list[str],
    chunk_size: int,
    retry_wait: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    if not pmids:
        return {"repair_chunks": 0, "repair_pmids": 0}
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    selected_by_pmid = {row["pmid"]: row for row in load_csv(hcc_root / "data" / "selected_evidence_v2.csv")}
    units = ontology_units(read_json(hcc_root / "data" / "ontology_v1.json"))
    client = OpenAIResponses(os.environ.get("OPENAI_API_KEY", "").strip(), retry_wait)
    completed_chunks = 0
    completed_pmids = 0
    for index, group in enumerate(chunked(pmids, chunk_size)):
        prefix = f"repair_{index:04d}"
        parsed_path = out / f"{prefix}_parsed.json"
        if parsed_path.exists():
            continue
        rows = [selected_by_pmid[pmid] for pmid in group]
        body = {
            "model": model,
            "instructions": instructions(units),
            "input": make_user_content(rows),
            "text": {"format": schema(list(units))},
            "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "project": "ESMO_HCC_2012_to_2025",
                "phase": "mapping_appraisal_gap_repair",
                "chunk": prefix,
            },
        }
        response = client.create(body)
        atomic_write_json(out / f"{prefix}_raw_response.json", response)
        parsed = json.loads(response_text(response))
        expected = set(group)
        observed = {str(item.get("pmid", "")) for item in parsed.get("results", [])}
        missing = sorted(expected - observed, key=int)
        extra = sorted(observed - expected)
        if missing and len(missing) == len(extra):
            for wrong, correct in zip(extra, missing, strict=True):
                for item in parsed.get("results", []):
                    if str(item.get("pmid", "")) == wrong:
                        item["pmid"] = correct
                        item["overall_rationale"] = (
                            item.get("overall_rationale", "")
                            + f" [Deterministic PMID transcription repair: model returned {wrong}, requested PMID was {correct}.]"
                        ).strip()
                        break
            observed = {str(item.get("pmid", "")) for item in parsed.get("results", [])}
            missing = sorted(expected - observed, key=int)
            extra = sorted(observed - expected)
        if missing or extra:
            atomic_write_json(out / f"{prefix}_parsed_with_coverage_error.json", parsed)
            raise RuntimeError(f"Repair chunk {prefix} coverage mismatch: missing={missing} extra={extra}")
        atomic_write_json(parsed_path, parsed)
        completed_chunks += 1
        completed_pmids += len(group)
        from hcc_gpt_map_appraise_direct import append_usage

        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_gap_repair",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "started_at": response.get("_request_started_at"),
                "completed_at": response.get("_request_completed_at"),
                "attempts": response.get("_retry_attempts", 0) + 1,
                "chunk_id": prefix,
                "usage": response.get("usage", {}),
            },
        )
    return {"repair_chunks": completed_chunks, "repair_pmids": completed_pmids}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair missing HCC mapping/appraisal PMIDs after background runs.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    salvage = salvage_coverage_errors(hcc_root)
    missing = missing_pmids(hcc_root)
    repair = repair_missing(hcc_root, args.model, missing, args.chunk_size, args.retry_wait, args.max_output_tokens)
    merge = merge_outputs(hcc_root, args.model)
    summary = {
        "created_at": utc_now(),
        "salvage": salvage,
        "missing_before_repair": len(missing),
        "repair": repair,
        "merge": merge,
    }
    atomic_write_json(hcc_root / "data" / "gpt_mapping_appraisal_direct" / "gap_repair_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if merge["missing_pmids"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
