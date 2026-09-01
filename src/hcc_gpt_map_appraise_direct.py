from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from hcc_gpt_map_appraise_batch import MODEL, clean, ontology_units


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
OPENAI_BASE_URL = "https://api.openai.com/v1"
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV header in {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def append_usage(hcc_root: Path, entry: dict[str, Any]) -> None:
    ledger_path = hcc_root / "run_state" / "cost_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read_json(ledger_path) if ledger_path.exists() else {"requests": []}
        ledger.setdefault("requests", []).append(entry)
        atomic_write_json(ledger_path, ledger)
        fcntl.flock(lock, fcntl.LOCK_UN)


def schema(unit_ids: list[str]) -> dict[str, Any]:
    assignment = {
        "type": "object",
        "properties": {
            "unit_id": {"type": "string", "enum": unit_ids},
            "role": {"type": "string", "enum": ["PRIMARY", "SECONDARY"]},
            "appraisal_status": {"type": "string", "enum": ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"]},
            "tier": {"type": "string", "enum": ["TIER_1", "TIER_2", "TIER_3", "TIER_4", "NOT_APPLICABLE"]},
            "study_design": {
                "type": "string",
                "enum": [
                    "META_ANALYSIS_OF_RANDOMIZED_TRIALS",
                    "META_ANALYSIS_OTHER_OR_MIXED",
                    "SYSTEMATIC_REVIEW",
                    "OTHER_REVIEW",
                    "RANDOMIZED_CONTROLLED_TRIAL",
                    "OTHER",
                ],
            },
            "human_clinical_relevance": {"type": "boolean"},
            "population_directness": {"type": "string", "enum": ["direct", "partial", "indirect", "unclear"]},
            "endpoint_strength": {"type": "string", "enum": ["hard_or_domain_appropriate", "surrogate_only", "contextual", "unclear"]},
            "can_support_guideline_narrative": {"type": "boolean"},
            "can_support_recommendation_change": {"type": "boolean"},
            "rejection_reason": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "unit_id",
            "role",
            "appraisal_status",
            "tier",
            "study_design",
            "human_clinical_relevance",
            "population_directness",
            "endpoint_strength",
            "can_support_guideline_narrative",
            "can_support_recommendation_change",
            "rejection_reason",
            "rationale",
        ],
        "additionalProperties": False,
    }
    result = {
        "type": "object",
        "properties": {
            "pmid": {"type": "string"},
            "out_of_scope": {"type": "boolean"},
            "questionable_assignment": {"type": "boolean"},
            "novel_topic": {"type": "boolean"},
            "novel_topic_label": {"type": "string"},
            "broad_review": {"type": "boolean"},
            "assignments": {"type": "array", "minItems": 0, "maxItems": 4, "items": assignment},
            "overall_rationale": {"type": "string"},
        },
        "required": [
            "pmid",
            "out_of_scope",
            "questionable_assignment",
            "novel_topic",
            "novel_topic_label",
            "broad_review",
            "assignments",
            "overall_rationale",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "hcc_mapping_appraisal_chunk",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {"type": "array", "items": result},
                "chunk_notes": {"type": "string"},
            },
            "required": ["results", "chunk_notes"],
            "additionalProperties": False,
        },
    }


def instructions(units: dict[str, str]) -> str:
    return (
        "You are mapping and appraising PubMed evidence for a blinded scientific reconstruction of the "
        "ESMO 2012 hepatocellular carcinoma guideline through 2025-02-28. Do not use later ESMO HCC "
        "guidelines or web knowledge. Use only the supplied records and the source-derived ontology.\n\n"
        "Return exactly one result for every supplied PMID. Assign a primary evidence unit and at most "
        "three secondary units unless the record is out of scope. Mark guideline/consensus-like records "
        "as REJECT if any survived deterministic filtering. Apply this project hierarchy exactly: Tier 1 "
        "meta-analysis of human randomized trials; Tier 2 meta-analysis of retrospective, non-randomized "
        "or mixed human studies; Tier 3 systematic review; Tier 4 other review or standalone randomized "
        "controlled trial. A clinically relevant standalone RCT is Tier 4 but may support recommendations; "
        "other reviews may inform context but must not alone support recommendation changes. Use "
        "domain-appropriate clinical endpoints for diagnostic, prognostic, epidemiologic, surveillance, and "
        "follow-up evidence.\n\n"
        "Evidence units:\n"
        + "\n".join(f"- {unit_id}: {title}" for unit_id, title in units.items())
    )


def make_user_content(rows: list[dict[str, str]]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"PMID: {clean(row.get('pmid'))}",
                    f"Title: {clean(row.get('title')) or '[missing]'}",
                    f"Abstract: {clean(row.get('abstract')) or '[no abstract available]'}",
                    f"Publication types: {clean(row.get('publication_types')) or '[not available]'}",
                    f"MeSH terms: {clean(row.get('mesh_terms')) or '[not available]'}",
                    f"Publication year: {clean(row.get('pub_year')) or '[not available]'}",
                    f"Deterministic evidence type: {clean(row.get('evidence_type')) or '[not available]'}",
                    f"Search provenance units: {clean(row.get('provenance_unit_ids')) or '[not available]'}",
                ]
            )
        )
    return "\n\n--- RECORD ---\n\n".join(blocks)


class OpenAIResponses:
    def __init__(self, api_key: str, retry_wait: int) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.retry_wait = retry_wait

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            started = utc_now()
            try:
                response = self.session.post(
                    f"{OPENAI_BASE_URL}/responses",
                    headers=self.headers,
                    data=json.dumps(body),
                    timeout=300,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                time.sleep(self.retry_wait)
                continue
            if response.status_code in TRANSIENT_STATUS:
                text = response.text.lower()
                if response.status_code == 429 and any(term in text for term in ("billing", "quota", "usage limit")):
                    raise RuntimeError(f"OpenAI provider quota/usage-limit rejection: HTTP {response.status_code}")
                time.sleep(self.retry_wait)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"OpenAI HTTP {response.status_code}: {response.text[:2000]}")
            data = response.json()
            data["_request_started_at"] = started
            data["_request_completed_at"] = utc_now()
            data["_retry_attempts"] = attempt - 1
            return data


def response_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    if not texts:
        raise RuntimeError("Responses output did not contain output_text.")
    return "\n".join(texts)


def chunks(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def completed_pmids_from_outputs(out: Path) -> set[str]:
    completed: set[str] = set()
    for path in sorted(out.glob("*_parsed.json")):
        try:
            data = read_json(path)
        except json.JSONDecodeError:
            continue
        for result in data.get("results", []):
            pmid = str(result.get("pmid", ""))
            if pmid:
                completed.add(pmid)
    return completed


def next_chunk_index(out: Path, prefix: str) -> int:
    max_index = -1
    for path in out.glob(f"{prefix}_*_parsed.json"):
        stem = path.name.removesuffix("_parsed.json")
        maybe = stem.rsplit("_", 1)[-1]
        if maybe.isdigit():
            max_index = max(max_index, int(maybe))
    return max_index + 1


def run_chunks(
    hcc_root: Path,
    model: str,
    chunk_size: int,
    retry_wait: int,
    max_output_tokens: int,
    worker_index: int,
    worker_count: int,
) -> dict[str, Any]:
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    out.mkdir(parents=True, exist_ok=True)
    state_name = (
        "openai_mapping_appraisal_direct_state.json"
        if worker_count == 1
        else f"openai_mapping_appraisal_direct_state_worker_{worker_index}_of_{worker_count}.json"
    )
    state_path = hcc_root / "run_state" / state_name
    state = read_json(state_path) if state_path.exists() else {"completed_chunks": {}, "failed_chunks": {}}
    ontology = read_json(hcc_root / "data" / "ontology_v1.json")
    units = ontology_units(ontology)
    rows, _fields = load_csv(hcc_root / "data" / "selected_evidence_v2.csv")
    completed_pmids = completed_pmids_from_outputs(out)
    rows = [
        row
        for original_index, row in enumerate(rows)
        if original_index % worker_count == worker_index and row["pmid"] not in completed_pmids
    ]
    chunk_prefix = "chunk" if worker_count == 1 else f"w{worker_index}"
    chunk_index_offset = next_chunk_index(out, chunk_prefix)
    client = OpenAIResponses(os.environ.get("OPENAI_API_KEY", "").strip(), retry_wait)
    all_chunks = chunks(rows, chunk_size)
    for index, chunk in enumerate(all_chunks):
        chunk_key = f"{chunk_prefix}_{index + chunk_index_offset:04d}"
        if chunk_key in state.get("completed_chunks", {}):
            continue
        body = {
            "model": model,
            "instructions": instructions(units),
            "input": make_user_content(chunk),
            "text": {"format": schema(list(units))},
            "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "project": "ESMO_HCC_2012_to_2025",
                "phase": "mapping_appraisal_direct",
                "chunk": chunk_key,
                "worker_index": str(worker_index),
                "worker_count": str(worker_count),
            },
        }
        response = client.create(body)
        raw_path = out / f"{chunk_key}_raw_response.json"
        atomic_write_json(raw_path, response)
        usage = response.get("usage", {})
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_direct",
                "model": response.get("model", model),
                "request_timestamp": response.get("_request_started_at"),
                "response_timestamp": response.get("_request_completed_at"),
                "response_id": response.get("id"),
                "chunk_id": chunk_key,
                "retry_count": response.get("_retry_attempts", 0),
                "usage": usage,
                "status": response.get("status"),
            },
        )
        try:
            parsed = json.loads(response_text(response))
        except Exception as exc:  # noqa: BLE001
            state.setdefault("failed_chunks", {})[chunk_key] = {"error": f"parse: {type(exc).__name__}: {exc}", "updated_at": utc_now()}
            atomic_write_json(state_path, state)
            raise
        expected = {row["pmid"] for row in chunk}
        observed = {str(item.get("pmid", "")) for item in parsed.get("results", [])}
        missing = sorted(expected - observed, key=int)
        extra = sorted(observed - expected)
        if missing and len(missing) == len(extra):
            results = parsed.get("results", [])
            for wrong, correct in zip(extra, missing, strict=True):
                for item in results:
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
            state.setdefault("failed_chunks", {})[chunk_key] = {"missing": missing, "extra": extra, "updated_at": utc_now()}
            atomic_write_json(out / f"{chunk_key}_parsed_with_coverage_error.json", parsed)
            atomic_write_json(state_path, state)
            raise RuntimeError(f"Chunk {chunk_key} PMID coverage mismatch: missing={len(missing)} extra={len(extra)}")
        atomic_write_json(out / f"{chunk_key}_parsed.json", parsed)
        state.setdefault("completed_chunks", {})[chunk_key] = {
            "pmids": sorted(expected, key=int),
            "response_id": response.get("id"),
            "usage": usage,
            "completed_at": utc_now(),
        }
        state.get("failed_chunks", {}).pop(chunk_key, None)
        atomic_write_json(state_path, state)
        print(json.dumps({"completed_chunk": chunk_key, "completed": len(state["completed_chunks"]), "total_chunks": len(all_chunks)}))
    summary = {
        "status": "complete",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "chunks": len(all_chunks),
        "completed_chunks": len(state.get("completed_chunks", {})),
    }
    atomic_write_json(out / "direct_mapping_appraisal_run_summary.json", summary)
    return summary


def merge_outputs(hcc_root: Path, model: str) -> dict[str, Any]:
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    selected, selected_fields = load_csv(hcc_root / "data" / "selected_evidence_v2.csv")
    parsed_by_pmid: dict[str, dict[str, Any]] = {}
    for path in sorted(out.glob("*_parsed.json")):
        data = read_json(path)
        for result in data.get("results", []):
            parsed_by_pmid[str(result["pmid"])] = result
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in selected:
        pmid = row["pmid"]
        result = parsed_by_pmid.get(pmid)
        out_row = dict(row)
        if result is None:
            missing.append(pmid)
            out_row["gpt_parse_status"] = "MISSING"
        else:
            result_assignments = result.get("assignments", [])
            out_row.update(
                {
                    "gpt_parse_status": "OK",
                    "gpt_out_of_scope": result.get("out_of_scope", ""),
                    "gpt_questionable_assignment": result.get("questionable_assignment", ""),
                    "gpt_novel_topic": result.get("novel_topic", ""),
                    "gpt_novel_topic_label": result.get("novel_topic_label", ""),
                    "gpt_broad_review": result.get("broad_review", ""),
                    "gpt_primary_unit_id": next((a.get("unit_id") for a in result_assignments if a.get("role") == "PRIMARY"), ""),
                    "gpt_all_unit_ids": "|".join(a.get("unit_id", "") for a in result_assignments),
                    "gpt_overall_rationale": result.get("overall_rationale", ""),
                    "gpt_model": model,
                }
            )
            for assignment in result_assignments:
                assignments.append(
                    {
                        "pmid": pmid,
                        "unit_id": assignment.get("unit_id", ""),
                        "role": assignment.get("role", ""),
                        "appraisal_status": assignment.get("appraisal_status", ""),
                        "tier": assignment.get("tier", ""),
                        "study_design": assignment.get("study_design", ""),
                        "human_clinical_relevance": assignment.get("human_clinical_relevance", ""),
                        "population_directness": assignment.get("population_directness", ""),
                        "endpoint_strength": assignment.get("endpoint_strength", ""),
                        "can_support_guideline_narrative": assignment.get("can_support_guideline_narrative", ""),
                        "can_support_recommendation_change": assignment.get("can_support_recommendation_change", ""),
                        "rejection_reason": assignment.get("rejection_reason", ""),
                        "rationale": assignment.get("rationale", ""),
                        "model": model,
                    }
                )
        rows.append(out_row)
    extra_fields = [
        "gpt_parse_status",
        "gpt_out_of_scope",
        "gpt_questionable_assignment",
        "gpt_novel_topic",
        "gpt_novel_topic_label",
        "gpt_broad_review",
        "gpt_primary_unit_id",
        "gpt_all_unit_ids",
        "gpt_overall_rationale",
        "gpt_model",
    ]
    write_csv(out / "selected_evidence_mapping_appraisal_merged.csv", rows, selected_fields + extra_fields)
    write_csv(
        out / "pmid_unit_appraisals.csv",
        assignments,
        [
            "pmid",
            "unit_id",
            "role",
            "appraisal_status",
            "tier",
            "study_design",
            "human_clinical_relevance",
            "population_directness",
            "endpoint_strength",
            "can_support_guideline_narrative",
            "can_support_recommendation_change",
            "rejection_reason",
            "rationale",
            "model",
        ],
    )
    summary = {
        "created_at": utc_now(),
        "selected_pmids": len(selected),
        "parsed_pmids": len(parsed_by_pmid),
        "missing_pmids": len(missing),
        "assignment_rows": len(assignments),
        "main_synthesis_assignments": sum(1 for row in assignments if row["appraisal_status"] == "MAIN_SYNTHESIS"),
        "context_only_assignments": sum(1 for row in assignments if row["appraisal_status"] == "CONTEXT_ONLY"),
        "appendix_assignments": sum(1 for row in assignments if row["appraisal_status"] == "APPENDIX"),
        "reject_assignments": sum(1 for row in assignments if row["appraisal_status"] == "REJECT"),
    }
    atomic_write_json(out / "direct_mapping_appraisal_merge_qc.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable direct OpenAI Responses fallback for HCC mapping/appraisal.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--mode", choices=["run", "merge", "all"], default="all")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=18000)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    if args.mode in {"run", "all"}:
        if args.worker_index < 0 or args.worker_index >= args.worker_count:
            raise RuntimeError("--worker-index must be in [0, --worker-count).")
        summary = run_chunks(
            hcc_root,
            args.model,
            args.chunk_size,
            args.retry_wait,
            args.max_output_tokens,
            args.worker_index,
            args.worker_count,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.mode == "run":
            return 0
    if args.mode in {"merge", "all"}:
        summary = merge_outputs(hcc_root, args.model)
        return 0 if summary["missing_pmids"] == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
