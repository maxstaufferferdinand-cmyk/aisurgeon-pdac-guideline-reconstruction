from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from hcc_gpt_map_appraise_batch import MODEL
from hcc_gpt_map_appraise_direct import (
    OpenAIResponses,
    append_usage,
    atomic_write_json,
    read_json,
    response_text,
    utc_now,
)


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} line {line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def chunk_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "hcc_unit_chunk_synthesis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "string"},
                "chunk_id": {"type": "string"},
                "main_findings": {"type": "string"},
                "context_findings": {"type": "string"},
                "appendix_findings": {"type": "string"},
                "practice_implications": {"type": "string"},
                "evidence_limitations": {"type": "string"},
                "cited_pmids": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": [
                "unit_id",
                "chunk_id",
                "main_findings",
                "context_findings",
                "appendix_findings",
                "practice_implications",
                "evidence_limitations",
                "cited_pmids",
                "notes",
            ],
            "additionalProperties": False,
        },
    }


def reducer_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "hcc_unit_final_synthesis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "string"},
                "evidence_update": {"type": "string"},
                "current_clinical_practice": {"type": "string"},
                "evidence_limitations": {"type": "string"},
                "change_signal": {
                    "type": "string",
                    "enum": ["CONFIRM", "MODIFY", "ADD", "REMOVE", "INSUFFICIENT_EVIDENCE"],
                },
                "recommendation_change_supported": {"type": "boolean"},
                "synthesis_rationale": {"type": "string"},
                "key_pmids": {"type": "array", "items": {"type": "string"}},
                "appendix_summary": {"type": "string"},
            },
            "required": [
                "unit_id",
                "evidence_update",
                "current_clinical_practice",
                "evidence_limitations",
                "change_signal",
                "recommendation_change_supported",
                "synthesis_rationale",
                "key_pmids",
                "appendix_summary",
            ],
            "additionalProperties": False,
        },
    }


def base_instructions() -> str:
    return (
        "You are performing evidence synthesis for a blinded scientific reconstruction of the "
        "ESMO 2012 hepatocellular carcinoma guideline through 2025-02-28. Do not use later "
        "ESMO HCC guidelines, the human 2025 benchmark, or web knowledge. Use only the supplied "
        "source-context and PubMed abstract-level evidence.\n\n"
        "Apply the project hierarchy exactly: Tier 1 meta-analysis of human randomized trials; "
        "Tier 2 meta-analysis of retrospective, non-randomized or mixed human studies; Tier 3 "
        "systematic review; Tier 4 other review or standalone randomized controlled trial. A "
        "clinically relevant standalone RCT is Tier 4 but can support recommendations. Other "
        "reviews may inform context but must not independently drive a recommendation change. "
        "Do not let APPENDIX or REJECT evidence drive current clinical practice. Use "
        "domain-appropriate endpoints for diagnostic, prognostic, epidemiologic, surveillance, "
        "and follow-up topics."
    )


def record_block(row: dict[str, Any]) -> str:
    abstract = clean(row.get("abstract"))
    if len(abstract) > 950:
        abstract = abstract[:950].rsplit(" ", 1)[0] + " ..."
    return "\n".join(
        [
            f"PMID: {clean(row.get('pmid'))}",
            f"Status: {clean(row.get('appraisal_status'))}",
            f"Tier: {clean(row.get('tier'))}",
            f"Design: {clean(row.get('study_design'))}",
            f"Can support recommendation change: {clean(row.get('can_support_recommendation_change'))}",
            f"Endpoint strength: {clean(row.get('endpoint_strength'))}",
            f"Title: {clean(row.get('title'))}",
            f"Publication: {clean(row.get('journal'))} {clean(row.get('pub_year'))}",
            f"Publication types: {clean(row.get('publication_types'))}",
            f"Abstract: {abstract or '[no abstract available]'}",
            f"Mapping rationale: {clean(row.get('mapping_rationale'))}",
        ]
    )


def source_context_text(unit: dict[str, Any]) -> str:
    blocks = []
    for item in unit.get("source_context", []):
        blocks.append(
            f"{clean(item.get('item_type'))} {clean(item.get('id'))}: "
            f"{clean(item.get('text'))} Citations: {item.get('citation_numbers', [])}; "
            f"grades/levels: {item.get('grades_or_levels', [])}"
        )
    return "\n".join(blocks) or "[No source context extracted for this unit.]"


def chunk_prompt(unit: dict[str, Any], chunk_id: str, rows: list[dict[str, Any]]) -> str:
    return (
        f"Evidence unit: {unit['evidence_unit_id']} - {unit['evidence_unit_title']}\n"
        f"Source chapter: {unit['chapter_title']}\n\n"
        "Relevant 2012 source context:\n"
        f"{source_context_text(unit)}\n\n"
        "Synthesize this technical evidence chunk. Preserve PMID references in the cited_pmids list. "
        "Focus on clinically usable evidence and identify APPENDIX-only material separately.\n\n"
        f"Chunk ID: {chunk_id}\n\n"
        + "\n\n--- PUBMED RECORD ---\n\n".join(record_block(row) for row in rows)
    )


def reducer_prompt(unit: dict[str, Any], summaries: list[dict[str, Any]], deterministic: dict[str, Any]) -> str:
    compact_summaries = [
        {
            "chunk_id": s["chunk_id"],
            "main_findings": s["main_findings"],
            "context_findings": s["context_findings"],
            "appendix_findings": s["appendix_findings"],
            "practice_implications": s["practice_implications"],
            "evidence_limitations": s["evidence_limitations"],
            "cited_pmids": s["cited_pmids"],
        }
        for s in summaries
    ]
    return (
        f"Evidence unit: {unit['evidence_unit_id']} - {unit['evidence_unit_title']}\n"
        f"Source chapter: {unit['chapter_title']}\n\n"
        "Relevant 2012 source context:\n"
        f"{source_context_text(unit)}\n\n"
        "Deterministic appraisal counts and PMID partitions:\n"
        f"{json.dumps(deterministic, ensure_ascii=False, sort_keys=True)}\n\n"
        "Reduce the chunk syntheses into one concise evidence-unit memo. Use the deterministic "
        "PMID partitions; do not cite or rely on rejected records. Choose the internal change "
        "signal relative to the 2012 source context.\n\n"
        f"Chunk syntheses:\n{json.dumps(compact_summaries, ensure_ascii=False)}"
    )


def call_openai(
    hcc_root: Path,
    client: OpenAIResponses,
    model: str,
    phase: str,
    request_id: str,
    prompt: str,
    json_schema: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "instructions": base_instructions(),
        "input": prompt,
        "text": {"format": json_schema},
        "reasoning": {"effort": "high"},
        "max_output_tokens": max_output_tokens,
        "metadata": {
            "project": "ESMO_HCC_2012_to_2025",
            "phase": phase,
            "request_id": request_id,
        },
    }
    response = client.create(body)
    append_usage(
        hcc_root,
        {
            "provider": "openai",
            "phase": phase,
            "model": response.get("model", model),
            "request_timestamp": utc_now(),
            "started_at": response.get("_request_started_at"),
            "completed_at": response.get("_request_completed_at"),
            "attempts": response.get("_retry_attempts", 0) + 1,
            "request_id": request_id,
            "usage": response.get("usage", {}),
        },
    )
    return response


def evidence_for_synthesis(unit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row for row in unit.get("mapped_evidence", [])
        if clean(row.get("appraisal_status")) in {"MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX"}
    ]
    def key(row: dict[str, Any]) -> tuple[int, int, str, str]:
        status_rank = {"MAIN_SYNTHESIS": 0, "CONTEXT_ONLY": 1, "APPENDIX": 2}.get(clean(row.get("appraisal_status")), 3)
        tier_rank = {"TIER_1": 0, "TIER_2": 1, "TIER_3": 2, "TIER_4": 3}.get(clean(row.get("tier")), 4)
        return (status_rank, tier_rank, clean(row.get("pub_year")), clean(row.get("pmid")))
    return sorted(rows, key=key)


def deterministic_partitions(unit: dict[str, Any]) -> dict[str, Any]:
    rows = unit.get("mapped_evidence", [])
    by_status: dict[str, list[str]] = {}
    for status in ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"]:
        by_status[status] = sorted(
            {clean(row.get("pmid")) for row in rows if clean(row.get("appraisal_status")) == status and clean(row.get("pmid"))},
            key=int,
        )
    return {
        "status_counts": dict(Counter(clean(row.get("appraisal_status")) for row in rows)),
        "tier_counts": dict(Counter(clean(row.get("tier")) for row in rows)),
        "main_synthesis_pmids": by_status["MAIN_SYNTHESIS"],
        "context_only_pmids": by_status["CONTEXT_ONLY"],
        "appendix_pmids": by_status["APPENDIX"],
        "reject_pmids": by_status["REJECT"],
    }


def synthesize_chunks(
    hcc_root: Path,
    client: OpenAIResponses,
    model: str,
    unit: dict[str, Any],
    rows: list[dict[str, Any]],
    chunk_size: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    out = hcc_root / "data" / "hcc_unit_synthesis"
    unit_id = unit["evidence_unit_id"]
    if not rows:
        empty = {
            "unit_id": unit_id,
            "chunk_id": f"{unit_id}_chunk_0000",
            "main_findings": "No MAIN_SYNTHESIS, CONTEXT_ONLY, or APPENDIX records were mapped to this evidence unit.",
            "context_findings": "",
            "appendix_findings": "",
            "practice_implications": "No evidence update can be synthesized from the mapped corpus.",
            "evidence_limitations": "The mapped evidence set is empty.",
            "cited_pmids": [],
            "notes": "Deterministic empty-unit placeholder.",
        }
        atomic_write_json(out / f"{unit_id}_chunk_0000_parsed.json", empty)
        return [empty]
    summaries: list[dict[str, Any]] = []
    for index, group in enumerate(chunks(rows, chunk_size)):
        chunk_id = f"{unit_id}_chunk_{index:04d}"
        parsed_path = out / f"{chunk_id}_parsed.json"
        raw_path = out / f"{chunk_id}_raw_response.json"
        if parsed_path.exists():
            summaries.append(read_json(parsed_path))
            continue
        prompt = chunk_prompt(unit, chunk_id, group)
        response = call_openai(
            hcc_root,
            client,
            model,
            "hcc_unit_evidence_chunk_synthesis",
            chunk_id,
            prompt,
            chunk_schema(),
            max_output_tokens,
        )
        atomic_write_json(raw_path, response)
        parsed = json.loads(response_text(response))
        parsed["unit_id"] = unit_id
        parsed["chunk_id"] = chunk_id
        atomic_write_json(parsed_path, parsed)
        summaries.append(parsed)
    return summaries


def synthesize_unit(
    hcc_root: Path,
    client: OpenAIResponses,
    model: str,
    unit: dict[str, Any],
    chunk_size: int,
    chunk_max_output_tokens: int,
    reducer_max_output_tokens: int,
) -> dict[str, Any]:
    out = hcc_root / "data" / "hcc_unit_synthesis"
    unit_id = unit["evidence_unit_id"]
    final_path = out / f"{unit_id}_final_parsed.json"
    if final_path.exists():
        return read_json(final_path)
    evidence_rows = evidence_for_synthesis(unit)
    summaries = synthesize_chunks(hcc_root, client, model, unit, evidence_rows, chunk_size, chunk_max_output_tokens)
    deterministic = deterministic_partitions(unit)
    prompt = reducer_prompt(unit, summaries, deterministic)
    response = call_openai(
        hcc_root,
        client,
        model,
        "hcc_unit_evidence_reducer_synthesis",
        f"{unit_id}_final",
        prompt,
        reducer_schema(),
        reducer_max_output_tokens,
    )
    atomic_write_json(out / f"{unit_id}_final_raw_response.json", response)
    parsed = json.loads(response_text(response))
    parsed["unit_id"] = unit_id
    parsed["evidence_unit_title"] = unit["evidence_unit_title"]
    parsed["chapter_id"] = unit["chapter_id"]
    parsed["chapter_title"] = unit["chapter_title"]
    parsed["main_synthesis_pmids"] = deterministic["main_synthesis_pmids"]
    parsed["context_only_pmids"] = deterministic["context_only_pmids"]
    parsed["appendix_pmids"] = deterministic["appendix_pmids"]
    parsed["reject_pmids"] = deterministic["reject_pmids"]
    parsed["status_counts"] = deterministic["status_counts"]
    parsed["tier_counts"] = deterministic["tier_counts"]
    parsed["chunk_count"] = len(summaries)
    atomic_write_json(final_path, parsed)
    return parsed


def merge(hcc_root: Path) -> dict[str, Any]:
    out = hcc_root / "data" / "hcc_unit_synthesis"
    master = read_jsonl(hcc_root / "data" / "guideline_integration_master_v2.jsonl")
    results = []
    missing = []
    for unit in master:
        path = out / f"{unit['evidence_unit_id']}_final_parsed.json"
        if not path.exists():
            missing.append(unit["evidence_unit_id"])
            continue
        results.append(read_json(path))
    results.sort(key=lambda r: (clean(r.get("chapter_id")), clean(r.get("unit_id"))))
    write_jsonl(hcc_root / "data" / "stageA_unit_evidence_synthesis.jsonl", results)
    status_counts = Counter(clean(row.get("change_signal")) for row in results)
    assignment_status_counts = Counter()
    for row in results:
        for status, count in row.get("status_counts", {}).items():
            assignment_status_counts[status] += count
    manifest = {
        "created_at": utc_now(),
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "unit_count": len(master),
        "completed_unit_count": len(results),
        "missing_units": missing,
        "change_signal_counts": dict(status_counts),
        "appraisal_assignment_status_counts": dict(assignment_status_counts),
        "output": str(hcc_root / "data" / "stageA_unit_evidence_synthesis.jsonl"),
    }
    atomic_write_json(hcc_root / "data" / "stageA_evidence_synthesis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run(args: argparse.Namespace) -> int:
    hcc_root = Path(args.hcc_root)
    master = read_jsonl(hcc_root / "data" / "guideline_integration_master_v2.jsonl")
    client = OpenAIResponses(os.environ.get("OPENAI_API_KEY", "").strip(), args.retry_wait)
    selected_units = set(args.unit_id or [])
    for unit in master:
        if selected_units and unit["evidence_unit_id"] not in selected_units:
            continue
        result = synthesize_unit(
            hcc_root,
            client,
            args.model,
            unit,
            args.chunk_size,
            args.chunk_max_output_tokens,
            args.reducer_max_output_tokens,
        )
        print(json.dumps({"unit_id": unit["evidence_unit_id"], "change_signal": result.get("change_signal")}, sort_keys=True))
    manifest = merge(hcc_root)
    return 0 if manifest["status"] == "COMPLETE" or selected_units else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="HCC final Stage-A unit evidence synthesis.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--mode", choices=["run", "merge", "all"], default="all")
    parser.add_argument("--unit-id", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=70)
    parser.add_argument("--chunk-max-output-tokens", type=int, default=9000)
    parser.add_argument("--reducer-max-output-tokens", type=int, default=9000)
    parser.add_argument("--retry-wait", type=int, default=120)
    args = parser.parse_args()
    if args.mode == "merge":
        manifest = merge(Path(args.hcc_root))
        return 0 if manifest["status"] == "COMPLETE" else 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
