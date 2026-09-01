from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
OPENAI_BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-5.6-sol"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def clean(value: str | None) -> str:
    return " ".join((value or "").replace("\ufeff", "").split())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_usage(hcc_root: Path, entry: dict[str, Any]) -> None:
    ledger_path = hcc_root / "run_state" / "cost_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"requests": [], "phases": []}
    ledger.setdefault("requests", []).append(entry)
    atomic_write_json(ledger_path, ledger)


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


def ontology_units(ontology: dict[str, Any]) -> dict[str, str]:
    units: dict[str, str] = {}
    for chapter in ontology["chapters"]:
        for unit in chapter["evidence_units"]:
            units[unit["unit_id"]] = f"{chapter['title']} / {unit['title']}"
    return units


def response_schema(unit_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "hcc_mapping_appraisal",
        "description": "HCC guideline evidence mapping and project-specific appraisal for one PMID.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pmid": {"type": "string"},
                "out_of_scope": {"type": "boolean"},
                "questionable_assignment": {"type": "boolean"},
                "novel_topic": {"type": "boolean"},
                "novel_topic_label": {"type": "string"},
                "broad_review": {"type": "boolean"},
                "assignments": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "unit_id": {"type": "string", "enum": unit_ids},
                            "role": {"type": "string", "enum": ["PRIMARY", "SECONDARY"]},
                            "appraisal_status": {
                                "type": "string",
                                "enum": ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"],
                            },
                            "tier": {
                                "type": "string",
                                "enum": ["TIER_1", "TIER_2", "TIER_3", "TIER_4", "NOT_APPLICABLE"],
                            },
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
                            "endpoint_strength": {
                                "type": "string",
                                "enum": ["hard_or_domain_appropriate", "surrogate_only", "contextual", "unclear"],
                            },
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
                    },
                },
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
        },
    }


def system_prompt(units: dict[str, str]) -> str:
    return (
        "You are mapping and appraising PubMed evidence for a blinded scientific reconstruction of the "
        "ESMO 2012 hepatocellular carcinoma guideline through 2025-02-28. Do not use later ESMO HCC "
        "guidelines or web knowledge. Use only the provided title, abstract, publication type, MeSH terms, "
        "and the source-derived HCC ontology below.\n\n"
        "Assign a primary evidence unit and at most three secondary units. Mark out_of_scope=true if the "
        "record is not human HCC clinical evidence. A clinically relevant standalone RCT is Tier 4 but may "
        "support a recommendation change. Other reviews may provide context but must not alone support a "
        "recommendation change. Apply this hierarchy exactly: Tier 1 meta-analysis of human randomized "
        "trials; Tier 2 meta-analysis of human retrospective, non-randomized or mixed studies; Tier 3 "
        "systematic review; Tier 4 other review or standalone randomized controlled trial.\n\n"
        "For diagnostic, prognostic, epidemiologic, surveillance, and follow-up topics, use domain-appropriate "
        "clinical endpoints rather than therapeutic-RCT standards. Do not invent data absent from title/abstract. "
        "If the abstract is missing, map conservatively from title/publication type/MeSH and mark uncertainty.\n\n"
        "Evidence units:\n"
        + "\n".join(f"- {unit_id}: {title}" for unit_id, title in units.items())
    )


def prepare(hcc_root: Path, model: str, reasoning_effort: str) -> dict[str, Any]:
    input_path = hcc_root / "data" / "selected_evidence_v2.csv"
    ontology = read_json(hcc_root / "data" / "ontology_v1.json")
    units = ontology_units(ontology)
    rows, fields = load_csv(input_path)
    out_dir = hcc_root / "data" / "gpt_mapping_appraisal"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_input = out_dir / "mapping_appraisal_input.csv"
    batch_input = out_dir / "mapping_appraisal_batch_input.jsonl"
    write_csv(mapping_input, rows, fields)
    schema = response_schema(list(units))
    prompt = system_prompt(units)
    with batch_input.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            pmid = clean(row.get("pmid"))
            article = (
                f"PMID: {pmid}\n"
                f"Title: {clean(row.get('title')) or '[missing]'}\n"
                f"Abstract: {clean(row.get('abstract')) or '[no abstract available]'}\n"
                f"Publication types: {clean(row.get('publication_types')) or '[not available]'}\n"
                f"MeSH terms: {clean(row.get('mesh_terms')) or '[not available]'}\n"
                f"Publication year: {clean(row.get('pub_year')) or '[not available]'}\n"
                f"Deterministic evidence type: {clean(row.get('evidence_type')) or '[not available]'}\n"
                f"Search provenance units: {clean(row.get('provenance_unit_ids')) or '[not available]'}"
            )
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": article},
                ],
                "response_format": {"type": "json_schema", "json_schema": schema},
                "max_completion_tokens": 1800,
                "reasoning_effort": reasoning_effort,
            }
            handle.write(
                json.dumps(
                    {
                        "custom_id": f"pmid-{pmid}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    size = batch_input.stat().st_size
    if len(rows) > 50_000:
        raise RuntimeError("OpenAI Batch request limit exceeded.")
    if size > 200 * 1024 * 1024:
        raise RuntimeError(f"OpenAI Batch input too large: {size / 1024 / 1024:.1f} MB")
    summary = {
        "created_at": utc_now(),
        "input_rows": len(rows),
        "batch_jsonl_mb": round(size / 1024 / 1024, 2),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "mapping_input": str(mapping_input),
        "batch_input": str(batch_input),
    }
    atomic_write_json(out_dir / "mapping_appraisal_prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


class OpenAIHTTP:
    def __init__(self, api_key: str, retry_wait: int = 120):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.retry_wait = retry_wait

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.session.request(
                    method,
                    OPENAI_BASE_URL + path,
                    headers={**self.headers, **kwargs.pop("headers", {})},
                    timeout=kwargs.pop("timeout", 180),
                    **kwargs,
                )
            except (requests.Timeout, requests.ConnectionError, requests.ChunkedEncodingError):
                time.sleep(self.retry_wait)
                continue
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                body = response.text.lower()
                if response.status_code == 429 and any(term in body for term in ("billing", "quota", "usage limit")):
                    raise RuntimeError(f"OpenAI provider quota/usage-limit rejection: HTTP {response.status_code}")
                time.sleep(self.retry_wait)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"OpenAI HTTP {response.status_code}: {response.text[:2000]}")
            return response


def state_path(hcc_root: Path) -> Path:
    return hcc_root / "run_state" / "openai_mapping_appraisal_batch_state.json"


def output_dir(hcc_root: Path) -> Path:
    return hcc_root / "data" / "gpt_mapping_appraisal"


def load_state(hcc_root: Path) -> dict[str, Any]:
    path = state_path(hcc_root)
    return read_json(path) if path.exists() else {}


def save_state(hcc_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_path(hcc_root), state)


def submit(hcc_root: Path, client: OpenAIHTTP, model: str) -> dict[str, Any]:
    state = load_state(hcc_root)
    prior_attempts = state.get("prior_attempts", [])
    reusable_input_file_id = None
    if (
        state.get("batch_id")
        and state.get("status") == "failed"
        and (state.get("request_counts") or {}).get("total", 0) == 0
        and state.get("input_file_id")
    ):
        prior_attempts.append(dict(state))
        reusable_input_file_id = state["input_file_id"]
    elif state.get("batch_id"):
        print(json.dumps({"status": "resume_existing", "batch_id": state["batch_id"]}, indent=2))
        return state
    batch_input = output_dir(hcc_root) / "mapping_appraisal_batch_input.jsonl"
    if reusable_input_file_id:
        file_response = {"id": reusable_input_file_id}
    else:
        with batch_input.open("rb") as handle:
            file_response = client.request(
                "POST",
                "/files",
                files={"file": (batch_input.name, handle, "application/jsonl")},
                data={"purpose": "batch"},
            ).json()
    batch = client.request(
        "POST",
        "/batches",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "input_file_id": file_response["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {
                    "project": "ESMO_HCC_2012_to_2025",
                    "task": "hcc_mapping_appraisal",
                    "model_requested": model,
                    "local_cost_mode": "PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION",
                },
            }
        ),
    ).json()
    state = {
        "input_file_id": file_response["id"],
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "created_at": batch.get("created_at"),
        "model_requested": model,
        "prior_attempts": prior_attempts,
    }
    save_state(hcc_root, state)
    append_usage(
        hcc_root,
        {
            "provider": "openai",
            "phase": "hcc_mapping_appraisal_batch_submit",
            "model": model,
            "request_timestamp": utc_now(),
            "batch_id": batch["id"],
            "input_file_id": file_response["id"],
            "status": batch.get("status"),
        },
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return state


def download(client: OpenAIHTTP, file_id: str, destination: Path) -> None:
    response = client.request("GET", f"/files/{file_id}/content", timeout=600)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def watch(hcc_root: Path, client: OpenAIHTTP, poll_seconds: int) -> dict[str, Any]:
    state = load_state(hcc_root)
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No OpenAI batch state found.")
    last_status = None
    while True:
        batch = client.request("GET", f"/batches/{batch_id}").json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}
        state.update(
            {
                "status": status,
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "request_counts": counts,
                "usage": batch.get("usage"),
                "last_polled_at": utc_now(),
            }
        )
        save_state(hcc_root, state)
        if status != last_status:
            print(json.dumps({"batch_id": batch_id, "status": status, "request_counts": counts}, indent=2))
            last_status = status
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_batch_poll",
                "model": state.get("model_requested"),
                "request_timestamp": utc_now(),
                "batch_id": batch_id,
                "status": status,
                "request_counts": counts,
                "usage": batch.get("usage"),
            },
        )
        if status in TERMINAL_BATCH_STATUSES:
            out = output_dir(hcc_root)
            if batch.get("output_file_id"):
                download(client, batch["output_file_id"], out / "mapping_appraisal_batch_output.jsonl")
            if batch.get("error_file_id"):
                download(client, batch["error_file_id"], out / "mapping_appraisal_batch_errors.jsonl")
            return batch
        time.sleep(poll_seconds)


def parse_result_line(obj: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None, dict[str, Any] | None]:
    pmid = (obj.get("custom_id") or "").removeprefix("pmid-")
    response = obj.get("response")
    error = obj.get("error")
    if error or not response:
        return pmid, None, "missing response", None
    body = response.get("body", {})
    usage = body.get("usage")
    if response.get("status_code") != 200:
        return pmid, None, f"HTTP {response.get('status_code')}", usage
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        return pmid, None, f"parse error: {type(exc).__name__}: {exc}", usage
    return pmid, parsed, None, usage


def merge(hcc_root: Path, model: str) -> dict[str, Any]:
    out = output_dir(hcc_root)
    output_path = out / "mapping_appraisal_batch_output.jsonl"
    if not output_path.exists():
        raise RuntimeError(f"Missing batch output: {output_path}")
    input_rows, input_fields = load_csv(out / "mapping_appraisal_input.csv")
    parsed_by_pmid: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {}
    with output_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pmid, parsed, error, usage = parse_result_line(obj)
            if usage:
                for key, value in usage.items():
                    if isinstance(value, int):
                        usage_totals[key] = usage_totals.get(key, 0) + value
            if error or parsed is None:
                failures.append({"line": line_no, "pmid": pmid, "error": error})
                continue
            parsed_by_pmid[pmid] = parsed

    rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    state = load_state(hcc_root)
    batch_id = state.get("batch_id", "")
    for row in input_rows:
        pmid = row["pmid"]
        parsed = parsed_by_pmid.get(pmid)
        out_row = dict(row)
        if parsed is None:
            missing.append(pmid)
            out_row.update({"gpt_parse_status": "MISSING"})
        else:
            assignments = parsed.get("assignments", [])
            out_row.update(
                {
                    "gpt_parse_status": "OK",
                    "gpt_out_of_scope": str(parsed.get("out_of_scope", "")),
                    "gpt_questionable_assignment": str(parsed.get("questionable_assignment", "")),
                    "gpt_novel_topic": str(parsed.get("novel_topic", "")),
                    "gpt_novel_topic_label": parsed.get("novel_topic_label", ""),
                    "gpt_broad_review": str(parsed.get("broad_review", "")),
                    "gpt_primary_unit_id": next((a.get("unit_id") for a in assignments if a.get("role") == "PRIMARY"), ""),
                    "gpt_all_unit_ids": "|".join(a.get("unit_id", "") for a in assignments),
                    "gpt_overall_rationale": parsed.get("overall_rationale", ""),
                    "gpt_model": model,
                    "gpt_batch_id": batch_id,
                }
            )
            for assignment in assignments:
                assignment_rows.append(
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
                        "batch_id": batch_id,
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
        "gpt_batch_id",
    ]
    write_csv(out / "selected_evidence_mapping_appraisal_merged.csv", rows, input_fields + extra_fields)
    write_csv(
        out / "pmid_unit_appraisals.csv",
        assignment_rows,
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
            "batch_id",
        ],
    )
    failure_path = out / "mapping_appraisal_parse_failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")
    summary = {
        "created_at": utc_now(),
        "input_rows": len(input_rows),
        "parsed_pmids": len(parsed_by_pmid),
        "missing_or_failed_pmids": len(set(missing) | {f["pmid"] for f in failures}),
        "assignment_rows": len(assignment_rows),
        "usage_totals": usage_totals,
        "batch_id": batch_id,
        "model": model,
    }
    atomic_write_json(out / "mapping_appraisal_merge_qc.json", summary)
    append_usage(
        hcc_root,
        {
            "provider": "openai",
            "phase": "hcc_mapping_appraisal_batch_merge",
            "model": model,
            "request_timestamp": utc_now(),
            "batch_id": batch_id,
            "usage_totals": usage_totals,
            "status": "merged",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI Batch mapping and appraisal for HCC selected evidence.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--mode", choices=["prepare", "submit", "watch", "merge", "all"], default="all")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--retry-wait", type=int, default=120)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    if args.mode in {"prepare", "all"}:
        prepare(hcc_root, args.model, args.reasoning_effort)
        if args.mode == "prepare":
            return 0
    client = None
    if args.mode in {"submit", "watch", "all"}:
        client = OpenAIHTTP(os.environ.get("OPENAI_API_KEY", "").strip(), retry_wait=args.retry_wait)
    if args.mode in {"submit", "all"}:
        submit(hcc_root, client, args.model)
        if args.mode == "submit":
            return 0
    if args.mode in {"watch", "all"}:
        batch = watch(hcc_root, client, args.poll_seconds)
        if batch.get("status") != "completed":
            print(json.dumps({"status": batch.get("status"), "message": "batch not completed"}, indent=2))
            return 2
        if args.mode == "watch":
            return 0
    if args.mode in {"merge", "all"}:
        summary = merge(hcc_root, args.model)
        return 0 if summary["missing_or_failed_pmids"] == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
