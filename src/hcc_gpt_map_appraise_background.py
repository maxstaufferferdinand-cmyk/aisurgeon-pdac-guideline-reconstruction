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

from hcc_gpt_map_appraise_batch import MODEL, clean, ontology_units
from hcc_gpt_map_appraise_direct import (
    completed_pmids_from_outputs,
    instructions,
    make_user_content,
    response_text,
    schema,
)


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_usage(hcc_root: Path, entry: dict[str, Any]) -> None:
    ledger_path = hcc_root / "run_state" / "cost_ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"requests": []}
    ledger.setdefault("requests", []).append(entry)
    atomic_write_json(ledger_path, ledger)


class Client:
    def __init__(self, api_key: str, retry_wait: int) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.retry_wait = retry_wait

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        while True:
            try:
                response = self.session.request(
                    method,
                    OPENAI_BASE_URL + path,
                    headers=self.headers,
                    timeout=kwargs.pop("timeout", 120),
                    **kwargs,
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
            return response.json()


def chunks(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def state_path(hcc_root: Path) -> Path:
    return hcc_root / "run_state" / "openai_mapping_appraisal_background_state.json"


def load_state(hcc_root: Path) -> dict[str, Any]:
    path = state_path(hcc_root)
    return read_json(path) if path.exists() else {"chunks": []}


def save_state(hcc_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_path(hcc_root), state)


def prepare_chunks(hcc_root: Path, chunk_size: int) -> dict[str, Any]:
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    state = load_state(hcc_root)
    if state.get("chunks"):
        return state
    completed = completed_pmids_from_outputs(out)
    selected = load_csv(hcc_root / "data" / "selected_evidence_v2.csv")
    remaining = [row for row in selected if row["pmid"] not in completed]
    chunk_records = []
    for index, chunk in enumerate(chunks(remaining, chunk_size)):
        chunk_records.append(
            {
                "chunk_id": f"bg_{index:04d}",
                "pmids": [row["pmid"] for row in chunk],
                "status": "prepared",
            }
        )
    state = {
        "created_at": utc_now(),
        "chunk_size": chunk_size,
        "already_completed_pmids_at_prepare": len(completed),
        "chunks": chunk_records,
    }
    save_state(hcc_root, state)
    print(json.dumps({"prepared_background_chunks": len(chunk_records), "remaining_pmids": len(remaining)}, indent=2))
    return state


def submit_ready(hcc_root: Path, client: Client, model: str, max_in_flight: int, max_output_tokens: int) -> None:
    state = load_state(hcc_root)
    ontology = read_json(hcc_root / "data" / "ontology_v1.json")
    units = ontology_units(ontology)
    selected_by_pmid = {row["pmid"]: row for row in load_csv(hcc_root / "data" / "selected_evidence_v2.csv")}
    in_flight = sum(1 for chunk in state["chunks"] if chunk.get("status") in {"queued", "in_progress", "submitted"})
    for chunk in state["chunks"]:
        if in_flight >= max_in_flight:
            break
        if chunk.get("response_id") or chunk.get("status") not in {"prepared", "retry"}:
            continue
        rows = [selected_by_pmid[pmid] for pmid in chunk["pmids"]]
        body = {
            "model": model,
            "background": True,
            "instructions": instructions(units),
            "input": make_user_content(rows),
            "text": {"format": schema(list(units))},
            "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "project": "ESMO_HCC_2012_to_2025",
                "phase": "mapping_appraisal_background",
                "chunk": chunk["chunk_id"],
            },
        }
        response = client.request("POST", "/responses", data=json.dumps(body), timeout=180)
        chunk.update(
            {
                "response_id": response.get("id"),
                "status": response.get("status"),
                "submitted_at": utc_now(),
                "model": response.get("model", model),
            }
        )
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_background_submit",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "response_id": response.get("id"),
                "chunk_id": chunk["chunk_id"],
                "status": response.get("status"),
                "usage": response.get("usage", {}),
            },
        )
        in_flight += 1
        save_state(hcc_root, state)


def reset_failed_to_retry(hcc_root: Path) -> dict[str, int]:
    state = load_state(hcc_root)
    reset_count = 0
    for chunk in state.get("chunks", []):
        if chunk.get("status") != "failed":
            continue
        chunk["status"] = "retry"
        chunk["reset_at"] = utc_now()
        chunk["previous_response_id"] = chunk.pop("response_id", None)
        chunk["previous_error"] = chunk.pop("error", None)
        reset_count += 1
    if reset_count:
        save_state(hcc_root, state)
    return {"failed_chunks_reset_to_retry": reset_count}


def parse_completed(hcc_root: Path, chunk: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    out = hcc_root / "data" / "gpt_mapping_appraisal_direct"
    raw_path = out / f"{chunk['chunk_id']}_raw_response.json"
    parsed_path = out / f"{chunk['chunk_id']}_parsed.json"
    atomic_write_json(raw_path, response)
    parsed = json.loads(response_text(response))
    expected = set(chunk["pmids"])
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
        atomic_write_json(out / f"{chunk['chunk_id']}_parsed_with_coverage_error.json", parsed)
        return {"ok": False, "missing": missing, "extra": extra}
    atomic_write_json(parsed_path, parsed)
    return {"ok": True, "missing": [], "extra": []}


def poll(hcc_root: Path, client: Client, model: str) -> dict[str, int]:
    state = load_state(hcc_root)
    counts: dict[str, int] = {}
    for chunk in state["chunks"]:
        status = chunk.get("status", "prepared")
        if status in {"completed", "failed", "cancelled"}:
            counts[status] = counts.get(status, 0) + 1
            continue
        response_id = chunk.get("response_id")
        if not response_id:
            counts[status] = counts.get(status, 0) + 1
            continue
        response = client.request("GET", f"/responses/{response_id}", timeout=120)
        status = response.get("status")
        chunk["status"] = status
        chunk["last_polled_at"] = utc_now()
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_background_poll",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "response_id": response_id,
                "chunk_id": chunk["chunk_id"],
                "status": status,
                "usage": response.get("usage", {}),
            },
        )
        if status == "completed":
            coverage = parse_completed(hcc_root, chunk, response)
            if coverage["ok"]:
                chunk["parsed_at"] = utc_now()
            else:
                chunk["status"] = "coverage_error"
                chunk["coverage_error"] = coverage
        elif status in {"failed", "cancelled", "incomplete"}:
            chunk["error"] = response.get("error") or response.get("incomplete_details")
        counts[chunk["status"]] = counts.get(chunk["status"], 0) + 1
        save_state(hcc_root, state)
    return counts


def run(
    hcc_root: Path,
    model: str,
    chunk_size: int,
    max_in_flight: int,
    poll_seconds: int,
    max_output_tokens: int,
    retry_wait: int,
    poll_only: bool,
    reset_failed: bool,
) -> None:
    prepare_chunks(hcc_root, chunk_size)
    if reset_failed:
        print(json.dumps(reset_failed_to_retry(hcc_root), sort_keys=True))
    client = Client(os.environ.get("OPENAI_API_KEY", "").strip(), retry_wait)
    while True:
        if not poll_only:
            submit_ready(hcc_root, client, model, max_in_flight, max_output_tokens)
        counts = poll(hcc_root, client, model)
        print(json.dumps({"timestamp": utc_now(), "counts": counts}, sort_keys=True))
        if poll_only:
            return
        if counts.get("completed", 0) + counts.get("coverage_error", 0) == sum(counts.values()):
            return
        if counts.get("failed", 0) or counts.get("cancelled", 0):
            raise RuntimeError(f"Background response failure: {counts}")
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Background Responses runner for HCC mapping/appraisal chunks.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--max-in-flight", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=30000)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument(
        "--poll-only",
        action="store_true",
        help="Poll already-submitted response IDs once without submitting prepared/retry chunks.",
    )
    parser.add_argument(
        "--reset-failed-to-retry",
        action="store_true",
        help="Explicitly reset failed chunks to retry after a provider/account problem has been fixed.",
    )
    args = parser.parse_args()
    run(
        Path(args.hcc_root),
        args.model,
        args.chunk_size,
        args.max_in_flight,
        args.poll_seconds,
        args.max_output_tokens,
        args.retry_wait,
        args.poll_only,
        args.reset_failed_to_retry,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
