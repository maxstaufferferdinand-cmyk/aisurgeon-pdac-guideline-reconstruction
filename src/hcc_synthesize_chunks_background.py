from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from hcc_gpt_map_appraise_background import Client, OPENAI_BASE_URL, TRANSIENT_STATUS
from hcc_gpt_map_appraise_batch import MODEL
from hcc_gpt_map_appraise_direct import append_usage, atomic_write_json, read_json, response_text, utc_now
from hcc_synthesize_units import (
    base_instructions,
    chunk_prompt,
    chunk_schema,
    chunks,
    evidence_for_synthesis,
    read_jsonl,
)


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def state_path(hcc_root: Path) -> Path:
    return hcc_root / "run_state" / "openai_stageA_chunk_synthesis_background_state.json"


def load_state(hcc_root: Path) -> dict[str, Any]:
    path = state_path(hcc_root)
    return read_json(path) if path.exists() else {"chunks": []}


def save_state(hcc_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_path(hcc_root), state)


def prepare(hcc_root: Path, chunk_size: int) -> dict[str, Any]:
    state = load_state(hcc_root)
    if state.get("chunks"):
        return state
    out = hcc_root / "data" / "hcc_unit_synthesis"
    master = read_jsonl(hcc_root / "data" / "guideline_integration_master_v2.jsonl")
    records: list[dict[str, Any]] = []
    for unit in master:
        evidence_rows = evidence_for_synthesis(unit)
        for index, group in enumerate(chunks(evidence_rows, chunk_size)):
            chunk_id = f"{unit['evidence_unit_id']}_chunk_{index:04d}"
            status = "completed" if (out / f"{chunk_id}_parsed.json").exists() else "prepared"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "unit_id": unit["evidence_unit_id"],
                    "pmids": [str(row.get("pmid")) for row in group],
                    "row_count": len(group),
                    "status": status,
                }
            )
        if not evidence_rows:
            chunk_id = f"{unit['evidence_unit_id']}_chunk_0000"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "unit_id": unit["evidence_unit_id"],
                    "pmids": [],
                    "row_count": 0,
                    "status": "completed" if (out / f"{chunk_id}_parsed.json").exists() else "prepared_empty",
                }
            )
    state = {
        "created_at": utc_now(),
        "chunk_size": chunk_size,
        "chunks": records,
    }
    save_state(hcc_root, state)
    print(json.dumps({"prepared_chunks": len(records), "already_completed": sum(1 for r in records if r["status"] == "completed")}, sort_keys=True))
    return state


def submit_ready(hcc_root: Path, client: Client, model: str, max_in_flight: int, max_output_tokens: int) -> None:
    state = load_state(hcc_root)
    out = hcc_root / "data" / "hcc_unit_synthesis"
    master = {row["evidence_unit_id"]: row for row in read_jsonl(hcc_root / "data" / "guideline_integration_master_v2.jsonl")}
    rows_by_unit = {uid: evidence_for_synthesis(unit) for uid, unit in master.items()}
    active = sum(1 for c in state["chunks"] if c.get("status") in {"queued", "in_progress", "submitted"})
    for chunk in state["chunks"]:
        if active >= max_in_flight:
            break
        if chunk.get("response_id") or chunk.get("status") not in {"prepared", "retry"}:
            continue
        parsed_path = out / f"{chunk['chunk_id']}_parsed.json"
        if parsed_path.exists():
            chunk["status"] = "completed"
            save_state(hcc_root, state)
            continue
        unit = master[chunk["unit_id"]]
        all_rows = rows_by_unit[chunk["unit_id"]]
        index = int(chunk["chunk_id"].rsplit("_", 1)[1])
        group = chunks(all_rows, state["chunk_size"])[index]
        body = {
            "model": model,
            "background": True,
            "instructions": base_instructions(),
            "input": chunk_prompt(unit, chunk["chunk_id"], group),
            "text": {"format": chunk_schema()},
            "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "project": "ESMO_HCC_2012_to_2025",
                "phase": "stageA_chunk_synthesis_background",
                "chunk": chunk["chunk_id"],
                "unit_id": chunk["unit_id"],
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
                "phase": "hcc_stageA_chunk_synthesis_background_submit",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "response_id": response.get("id"),
                "chunk_id": chunk["chunk_id"],
                "usage": response.get("usage", {}),
            },
        )
        active += 1
        save_state(hcc_root, state)


def parse_completed(hcc_root: Path, chunk: dict[str, Any], response: dict[str, Any]) -> None:
    out = hcc_root / "data" / "hcc_unit_synthesis"
    atomic_write_json(out / f"{chunk['chunk_id']}_raw_response.json", response)
    parsed = json.loads(response_text(response))
    parsed["unit_id"] = chunk["unit_id"]
    parsed["chunk_id"] = chunk["chunk_id"]
    atomic_write_json(out / f"{chunk['chunk_id']}_parsed.json", parsed)


def poll(hcc_root: Path, client: Client, model: str) -> dict[str, int]:
    state = load_state(hcc_root)
    counts: dict[str, int] = {}
    for chunk in state["chunks"]:
        status = chunk.get("status", "prepared")
        if status in {"completed", "failed", "cancelled"}:
            counts[status] = counts.get(status, 0) + 1
            continue
        if status == "prepared_empty":
            chunk["status"] = "completed"
            counts["completed"] = counts.get("completed", 0) + 1
            save_state(hcc_root, state)
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
                "phase": "hcc_stageA_chunk_synthesis_background_poll",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "response_id": response_id,
                "chunk_id": chunk["chunk_id"],
                "status": status,
                "usage": response.get("usage", {}),
            },
        )
        if status == "completed":
            parse_completed(hcc_root, chunk, response)
            chunk["parsed_at"] = utc_now()
        elif status in {"failed", "cancelled", "incomplete"}:
            chunk["error"] = response.get("error") or response.get("incomplete_details")
        counts[chunk["status"]] = counts.get(chunk["status"], 0) + 1
        save_state(hcc_root, state)
    return counts


def cancel_queued_to_retry(hcc_root: Path, client: Client) -> dict[str, int]:
    state = load_state(hcc_root)
    reset_count = 0
    cancel_attempts = 0
    cancel_successes = 0
    cancel_failures = 0
    for chunk in state.get("chunks", []):
        if chunk.get("status") != "queued":
            continue
        record: dict[str, Any] = {"attempted_at": utc_now(), "response_id": chunk.get("response_id")}
        response_id = chunk.get("response_id")
        if response_id:
            cancel_attempts += 1
            try:
                response = client.request("POST", f"/responses/{response_id}/cancel", timeout=120)
                record.update({"http_status": 200, "result_status": response.get("status"), "request_id": response.get("id")})
                cancel_successes += 1
            except RuntimeError as exc:
                record.update({"http_status": "error", "error_category": str(exc)[:500]})
                cancel_failures += 1
        chunk.setdefault("cancel_history", []).append(record)
        chunk["status"] = "retry"
        chunk["reset_at"] = utc_now()
        chunk["previous_response_id"] = chunk.pop("response_id", None)
        reset_count += 1
    if reset_count:
        save_state(hcc_root, state)
    return {
        "queued_chunks_reset_to_retry": reset_count,
        "cancel_attempts": cancel_attempts,
        "cancel_successes": cancel_successes,
        "cancel_failures": cancel_failures,
    }


def run(args: argparse.Namespace) -> None:
    hcc_root = Path(args.hcc_root)
    prepare(hcc_root, args.chunk_size)
    client = Client(os.environ.get("OPENAI_API_KEY", "").strip(), args.retry_wait)
    if args.cancel_queued_to_retry:
        print(json.dumps(cancel_queued_to_retry(hcc_root, client), sort_keys=True))
    while True:
        submit_ready(hcc_root, client, args.model, args.max_in_flight, args.max_output_tokens)
        counts = poll(hcc_root, client, args.model)
        print(json.dumps({"timestamp": utc_now(), "counts": counts}, sort_keys=True))
        if args.poll_only:
            return
        if counts.get("completed", 0) == sum(counts.values()):
            return
        if counts.get("failed", 0) or counts.get("cancelled", 0):
            raise RuntimeError(f"Stage-A chunk synthesis background failure: {counts}")
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Background runner for HCC Stage-A chunk synthesis.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--max-in-flight", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--poll-only", action="store_true", help="Poll once and exit after any requested queue reset.")
    parser.add_argument(
        "--cancel-queued-to-retry",
        action="store_true",
        help="Cancel currently queued background responses and reset only those chunks to retry.",
    )
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
