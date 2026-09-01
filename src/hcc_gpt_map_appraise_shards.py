from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from hcc_gpt_map_appraise_batch import (
    MODEL,
    TERMINAL_BATCH_STATUSES,
    OpenAIHTTP,
    append_usage,
    atomic_write_json,
    merge as merge_combined_output,
    output_dir,
    read_json,
    utc_now,
)


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def state_path(hcc_root: Path) -> Path:
    return hcc_root / "run_state" / "openai_mapping_appraisal_shards_state.json"


def load_state(hcc_root: Path) -> dict[str, Any]:
    path = state_path(hcc_root)
    return read_json(path) if path.exists() else {"shards": []}


def save_state(hcc_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_path(hcc_root), state)


def split_shards(hcc_root: Path, shard_size: int) -> dict[str, Any]:
    out = output_dir(hcc_root)
    source = out / "mapping_appraisal_batch_input.jsonl"
    if not source.exists():
        raise RuntimeError("Prepared mapping_appraisal_batch_input.jsonl is missing.")
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(hcc_root)
    if state.get("shards"):
        print(json.dumps({"status": "shards_already_defined", "shards": len(state["shards"])}, indent=2))
        return state
    shards: list[dict[str, Any]] = []
    handle = None
    try:
        for idx, line in enumerate(source.open(encoding="utf-8")):
            shard_index = idx // shard_size
            if idx % shard_size == 0:
                if handle is not None:
                    handle.close()
                shard_path = shard_dir / f"mapping_appraisal_shard_{shard_index:03d}.jsonl"
                handle = shard_path.open("w", encoding="utf-8", newline="\n")
                shards.append({"index": shard_index, "path": str(shard_path), "status": "prepared"})
            handle.write(line)
    finally:
        if handle is not None:
            handle.close()
    for shard in shards:
        shard["bytes"] = Path(shard["path"]).stat().st_size
    state["created_at"] = utc_now()
    state["shards"] = shards
    save_state(hcc_root, state)
    print(json.dumps({"status": "prepared", "shards": len(shards), "shard_size": shard_size}, indent=2))
    return state


def submit_shards(hcc_root: Path, client: OpenAIHTTP, model: str) -> dict[str, Any]:
    state = load_state(hcc_root)
    for shard in state.get("shards", []):
        if shard.get("batch_id") and shard.get("status") not in {"failed", "expired", "cancelled"}:
            continue
        path = Path(shard["path"])
        if not shard.get("input_file_id"):
            with path.open("rb") as handle:
                file_response = client.request(
                    "POST",
                    "/files",
                    files={"file": (path.name, handle, "application/jsonl")},
                    data={"purpose": "batch"},
                    timeout=300,
                ).json()
            shard["input_file_id"] = file_response["id"]
        batch = client.request(
            "POST",
            "/batches",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "input_file_id": shard["input_file_id"],
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "metadata": {
                        "project": "ESMO_HCC_2012_to_2025",
                        "task": "hcc_mapping_appraisal_shard",
                        "shard_index": str(shard["index"]),
                        "model_requested": model,
                        "local_cost_mode": "PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION",
                    },
                }
            ),
        ).json()
        shard.update(
            {
                "batch_id": batch["id"],
                "status": batch.get("status"),
                "created_at": batch.get("created_at"),
                "request_counts": batch.get("request_counts"),
                "usage": batch.get("usage"),
            }
        )
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_mapping_appraisal_shard_submit",
                "model": model,
                "request_timestamp": utc_now(),
                "batch_id": batch["id"],
                "input_file_id": shard["input_file_id"],
                "shard_index": shard["index"],
                "status": batch.get("status"),
            },
        )
        save_state(hcc_root, state)
        print(json.dumps({"submitted_shard": shard["index"], "batch_id": batch["id"], "status": batch.get("status")}))
    return state


def download(client: OpenAIHTTP, file_id: str, destination: Path) -> None:
    if destination.exists():
        return
    response = client.request("GET", f"/files/{file_id}/content", timeout=600)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def watch_shards(hcc_root: Path, client: OpenAIHTTP, poll_seconds: int, model: str) -> dict[str, Any]:
    state = load_state(hcc_root)
    out = output_dir(hcc_root)
    while True:
        pending = 0
        statuses: dict[str, int] = {}
        for shard in state.get("shards", []):
            status = shard.get("status")
            if status in TERMINAL_BATCH_STATUSES and shard.get("output_downloaded"):
                statuses[status] = statuses.get(status, 0) + 1
                continue
            batch_id = shard.get("batch_id")
            if not batch_id:
                pending += 1
                continue
            batch = client.request("GET", f"/batches/{batch_id}").json()
            status = batch.get("status")
            shard.update(
                {
                    "status": status,
                    "request_counts": batch.get("request_counts"),
                    "usage": batch.get("usage"),
                    "output_file_id": batch.get("output_file_id"),
                    "error_file_id": batch.get("error_file_id"),
                    "last_polled_at": utc_now(),
                }
            )
            statuses[status] = statuses.get(status, 0) + 1
            append_usage(
                hcc_root,
                {
                    "provider": "openai",
                    "phase": "hcc_mapping_appraisal_shard_poll",
                    "model": model,
                    "request_timestamp": utc_now(),
                    "batch_id": batch_id,
                    "shard_index": shard["index"],
                    "status": status,
                    "request_counts": batch.get("request_counts"),
                    "usage": batch.get("usage"),
                },
            )
            if status in TERMINAL_BATCH_STATUSES:
                if batch.get("output_file_id"):
                    download(
                        client,
                        batch["output_file_id"],
                        out / "shards" / f"mapping_appraisal_shard_{shard['index']:03d}_output.jsonl",
                    )
                    shard["output_downloaded"] = True
                if batch.get("error_file_id"):
                    download(
                        client,
                        batch["error_file_id"],
                        out / "shards" / f"mapping_appraisal_shard_{shard['index']:03d}_errors.jsonl",
                    )
                    shard["error_downloaded"] = True
        save_state(hcc_root, state)
        print(json.dumps({"timestamp": utc_now(), "statuses": statuses}, sort_keys=True))
        if not pending and all(shard.get("status") in TERMINAL_BATCH_STATUSES for shard in state.get("shards", [])):
            return state
        time.sleep(poll_seconds)


def combine_outputs(hcc_root: Path, model: str) -> dict[str, Any]:
    state = load_state(hcc_root)
    failed = [shard for shard in state.get("shards", []) if shard.get("status") != "completed"]
    if failed:
        raise RuntimeError(f"{len(failed)} shard batches did not complete.")
    out = output_dir(hcc_root)
    combined = out / "mapping_appraisal_batch_output.jsonl"
    with combined.open("w", encoding="utf-8", newline="\n") as dest:
        for shard in sorted(state["shards"], key=lambda item: item["index"]):
            shard_output = out / "shards" / f"mapping_appraisal_shard_{shard['index']:03d}_output.jsonl"
            if not shard_output.exists():
                raise RuntimeError(f"Missing shard output: {shard_output}")
            with shard_output.open(encoding="utf-8") as src:
                for line in src:
                    dest.write(line)
    return merge_combined_output(hcc_root, model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard and run HCC mapping/appraisal OpenAI Batch jobs.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--mode", choices=["split", "submit", "watch", "merge", "all"], default="all")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--shard-size", type=int, default=1500)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    if args.mode in {"split", "all"}:
        split_shards(hcc_root, args.shard_size)
        if args.mode == "split":
            return 0
    client = None
    if args.mode in {"submit", "watch", "all"}:
        client = OpenAIHTTP(os.environ.get("OPENAI_API_KEY", "").strip())
    if args.mode in {"submit", "all"}:
        submit_shards(hcc_root, client, args.model)
        if args.mode == "submit":
            return 0
    if args.mode in {"watch", "all"}:
        state = watch_shards(hcc_root, client, args.poll_seconds, args.model)
        if any(shard.get("status") != "completed" for shard in state.get("shards", [])):
            return 2
        if args.mode == "watch":
            return 0
    if args.mode in {"merge", "all"}:
        summary = combine_outputs(hcc_root, args.model)
        return 0 if summary["missing_or_failed_pmids"] == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
