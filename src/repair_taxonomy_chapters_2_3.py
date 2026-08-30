#!/usr/bin/env python3
"""
Repair ONLY taxonomy chapters 2 and 3 after their original GPT-5.6 Sol/high
requests exhausted the 12,000 completion-token budget on reasoning and returned
no visible JSON.

This script:
- reuses the exact original two request bodies from
  data/gpt_new_subunit_taxonomy_batch_input.jsonl
- changes only max_completion_tokens to 40,000
- submits only 2 Batch requests
- waits for completion
- verifies non-empty, parseable JSON output with finish_reason=stop
- backs up the original 8-request Batch output
- replaces ONLY taxonomy-chapter-2 and taxonomy-chapter-3 in the original output
- leaves the other six completed chapter taxonomies untouched

After success, rerun the existing local merge:
    uv run python .\src\gpt_design_new_subunit_taxonomy_batch.py --mode merge
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ORIGINAL_INPUT = DATA / "gpt_new_subunit_taxonomy_batch_input.jsonl"
ORIGINAL_OUTPUT = DATA / "gpt_new_subunit_taxonomy_batch_output.jsonl"
BACKUP_OUTPUT = DATA / "gpt_new_subunit_taxonomy_batch_output_before_ch2_ch3_repair.jsonl"

REPAIR_INPUT = DATA / "gpt_new_subunit_taxonomy_ch2_ch3_repair_input.jsonl"
REPAIR_OUTPUT = DATA / "gpt_new_subunit_taxonomy_ch2_ch3_repair_output.jsonl"
REPAIR_ERRORS = DATA / "gpt_new_subunit_taxonomy_ch2_ch3_repair_errors.jsonl"
REPAIR_STATE = DATA / "gpt_new_subunit_taxonomy_ch2_ch3_repair_state.json"
REPAIR_AUDIT = DATA / "gpt_new_subunit_taxonomy_ch2_ch3_repair_audit.json"

TARGETS = {"taxonomy-chapter-2", "taxonomy-chapter-3"}
API = "https://api.openai.com/v1"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
TRANSIENT = {408, 409, 429, 500, 502, 503, 504}


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


class Client:
    def __init__(self, key: str, retry_wait: int = 120):
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.s = requests.Session()
        self.headers = {"Authorization": f"Bearer {key}"}
        self.retry_wait = retry_wait

    def request(self, method: str, path: str, *, timeout: int = 900, **kwargs):
        url = path if path.startswith("http") else API + path
        while True:
            headers = dict(self.headers)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.s.request(
                    method, url, headers=headers, timeout=timeout, **kwargs
                )
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.ChunkedEncodingError,
                requests.ContentDecodingError,
            ) as e:
                print(
                    f"WARN: {type(e).__name__}: {e}; "
                    f"retrying in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code in TRANSIENT:
                print(
                    f"WARN: HTTP {r.status_code}; "
                    f"retrying in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {r.status_code}: {r.text[:5000]}"
                )
            return r


def load_state() -> dict[str, Any]:
    if not REPAIR_STATE.exists():
        return {}
    return json.loads(REPAIR_STATE.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    REPAIR_STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare() -> list[dict[str, Any]]:
    source = load_jsonl(ORIGINAL_INPUT)
    selected = []

    for row in source:
        if row.get("custom_id") not in TARGETS:
            continue
        repaired = json.loads(json.dumps(row))
        repaired["body"]["max_completion_tokens"] = 40000
        selected.append(repaired)

    ids = {x["custom_id"] for x in selected}
    if ids != TARGETS:
        raise RuntimeError(
            f"Expected repair requests {sorted(TARGETS)}, found {sorted(ids)}"
        )

    write_jsonl(REPAIR_INPUT, selected)

    print("Prepared targeted repair Batch.")
    print(f"  requests: {len(selected)}")
    for row in selected:
        print(
            f"  {row['custom_id']}: "
            f"model={row['body'].get('model')}, "
            f"reasoning={row['body'].get('reasoning_effort')}, "
            f"max_completion_tokens={row['body'].get('max_completion_tokens')}"
        )
    print(f"  input: {REPAIR_INPUT}")
    return selected


def submit_and_watch(client: Client) -> dict[str, Any]:
    state = load_state()

    if not state.get("batch_id"):
        with REPAIR_INPUT.open("rb") as f:
            uploaded = client.request(
                "POST",
                "/files",
                files={
                    "file": (
                        REPAIR_INPUT.name,
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
                "task": "repair_new_subunit_taxonomy_chapters_2_3",
                "max_completion_tokens": "40000",
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
        }
        save_state(state)
        print(f"Batch id: {batch['id']}")
    else:
        print(f"Resuming repair Batch: {state['batch_id']}")

    while True:
        batch = client.request(
            "GET", f"/batches/{state['batch_id']}"
        ).json()
        counts = batch.get("request_counts") or {}
        status = batch.get("status")

        print(
            f"status={status}; total={counts.get('total')}; "
            f"completed={counts.get('completed')}; failed={counts.get('failed')}"
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

        if status in TERMINAL:
            if batch.get("output_file_id"):
                REPAIR_OUTPUT.write_bytes(
                    client.request(
                        "GET",
                        f"/files/{batch['output_file_id']}/content",
                    ).content
                )
            if batch.get("error_file_id"):
                REPAIR_ERRORS.write_bytes(
                    client.request(
                        "GET",
                        f"/files/{batch['error_file_id']}/content",
                    ).content
                )
            return batch

        time.sleep(120)


def validate_repair_output() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(REPAIR_OUTPUT)

    if len(rows) != 2:
        raise RuntimeError(
            f"Expected 2 repair output rows, found {len(rows)}"
        )

    audit = {}
    seen = set()

    for row in rows:
        cid = row.get("custom_id")
        seen.add(cid)

        response = row.get("response")
        if row.get("error") or not response:
            raise RuntimeError(
                f"{cid}: Batch response error: {row.get('error')}"
            )
        if response.get("status_code") != 200:
            raise RuntimeError(
                f"{cid}: HTTP {response.get('status_code')}"
            )

        body = response["body"]
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")

        usage = body.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        print()
        print(cid)
        print(f"  finish_reason:   {finish_reason}")
        print(f"  content length:  {len(content):,}")
        print(f"  prompt tokens:   {usage.get('prompt_tokens')}")
        print(f"  completion:      {usage.get('completion_tokens')}")
        print(f"  reasoning tokens:{completion_details.get('reasoning_tokens')}")

        if finish_reason != "stop":
            raise RuntimeError(
                f"{cid}: repair still did not finish normally "
                f"(finish_reason={finish_reason})"
            )
        if not content.strip():
            raise RuntimeError(f"{cid}: repair returned empty content")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{cid}: repair output still not parseable JSON: {e}"
            ) from e

        if not parsed.get("proposed_clusters"):
            raise RuntimeError(
                f"{cid}: parsed JSON has no proposed_clusters"
            )

        audit[cid] = {
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "content_length": len(content),
            "proposed_cluster_count": len(parsed["proposed_clusters"]),
            "model": body.get("model"),
        }

    if seen != TARGETS:
        raise RuntimeError(
            f"Repair output IDs mismatch: {sorted(seen)}"
        )

    return rows, audit


def patch_original_output(repaired_rows: list[dict[str, Any]]) -> None:
    original = load_jsonl(ORIGINAL_OUTPUT)
    repair_by_id = {x["custom_id"]: x for x in repaired_rows}

    if not BACKUP_OUTPUT.exists():
        shutil.copy2(ORIGINAL_OUTPUT, BACKUP_OUTPUT)
        print(f"Backup created: {BACKUP_OUTPUT}")
    else:
        print(f"Backup already exists: {BACKUP_OUTPUT}")

    patched = []
    replacements = 0

    for row in original:
        cid = row.get("custom_id")
        if cid in repair_by_id:
            patched.append(repair_by_id[cid])
            replacements += 1
        else:
            patched.append(row)

    if replacements != 2:
        raise RuntimeError(
            f"Expected to replace 2 original responses, replaced {replacements}"
        )

    write_jsonl(ORIGINAL_OUTPUT, patched)
    print(f"Patched original output: {ORIGINAL_OUTPUT}")


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this PowerShell session."
        )

    prepare()
    client = Client(key)
    batch = submit_and_watch(client)

    if batch.get("status") != "completed":
        raise RuntimeError(
            f"Repair Batch ended with status {batch.get('status')}"
        )

    repaired_rows, audit = validate_repair_output()
    patch_original_output(repaired_rows)

    REPAIR_AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Chapters 2 and 3 taxonomy repair completed successfully.")
    print(f"Audit: {REPAIR_AUDIT}")
    print()
    print("Now rerun ONLY the existing local taxonomy merge:")
    print(
        r"uv run python .\src\gpt_design_new_subunit_taxonomy_batch.py --mode merge"
    )


if __name__ == "__main__":
    main()
