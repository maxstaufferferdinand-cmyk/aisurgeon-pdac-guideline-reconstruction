#!/usr/bin/env python3
"""
STEP 2D2 — Resolve the 9 NEW_MAJOR_CHAPTER_CANDIDATE records.

Decision already fixed by manual review:
    ACCEPTED_NEW_MAJOR_CHAPTERS = 0

This script does NOT allow GPT to create a new major chapter.
It maps each of the 9 records into one or more existing major chapters and
then into either:
    - an existing evidence unit from evidence_unit_ontology.json, or
    - an accepted proposed NEW_SUBUNIT from new_subunit_cluster_taxonomy_v1.json.

The nine records remain fully auditable.

Inputs
------
data/recovery_new_major_chapter_candidates.jsonl
data/evidence_unit_ontology.json
data/new_subunit_cluster_taxonomy_v1.json

Outputs
-------
data/gpt_new_major_candidate_resolution_batch_input.jsonl
data/gpt_new_major_candidate_resolution_batch_output.jsonl
data/new_major_chapter_candidates_resolved.jsonl
data/new_major_chapter_candidates_resolution_manifest.json
data/new_major_chapter_candidates_resolution_parse_failures.jsonl
data/gpt_new_major_candidate_resolution_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCE = DATA / "recovery_new_major_chapter_candidates.jsonl"
ONTOLOGY = DATA / "evidence_unit_ontology.json"
NEW_TAXONOMY = DATA / "new_subunit_cluster_taxonomy_v1.json"

BATCH_INPUT = DATA / "gpt_new_major_candidate_resolution_batch_input.jsonl"
BATCH_OUTPUT = DATA / "gpt_new_major_candidate_resolution_batch_output.jsonl"
BATCH_ERRORS = DATA / "gpt_new_major_candidate_resolution_batch_errors.jsonl"
STATE = DATA / "gpt_new_major_candidate_resolution_state.json"

OUT_RESOLVED = DATA / "new_major_chapter_candidates_resolved.jsonl"
OUT_MANIFEST = DATA / "new_major_chapter_candidates_resolution_manifest.json"
OUT_FAILURES = DATA / "new_major_chapter_candidates_resolution_parse_failures.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL = {"completed", "failed", "expired", "cancelled"}

# Manual structural review: none of these papers justifies a new major chapter.
# Allowed chapters are intentionally conservative and paper-specific.
ALLOWED_CHAPTERS = {
    "25805376": ["2"],                  # pancreatic cysts
    "26808546": ["5"],                  # predictive biomarker / gemcitabine-erlotinib
    "30323161": ["2"],                  # cystic-neoplasm follow-up
    "31267936": ["2"],                  # pancreatic cyst diagnosis/treatment/follow-up
    "31542591": ["5"],                  # BRCA / platinum sensitivity
    "32299515": ["4.1", "4.2", "4.3"],  # implementation / best-practice care
    "33778702": ["1", "2", "5"],        # hereditary genetics + imaging
    "35067789": ["4.1", "4.3"],         # access/disparities
    "35124465": ["4.3", "6"],            # exocrine insufficiency / supportive care
}

SYSTEM_PROMPT = """You are resolving nine residual ontology edge cases in a pancreatic-cancer living-guideline evidence pipeline.

A manual structural review has already concluded that NONE of these records justifies a new major guideline chapter. You are therefore NOT allowed to create a new major chapter.

Your task is only to map the publication to the most appropriate evidence unit(s) among the explicitly allowed existing major chapters and allowed units supplied in the request.

Rules:
1. Do not create a new major chapter.
2. Use only the allowed unit IDs in the request.
3. Assign one PRIMARY unit.
4. Add SECONDARY units only if the paper substantively addresses a separate guideline-level question.
5. Keep the number of assignments minimal.
6. Existing units and proposed NEW_SUBUNIT units are both valid destinations.
7. Do not perform final evidence eligibility appraisal here. Study design, surrogate endpoints, or clinical translatability will be assessed later.
8. Do not infer facts absent from title/abstract.
9. Return only schema-valid JSON.
"""


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_unit_catalog() -> dict[str, list[dict[str, str]]]:
    old = load_json(ONTOLOGY)
    new = load_json(NEW_TAXONOMY)

    catalog: dict[str, list[dict[str, str]]] = {}

    for cid, chapter in old["chapters"].items():
        catalog[cid] = []
        for u in chapter["evidence_units"]:
            catalog[cid].append({
                "unit_key": clean(u["id"]),
                "unit_type": "EXISTING_UNIT",
                "title": clean(u["name"]),
                "definition": clean(u["definition"]),
                "boundary": clean(u["boundary"]),
            })

    for chapter in new["chapters"]:
        cid = clean(chapter["chapter_id"])
        for c in chapter["proposed_clusters"]:
            if clean(c["disposition"]) != "NEW_SUBUNIT":
                continue
            catalog[cid].append({
                "unit_key": clean(c["temporary_cluster_id"]),
                "unit_type": "NEW_SUBUNIT",
                "title": clean(c["title"]),
                "definition": clean(c["definition"]),
                "boundary": clean(c["boundary"]),
            })

    return catalog


def response_schema(allowed_unit_keys: list[str]) -> dict[str, Any]:
    return {
        "name": "resolve_new_major_chapter_candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "primary_unit_key": {
                    "type": "string",
                    "enum": allowed_unit_keys,
                },
                "secondary_unit_keys": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_unit_keys},
                    "minItems": 0,
                    "maxItems": 4,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "primary_unit_key",
                "secondary_unit_keys",
                "confidence",
                "rationale",
            ],
            "additionalProperties": False,
        },
    }


def prepare(model: str, effort: str) -> None:
    rows = load_jsonl(SOURCE)
    catalog = build_unit_catalog()

    source_by_pmid = {clean(r.get("pmid")): r for r in rows}
    expected = set(ALLOWED_CHAPTERS)

    if set(source_by_pmid) != expected:
        raise RuntimeError(
            "The 9-candidate source set does not match the manually reviewed PMID set.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found:    {sorted(source_by_pmid)}"
        )

    requests_out = []

    for pmid in sorted(expected):
        rec = source_by_pmid[pmid]
        allowed_chapters = ALLOWED_CHAPTERS[pmid]

        allowed_units = []
        for cid in allowed_chapters:
            allowed_units.extend(catalog[cid])

        allowed_keys = [u["unit_key"] for u in allowed_units]
        if len(allowed_keys) != len(set(allowed_keys)):
            raise RuntimeError(
                f"Duplicate unit key across allowed chapters for PMID {pmid}. "
                "Unit keys must be globally unique for this request."
            )

        units_text = []
        for u in allowed_units:
            units_text += [
                f"{u['unit_key']} — {u['title']}",
                f"Type: {u['unit_type']}",
                f"Definition: {u['definition']}",
                f"Boundary: {u['boundary']}",
                "",
            ]

        user = "\n".join([
            f"PMID: {pmid}",
            f"Allowed major chapters: {', '.join(allowed_chapters)}",
            f"Title: {clean(rec.get('title'))}",
            f"Abstract: {clean(rec.get('abstract')) or '[no abstract available]'}",
            f"Publication types: {clean(rec.get('publication_types')) or '[not available]'}",
            f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}",
            "",
            "ALLOWED EVIDENCE UNITS:",
            *units_text,
            "Map this publication to the minimal appropriate set of units.",
        ])

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema(allowed_keys),
            },
            "max_completion_tokens": 3500,
            "reasoning_effort": effort,
        }

        requests_out.append({
            "custom_id": f"major-resolve-{pmid}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    write_jsonl(BATCH_INPUT, requests_out)

    print("\nSTEP 2D2 prepared.")
    print(f"  source major-chapter candidates: {len(rows)}")
    print("  accepted new major chapters:     0")
    print(f"  Batch requests:                  {len(requests_out)}")
    print(f"  unique custom IDs:               {len({x['custom_id'] for x in requests_out})}/{len(requests_out)}")
    print(f"  model:                           {model}")
    print(f"  reasoning effort:                {effort}")
    print(f"  input:                           {BATCH_INPUT}")


class Client:
    def __init__(self, key: str, retry_wait: int):
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.s = requests.Session()
        self.headers = {"Authorization": f"Bearer {key}"}
        self.retry_wait = retry_wait

    def request(self, method: str, path: str, *, timeout: int = 600, **kwargs):
        url = path if path.startswith("http") else OPENAI_BASE_URL + path
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
                print(f"WARN {type(e).__name__}: {e}; retry in {self.retry_wait}s")
                time.sleep(self.retry_wait)
                continue

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                print(f"WARN HTTP {r.status_code}; retry in {self.retry_wait}s")
                time.sleep(self.retry_wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:5000]}")
            return r


def load_state() -> dict[str, Any]:
    if not STATE.exists():
        return {}
    return load_json(STATE)


def save_state(state: dict[str, Any]) -> None:
    STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def submit(client: Client, model: str, effort: str) -> dict[str, Any]:
    state = load_state()
    if state.get("batch_id"):
        print(f"Existing Batch: {state['batch_id']}")
        print("Resuming; no duplicate Batch submitted.")
        return state

    with BATCH_INPUT.open("rb") as f:
        upload = client.request(
            "POST",
            "/files",
            files={"file": (BATCH_INPUT.name, f, "application/jsonl")},
            data={"purpose": "batch"},
        ).json()

    payload = {
        "input_file_id": upload["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "project": "ESMO_PDAC_2015_to_2023_PoC",
            "task": "resolve_9_new_major_chapter_candidates",
            "accepted_new_major_chapters": "0",
        },
    }

    batch = client.request(
        "POST",
        "/batches",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    ).json()

    state = {
        "input_file_id": upload["id"],
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "model": model,
        "reasoning_effort": effort,
    }
    save_state(state)

    print(f"Batch id: {batch['id']}")
    print(f"Status:   {batch.get('status')}")
    return state


def download(client: Client, file_id: str, destination: Path) -> None:
    destination.write_bytes(
        client.request(
            "GET",
            f"/files/{file_id}/content",
            timeout=900,
        ).content
    )


def watch(client: Client, poll_seconds: int) -> dict[str, Any]:
    state = load_state()
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No batch_id in state.")

    while True:
        batch = client.request("GET", f"/batches/{batch_id}").json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}

        print(
            f"status={status}; total={counts.get('total')}; "
            f"completed={counts.get('completed')}; failed={counts.get('failed')}"
        )

        state.update({
            "status": status,
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "request_counts": counts,
        })
        save_state(state)

        if status in TERMINAL:
            if batch.get("output_file_id"):
                download(client, batch["output_file_id"], BATCH_OUTPUT)
            if batch.get("error_file_id"):
                download(client, batch["error_file_id"], BATCH_ERRORS)
            return batch

        time.sleep(poll_seconds)


def parse_custom_id(value: str) -> str:
    m = re.fullmatch(r"major-resolve-(\d+)", value)
    if not m:
        raise ValueError(f"Invalid custom_id: {value}")
    return m.group(1)


def merge(model: str) -> None:
    source_rows = load_jsonl(SOURCE)
    source = {clean(r.get("pmid")): r for r in source_rows}
    catalog = build_unit_catalog()

    unit_meta = {}
    for cid, units in catalog.items():
        for u in units:
            unit_meta[u["unit_key"]] = {
                **u,
                "chapter_id": cid,
            }

    results = {}
    failures = []

    with BATCH_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            try:
                pmid = parse_custom_id(custom_id)
                response = obj.get("response")
                if (
                    obj.get("error")
                    or not response
                    or response.get("status_code") != 200
                ):
                    raise ValueError(
                        obj.get("error")
                        or f"HTTP {response.get('status_code') if response else 'missing'}"
                    )

                content = response["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                primary = clean(parsed["primary_unit_key"])
                secondary = []
                for x in parsed.get("secondary_unit_keys") or []:
                    x = clean(x)
                    if x and x != primary and x not in secondary:
                        secondary.append(x)

                allowed_chapters = set(ALLOWED_CHAPTERS[pmid])
                all_keys = [primary] + secondary

                for unit_key in all_keys:
                    if unit_key not in unit_meta:
                        raise ValueError(f"Unknown unit key {unit_key}")
                    if unit_meta[unit_key]["chapter_id"] not in allowed_chapters:
                        raise ValueError(
                            f"Unit {unit_key} lies outside allowed chapters "
                            f"{sorted(allowed_chapters)}"
                        )

                results[pmid] = {
                    "pmid": pmid,
                    "original_status": "NEW_MAJOR_CHAPTER_CANDIDATE",
                    "major_chapter_resolution": "NO_NEW_MAJOR_CHAPTER",
                    "primary_unit_key": primary,
                    "primary_unit_type": unit_meta[primary]["unit_type"],
                    "primary_chapter_id": unit_meta[primary]["chapter_id"],
                    "primary_unit_title": unit_meta[primary]["title"],
                    "secondary_assignments": [
                        {
                            "unit_key": k,
                            "unit_type": unit_meta[k]["unit_type"],
                            "chapter_id": unit_meta[k]["chapter_id"],
                            "unit_title": unit_meta[k]["title"],
                        }
                        for k in secondary
                    ],
                    "confidence": clean(parsed["confidence"]),
                    "rationale": clean(parsed["rationale"]),
                    "model": model,
                    "title": clean(source[pmid].get("title")),
                    "abstract": clean(source[pmid].get("abstract")),
                    "publication_types": clean(source[pmid].get("publication_types")),
                    "evidence_labels": clean(source[pmid].get("evidence_labels")),
                }

            except Exception as e:
                failures.append({
                    "line": line_no,
                    "custom_id": custom_id,
                    "error": f"{type(e).__name__}: {e}",
                })

    missing = sorted(set(source) - set(results))
    write_jsonl(OUT_RESOLVED, [results[p] for p in sorted(results)])
    write_jsonl(OUT_FAILURES, failures)

    manifest = {
        "source_candidates": len(source),
        "accepted_new_major_chapters": 0,
        "resolved_candidates": len(results),
        "missing_candidates": missing,
        "parse_failures": len(failures),
        "status": (
            "COMPLETE"
            if len(results) == len(source) and not failures
            else "INCOMPLETE"
        ),
        "principle": (
            "All 9 residual NEW_MAJOR_CHAPTER_CANDIDATE records were structurally "
            "reviewed as not requiring a new major chapter. This mapping resolves "
            "them into existing major chapters and existing/new evidence units. "
            "Final clinical evidence eligibility is assessed later."
        ),
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSTEP 2D2 merge completed.")
    print(f"  source candidates:            {len(source)}")
    print("  accepted new major chapters:  0")
    print(f"  resolved:                     {len(results)}")
    print(f"  missing:                      {len(missing)}")
    print(f"  parse failures:               {len(failures)}")
    print()

    for pmid in sorted(results):
        r = results[pmid]
        print(
            f"  PMID {pmid} -> chapter {r['primary_chapter_id']} | "
            f"{r['primary_unit_type']} | {r['primary_unit_title']}"
        )
        for s in r["secondary_assignments"]:
            print(
                f"      secondary -> chapter {s['chapter_id']} | "
                f"{s['unit_type']} | {s['unit_title']}"
            )

    print()
    print(f"  resolved: {OUT_RESOLVED}")
    print(f"  manifest: {OUT_MANIFEST}")
    print(f"  failures: {OUT_FAILURES}")

    if missing or failures:
        print("\nWARNING: repair unresolved records before freezing ontology v2.")


def synchronous_test(client: Client) -> None:
    with BATCH_INPUT.open("r", encoding="utf-8") as f:
        req = json.loads(f.readline())
    print(f"Testing: {req['custom_id']}")
    r = client.request(
        "POST",
        "/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(req["body"]),
    )
    print(f"HTTP {r.status_code}")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:10000])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["prepare", "test", "submit", "watch", "merge", "all"],
        default="prepare",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    if args.reset_state and STATE.exists():
        STATE.unlink()
        print(f"Deleted state: {STATE}")

    if args.mode in {"prepare", "all"}:
        prepare(args.model, args.reasoning_effort)
        if args.mode == "prepare":
            return 0

    key = os.environ.get("OPENAI_API_KEY", "").strip()

    if args.mode in {"test", "submit", "watch", "all"} and not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = Client(key, args.retry_wait) if key else None

    if args.mode == "test":
        synchronous_test(client)
        return 0

    if args.mode in {"submit", "all"}:
        submit(client, args.model, args.reasoning_effort)
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
