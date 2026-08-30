#!/usr/bin/env python3
"""
STEP 2C — Assign every NEW_SUBUNIT candidate paper to the STEP 2B chapter taxonomy.

Purpose
-------
Use the broad medical taxonomy proposed in STEP 2B to map each of the 1,191
candidate PMID+chapter records to:
- one primary proposed theme cluster,
- optional secondary proposed theme cluster(s).

IMPORTANT
---------
- Medical topic boundaries are already fixed by STEP 2B.
- This step does NOT create further clusters.
- It does NOT split units by desired paper count.
- A final unit may contain hundreds of papers.
- Later technical synthesis chunking is independent of ontology structure.

Inputs
------
data/new_subunit_candidates_combined.jsonl
data/new_subunit_cluster_taxonomy_v1.json
data/manual_evidence_exclusions.jsonl  (optional safety exclusion)

Outputs
-------
data/gpt_new_subunit_assignment_batch_input.jsonl
data/gpt_new_subunit_assignment_batch_output.jsonl
data/new_subunit_candidate_assignments.csv
data/new_subunit_candidate_assignments_expanded.jsonl
data/new_subunit_candidate_assignments_new_units.jsonl
data/new_subunit_candidate_assignments_existing_units.jsonl
data/new_subunit_candidate_assignments_out_of_scope.jsonl
data/new_subunit_candidate_assignment_manifest.json
data/new_subunit_candidate_assignment_parse_failures.jsonl
data/gpt_new_subunit_assignment_state.json

Default:
    model = gpt-5.6-sol
    reasoning_effort = high
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"

CANDIDATES = DATA / "new_subunit_candidates_combined.jsonl"
TAXONOMY = DATA / "new_subunit_cluster_taxonomy_v1.json"
MANUAL_EXCLUSIONS = DATA / "manual_evidence_exclusions.jsonl"

BATCH_INPUT = DATA / "gpt_new_subunit_assignment_batch_input.jsonl"
BATCH_OUTPUT = DATA / "gpt_new_subunit_assignment_batch_output.jsonl"
BATCH_ERRORS = DATA / "gpt_new_subunit_assignment_batch_errors.jsonl"
STATE = DATA / "gpt_new_subunit_assignment_state.json"

OUT_CSV = DATA / "new_subunit_candidate_assignments.csv"
OUT_EXPANDED = DATA / "new_subunit_candidate_assignments_expanded.jsonl"
OUT_NEW = DATA / "new_subunit_candidate_assignments_new_units.jsonl"
OUT_EXISTING = DATA / "new_subunit_candidate_assignments_existing_units.jsonl"
OUT_OOS = DATA / "new_subunit_candidate_assignments_out_of_scope.jsonl"
OUT_MANIFEST = DATA / "new_subunit_candidate_assignment_manifest.json"
OUT_FAILURES = DATA / "new_subunit_candidate_assignment_parse_failures.jsonl"
LOG_PATH = LOGS / "new_subunit_candidate_assignment_log.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

SYSTEM_PROMPT = """You are performing precise topic assignment for a medical clinical-practice-guideline evidence-update workflow.

CONTEXT
A prior ontology-design step has already consolidated broad new-topic candidates into a compact set of medically coherent proposed theme clusters within each major ESMO pancreatic-cancer guideline chapter.

YOUR TASK
Assign this publication to the most appropriate proposed theme cluster(s) WITHIN THE FIXED MAJOR CHAPTER.

CRITICAL RULES
1. Do NOT create any new cluster in this step.
2. Do NOT change the major chapter.
3. Assign the SMALLEST set of clusters that captures the substantive scientific content relevant to this chapter.
4. The PRIMARY cluster must reflect the main scientific/clinical question.
5. Add SECONDARY clusters only when substantively studied. Background mentions, incidental covariates, and generic discussion do not count.
6. Do NOT optimize assignment by paper count. A cluster may ultimately contain hundreds of papers.
7. Prefer one broad medically correct cluster over several narrow overlapping clusters.
8. If the primary cluster has disposition OUT_OF_SCOPE_THEME, do not assign secondary clusters.
9. Do NOT assess evidence quality, risk of bias, efficacy, recommendation strength, or whether the guideline should change.
10. Do NOT infer content absent from title/abstract.
11. Guidelines and consensus statements were excluded upstream and must not be reintroduced.
12. Return only schema-valid JSON.
"""


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if optional:
            return []
        raise FileNotFoundError(f"Missing required file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path.name} line {line_no}: {e}"
                ) from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def manual_excluded_pmids() -> set[str]:
    return {
        clean(r.get("pmid"))
        for r in load_jsonl(MANUAL_EXCLUSIONS, optional=True)
        if clean(r.get("pmid"))
    }


def chapter_taxonomy_map(taxonomy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        clean(ch["chapter_id"]): ch["proposed_clusters"]
        for ch in taxonomy["chapters"]
    }


def cluster_maps(taxonomy: dict[str, Any]):
    by_chapter = chapter_taxonomy_map(taxonomy)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for cid, clusters in by_chapter.items():
        for c in clusters:
            tid = clean(c["temporary_cluster_id"])
            key = (cid, tid)
            if key in meta:
                raise RuntimeError(f"Duplicate temporary cluster ID within chapter: {key}")
            meta[key] = c
    return by_chapter, meta


def cluster_prompt(cid: str, clusters: list[dict[str, Any]]) -> str:
    lines = [
        f"FIXED MAJOR CHAPTER: {cid}",
        "",
        "ALLOWED PROPOSED THEME CLUSTERS:",
    ]
    for c in clusters:
        lines += [
            "",
            f"{clean(c['temporary_cluster_id'])} — {clean(c['title'])}",
            f"Disposition: {clean(c['disposition'])}",
            f"Definition: {clean(c['definition'])}",
            f"Boundary: {clean(c['boundary'])}",
        ]
        if clean(c.get("existing_unit_id")):
            lines.append(
                f"Existing evidence unit target: {clean(c['existing_unit_id'])}"
            )
    return "\n".join(lines)


def candidate_text(rec: dict[str, Any]) -> str:
    proposals = rec.get("candidate_proposals") or []
    proposal_lines = []
    for i, p in enumerate(proposals, 1):
        proposal_lines += [
            f"Candidate proposal {i}: {clean(p.get('candidate_title'))}",
            f"Candidate description {i}: {clean(p.get('candidate_description'))}",
        ]
        rationale = clean(p.get("mapping_rationale"))
        if rationale:
            proposal_lines.append(f"Prior mapping rationale {i}: {rationale}")

    return "\n".join([
        f"PMID: {clean(rec.get('pmid'))}",
        f"Title: {clean(rec.get('title')) or '[missing]'}",
        f"Abstract: {clean(rec.get('abstract')) or '[no abstract available]'}",
        f"Publication types: {clean(rec.get('publication_types')) or '[not available]'}",
        f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}",
        f"MeSH terms: {clean(rec.get('mesh_terms')) or '[not available]'}",
        f"Keywords: {clean(rec.get('keywords')) or '[not available]'}",
        *proposal_lines,
    ])


def response_schema(cluster_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "pdac_new_subunit_candidate_assignment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "primary_cluster_id": {
                    "type": "string",
                    "enum": cluster_ids,
                },
                "secondary_cluster_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": cluster_ids},
                    "minItems": 0,
                    "maxItems": 4,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "broad_multidomain_review": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "primary_cluster_id",
                "secondary_cluster_ids",
                "confidence",
                "broad_multidomain_review",
                "rationale",
            ],
            "additionalProperties": False,
        },
    }


def prepare(model: str, effort: str) -> dict[str, Any]:
    taxonomy = load_json(TAXONOMY)
    if taxonomy.get("status") != "COMPLETE":
        raise RuntimeError(
            f"Taxonomy is not COMPLETE: status={taxonomy.get('status')}"
        )

    candidates = load_jsonl(CANDIDATES)
    excluded = manual_excluded_pmids()
    by_chapter, _ = cluster_maps(taxonomy)

    eligible: list[dict[str, Any]] = []
    excluded_count = 0

    for rec in candidates:
        pmid = clean(rec.get("pmid"))
        cid = clean(rec.get("chapter_id"))

        if pmid in excluded:
            excluded_count += 1
            continue
        if cid not in by_chapter:
            raise RuntimeError(
                f"No STEP 2B taxonomy found for chapter {cid}, PMID {pmid}"
            )
        if not by_chapter[cid]:
            raise RuntimeError(f"Chapter {cid} has no proposed clusters.")
        eligible.append(rec)

    custom_ids = []
    chapter_counts = Counter()

    with BATCH_INPUT.open("w", encoding="utf-8") as f:
        for rec in eligible:
            pmid = clean(rec.get("pmid"))
            cid = clean(rec.get("chapter_id"))
            clusters = by_chapter[cid]
            cluster_ids = [clean(c["temporary_cluster_id"]) for c in clusters]

            custom_id = f"assign-{pmid}-{cid.replace('.', '_')}"
            custom_ids.append(custom_id)

            user = (
                cluster_prompt(cid, clusters)
                + "\n\nPUBLICATION TO ASSIGN:\n"
                + candidate_text(rec)
                + "\n\nAssign this publication to the broad medical taxonomy above."
            )

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema(cluster_ids),
                },
                "max_completion_tokens": 2200,
                "reasoning_effort": effort,
            }

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            chapter_counts[cid] += 1

    duplicates = [
        cid for cid, n in Counter(custom_ids).items() if n > 1
    ]
    if duplicates:
        raise RuntimeError(
            f"HARD FAIL: duplicate custom IDs: {duplicates[:10]}"
        )

    size_mb = BATCH_INPUT.stat().st_size / 1024 / 1024
    if len(custom_ids) > 50_000:
        raise RuntimeError("Batch exceeds 50,000 requests.")
    if size_mb > 190:
        raise RuntimeError(
            f"Batch is {size_mb:.2f} MB; split before upload."
        )

    summary = {
        "source_candidates": len(candidates),
        "manual_exclusions_removed": excluded_count,
        "eligible_candidate_assignments": len(eligible),
        "chapter_request_counts": dict(chapter_counts),
        "batch_jsonl_mb": round(size_mb, 2),
        "model": model,
        "reasoning_effort": effort,
    }

    print("\nSTEP 2C candidate assignment prepared.")
    print(f"  source candidates:           {len(candidates):,}")
    print(f"  manual exclusions removed:   {excluded_count:,}")
    print(f"  eligible assignments:        {len(eligible):,}")
    print(f"  unique custom IDs:           {len(set(custom_ids)):,}/{len(custom_ids):,}")
    print(f"  JSONL MB:                    {size_mb:.2f}")
    print(f"  model:                       {model}")
    print(f"  reasoning effort:            {effort}")
    print()
    for cid in by_chapter:
        print(f"  chapter {cid:>3}: {chapter_counts[cid]:,} requests")
    print(f"\n  batch input: {BATCH_INPUT}")
    return summary


class Client:
    def __init__(self, key: str, retry_wait: int):
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {key}"}
        self.retry_wait = retry_wait

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 600,
        **kwargs: Any,
    ) -> requests.Response:
        url = path if path.startswith("http") else OPENAI_BASE_URL + path

        while True:
            headers = dict(self.headers)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.ChunkedEncodingError,
                requests.ContentDecodingError,
            ) as e:
                print(
                    f"WARN {type(e).__name__}: {e}; "
                    f"retry in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                print(
                    f"WARN HTTP {r.status_code}; "
                    f"retry in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {r.status_code}: {r.text[:5000]}"
                )
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
        print(f"Existing assignment Batch: {state['batch_id']}")
        print("Resuming it; no duplicate Batch submitted.")
        return state

    with BATCH_INPUT.open("rb") as f:
        upload = client.request(
            "POST",
            "/files",
            files={
                "file": (
                    BATCH_INPUT.name,
                    f,
                    "application/jsonl",
                )
            },
            data={"purpose": "batch"},
        ).json()

    payload = {
        "input_file_id": upload["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "project": "ESMO_PDAC_2015_to_2023_PoC",
            "task": "assign_new_subunit_candidates_to_taxonomy",
            "model": model,
            "reasoning_effort": effort,
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
        raise RuntimeError("No assignment batch_id in state.")

    while True:
        batch = client.request(
            "GET", f"/batches/{batch_id}"
        ).json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}

        print(
            f"status={status}; total={counts.get('total')}; "
            f"completed={counts.get('completed')}; "
            f"failed={counts.get('failed')}"
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

        if status in TERMINAL_BATCH_STATUSES:
            if batch.get("output_file_id"):
                download(
                    client,
                    batch["output_file_id"],
                    BATCH_OUTPUT,
                )
            if batch.get("error_file_id"):
                download(
                    client,
                    batch["error_file_id"],
                    BATCH_ERRORS,
                )
            return batch

        time.sleep(poll_seconds)


def parse_custom_id(value: str) -> tuple[str, str]:
    m = re.fullmatch(r"assign-(\d+)-([0-9_]+)", value)
    if not m:
        raise ValueError(f"Invalid custom_id: {value}")
    return m.group(1), m.group(2).replace("_", ".")


def normalize_mapping(
    parsed: dict[str, Any],
    cid: str,
    cluster_ids: list[str],
    cluster_meta: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    primary = clean(parsed["primary_cluster_id"])
    secondary = list(parsed.get("secondary_cluster_ids") or [])
    confidence = parsed["confidence"]
    broad = bool(parsed["broad_multidomain_review"])
    rationale = clean(parsed["rationale"])

    allowed = set(cluster_ids)
    if primary not in allowed:
        raise ValueError(f"Invalid primary cluster: {primary}")
    if any(x not in allowed for x in secondary):
        raise ValueError("Invalid secondary cluster.")

    # Deduplicate secondary and remove the primary if repeated.
    secondary_set = set(secondary)
    secondary_set.discard(primary)
    secondary = [x for x in cluster_ids if x in secondary_set]

    primary_meta = cluster_meta[(cid, primary)]

    normalization = []
    if primary_meta["disposition"] == "OUT_OF_SCOPE_THEME" and secondary:
        secondary = []
        normalization.append(
            "cleared_secondary_clusters_because_primary_is_out_of_scope"
        )

    return {
        "primary_cluster_id": primary,
        "secondary_cluster_ids": secondary,
        "confidence": confidence,
        "broad_multidomain_review": broad,
        "rationale": rationale,
        "deterministic_normalization": "; ".join(normalization),
    }


def parse_output():
    taxonomy = load_json(TAXONOMY)
    candidates = load_jsonl(CANDIDATES)
    excluded = manual_excluded_pmids()

    by_chapter, cluster_meta = cluster_maps(taxonomy)
    source = {
        (clean(r.get("pmid")), clean(r.get("chapter_id"))): r
        for r in candidates
        if clean(r.get("pmid")) not in excluded
    }

    cluster_ids_by_chapter = {
        cid: [clean(c["temporary_cluster_id"]) for c in clusters]
        for cid, clusters in by_chapter.items()
    }

    results: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    with BATCH_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            try:
                pmid, cid = parse_custom_id(custom_id)
                response = obj.get("response")

                if (
                    obj.get("error")
                    or not response
                    or response.get("status_code") != 200
                ):
                    raise ValueError(
                        obj.get("error")
                        or f"HTTP "
                           f"{response.get('status_code') if response else 'missing'}"
                    )

                content = (
                    response["body"]["choices"][0]["message"]["content"]
                )
                parsed = json.loads(content)

                result = normalize_mapping(
                    parsed,
                    cid,
                    cluster_ids_by_chapter[cid],
                    cluster_meta,
                )

                key = (pmid, cid)
                if key not in source:
                    raise ValueError(
                        f"No source candidate for {key}"
                    )
                if key in results:
                    raise ValueError(
                        f"Duplicate result for {key}"
                    )

                results[key] = result

            except Exception as e:
                failures.append(
                    {
                        "line": line_no,
                        "custom_id": custom_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    return source, results, failures, cluster_meta


def merge(model: str) -> dict[str, Any]:
    source, results, failures, cluster_meta = parse_output()
    state = load_state()

    flat_rows: list[dict[str, str]] = []
    expanded: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    existing_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []

    cluster_counts = Counter()
    disposition_assignment_counts = Counter()
    primary_disposition_counts = Counter()
    confidence_counts = Counter()
    missing = []

    all_fields = set()

    for key, rec in source.items():
        pmid, cid = key
        result = results.get(key)

        if not result:
            missing.append(
                {"pmid": pmid, "chapter_id": cid}
            )
            continue

        primary = result["primary_cluster_id"]
        secondary = result["secondary_cluster_ids"]
        all_clusters = [primary] + secondary

        primary_meta = cluster_meta[(cid, primary)]
        primary_disposition = primary_meta["disposition"]

        confidence_counts[result["confidence"]] += 1
        primary_disposition_counts[primary_disposition] += 1

        enriched = dict(rec)
        enriched.update(
            {
                "primary_cluster_id": primary,
                "primary_cluster_title": clean(
                    primary_meta["title"]
                ),
                "primary_cluster_disposition":
                    primary_disposition,
                "primary_existing_unit_id": clean(
                    primary_meta.get("existing_unit_id")
                ),
                "secondary_cluster_ids": secondary,
                "all_cluster_ids": all_clusters,
                "assignment_confidence":
                    result["confidence"],
                "assignment_rationale":
                    result["rationale"],
                "broad_multidomain_review":
                    result["broad_multidomain_review"],
                "deterministic_normalization":
                    result["deterministic_normalization"],
                "assignment_model": model,
                "assignment_batch_id":
                    clean(state.get("batch_id")),
            }
        )

        for cluster_id in all_clusters:
            meta = cluster_meta[(cid, cluster_id)]
            cluster_counts[(cid, cluster_id)] += 1
            disposition_assignment_counts[
                meta["disposition"]
            ] += 1

            expanded_rec = dict(enriched)
            expanded_rec.update(
                {
                    "assigned_cluster_id": cluster_id,
                    "assigned_cluster_title":
                        clean(meta["title"]),
                    "assigned_cluster_disposition":
                        meta["disposition"],
                    "assigned_existing_unit_id":
                        clean(meta.get("existing_unit_id")),
                    "assignment_role":
                        "primary"
                        if cluster_id == primary
                        else "secondary",
                }
            )
            expanded.append(expanded_rec)

            if meta["disposition"] == "NEW_SUBUNIT":
                new_rows.append(expanded_rec)
            elif meta["disposition"] == "MERGE_INTO_EXISTING_UNIT":
                existing_rows.append(expanded_rec)
            elif meta["disposition"] == "OUT_OF_SCOPE_THEME":
                oos_rows.append(expanded_rec)

        flat = {}
        for k, v in enriched.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = clean(v)
        flat_rows.append(flat)
        all_fields.update(flat)

    write_jsonl(OUT_EXPANDED, expanded)
    write_jsonl(OUT_NEW, new_rows)
    write_jsonl(OUT_EXISTING, existing_rows)
    write_jsonl(OUT_OOS, oos_rows)
    write_jsonl(OUT_FAILURES, failures)

    preferred = [
        "pmid", "title", "abstract", "chapter_id", "chapter_title",
        "publication_types", "evidence_labels",
        "primary_cluster_id", "primary_cluster_title",
        "primary_cluster_disposition", "primary_existing_unit_id",
        "secondary_cluster_ids", "all_cluster_ids",
        "assignment_confidence", "assignment_rationale",
        "broad_multidomain_review",
        "candidate_proposals",
        "assignment_model", "assignment_batch_id",
        "deterministic_normalization",
    ]
    fieldnames = [x for x in preferred if x in all_fields]
    fieldnames += sorted(
        all_fields - set(fieldnames),
        key=str.casefold,
    )

    with OUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    # Produce readable cluster count table in manifest.
    taxonomy = load_json(TAXONOMY)
    _, cluster_meta2 = cluster_maps(taxonomy)

    cluster_count_manifest = {}
    for (cid, cluster_id), count in sorted(
        cluster_counts.items()
    ):
        meta = cluster_meta2[(cid, cluster_id)]
        cluster_count_manifest[f"{cid}::{cluster_id}"] = {
            "chapter_id": cid,
            "cluster_id": cluster_id,
            "title": clean(meta["title"]),
            "disposition": meta["disposition"],
            "existing_unit_id": clean(
                meta.get("existing_unit_id")
            ),
            "paper_assignments": count,
        }

    manifest = {
        "input_candidates": len(source),
        "successful_assignments": len(results),
        "missing_assignments": len(missing),
        "parse_failures": len(failures),
        "primary_disposition_counts":
            dict(primary_disposition_counts),
        "all_cluster_assignment_disposition_counts":
            dict(disposition_assignment_counts),
        "confidence_counts": dict(confidence_counts),
        "expanded_paper_cluster_assignments":
            len(expanded),
        "new_subunit_cluster_assignments":
            len(new_rows),
        "merge_into_existing_unit_assignments":
            len(existing_rows),
        "out_of_scope_theme_assignments":
            len(oos_rows),
        "cluster_counts": cluster_count_manifest,
        "missing_records": missing,
        "design_note": (
            "Paper count does not define unit boundaries. "
            "High-count units remain medically intact and will "
            "be technically chunked only during evidence synthesis."
        ),
        "next_step": (
            "Review resulting cluster paper counts; promote accepted "
            "NEW_SUBUNIT clusters into ontology v2 and merge "
            "MERGE_INTO_EXISTING_UNIT assignments into their target "
            "existing evidence units."
        ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSTEP 2C assignment merge completed.")
    print(f"  input candidates:              {len(source):,}")
    print(f"  successful assignments:        {len(results):,}")
    print(f"  missing assignments:           {len(missing):,}")
    print(f"  parse failures:                {len(failures):,}")
    print(f"  expanded cluster assignments:  {len(expanded):,}")
    print()
    print("Primary disposition counts:")
    for k, v in primary_disposition_counts.items():
        print(f"  {k}: {v:,}")
    print()
    print("Assignment confidence:")
    for k, v in confidence_counts.items():
        print(f"  {k}: {v:,}")
    print()
    print("Largest 20 proposed cluster assignments:")
    largest = sorted(
        cluster_count_manifest.values(),
        key=lambda x: x["paper_assignments"],
        reverse=True,
    )[:20]
    for item in largest:
        print(
            f"  chapter {item['chapter_id']:>3} | "
            f"{item['paper_assignments']:>4} | "
            f"{item['disposition']:<25} | "
            f"{item['title']}"
        )
    print()
    print(f"  CSV:       {OUT_CSV}")
    print(f"  expanded:  {OUT_EXPANDED}")
    print(f"  new units: {OUT_NEW}")
    print(f"  existing:  {OUT_EXISTING}")
    print(f"  OOS:       {OUT_OOS}")
    print(f"  manifest:  {OUT_MANIFEST}")
    print(f"  failures:  {OUT_FAILURES}")

    if missing or failures:
        print(
            "\nWARNING: repair missing/failed assignments "
            "before ontology v2 is finalized."
        )

    return manifest


def synchronous_test(client: Client) -> None:
    if not BATCH_INPUT.exists():
        raise RuntimeError(
            "Batch input missing. Run --mode prepare first."
        )

    with BATCH_INPUT.open("r", encoding="utf-8") as f:
        request_obj = json.loads(f.readline())

    print(f"Testing: {request_obj['custom_id']}")
    response = client.request(
        "POST",
        "/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_obj["body"]),
    )
    print(f"HTTP {response.status_code}")
    print(
        json.dumps(
            response.json(),
            ensure_ascii=False,
            indent=2,
        )[:9000]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "STEP 2C: assign every new-subunit candidate "
            "paper to the STEP 2B broad medical taxonomy."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "prepare", "test", "submit",
            "watch", "merge", "all"
        ],
        default="prepare",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OPENAI_MODEL", "gpt-5.6-sol"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=[
            "none", "low", "medium",
            "high", "xhigh", "max"
        ],
        default="high",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
    )
    args = parser.parse_args()

    if args.reset_state and STATE.exists():
        STATE.unlink()
        print(f"Deleted state: {STATE}")

    if args.mode in {"prepare", "all"}:
        prepare(
            args.model,
            args.reasoning_effort,
        )
        if args.mode == "prepare":
            return 0

    api_key = os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip()

    if (
        args.mode in {
            "test", "submit", "watch", "all"
        }
        and not api_key
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this "
            "PowerShell session."
        )

    client = (
        Client(
            api_key,
            args.retry_wait,
        )
        if api_key
        else None
    )

    if args.mode == "test":
        synchronous_test(client)
        return 0

    if args.mode in {"submit", "all"}:
        submit(
            client,
            args.model,
            args.reasoning_effort,
        )
        if args.mode == "submit":
            return 0

    if args.mode in {"watch", "all"}:
        batch = watch(
            client,
            args.poll_seconds,
        )
        print(
            f"Batch terminal status: "
            f"{batch.get('status')}"
        )
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
