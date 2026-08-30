#!/usr/bin/env python3
"""
STEP 2B — Build a broad, medically coherent NEW-SUBUNIT taxonomy.

Purpose
-------
Cluster the 1,191 consolidated NEW_SUBUNIT_CANDIDATE records WITHIN each
existing major ESMO-PDAC chapter into a SMALLER NUMBER OF GENUINELY DISTINCT
GUIDELINE-LEVEL TOPICS.

CRITICAL DESIGN PRINCIPLE
-------------------------
This script must NOT create tiny subunits merely to reduce papers per unit.
A valid evidence unit may contain 100, 300, 500, or more papers if they address
the same guideline-level clinical/scientific question.

Medical topic boundaries and technical LLM chunk sizes are separate concepts:
- Evidence unit = one coherent medical guideline question/topic.
- Technical synthesis chunks = later implementation detail if one evidence unit
  contains too many abstracts for a single synthesis request.

This step proposes the taxonomy ONLY. It does NOT yet assign every paper to the
new clusters. That assignment will be STEP 2C.

Inputs
------
data/new_subunit_candidates_combined.jsonl
data/evidence_unit_ontology.json

Outputs
-------
data/gpt_new_subunit_taxonomy_batch_input.jsonl
data/gpt_new_subunit_taxonomy_batch_output.jsonl
data/new_subunit_cluster_taxonomy_v1.json
data/new_subunit_cluster_taxonomy_v1_manifest.json
data/new_subunit_cluster_taxonomy_parse_failures.jsonl
data/gpt_new_subunit_taxonomy_state.json

Default model:
    gpt-5.6-sol
Default reasoning:
    high
"""

from __future__ import annotations

import argparse
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
ONTOLOGY = DATA / "evidence_unit_ontology.json"

BATCH_INPUT = DATA / "gpt_new_subunit_taxonomy_batch_input.jsonl"
BATCH_OUTPUT = DATA / "gpt_new_subunit_taxonomy_batch_output.jsonl"
BATCH_ERRORS = DATA / "gpt_new_subunit_taxonomy_batch_errors.jsonl"
STATE = DATA / "gpt_new_subunit_taxonomy_state.json"

OUT_TAXONOMY = DATA / "new_subunit_cluster_taxonomy_v1.json"
OUT_MANIFEST = DATA / "new_subunit_cluster_taxonomy_v1_manifest.json"
OUT_FAILURES = DATA / "new_subunit_cluster_taxonomy_parse_failures.jsonl"
LOG_PATH = LOGS / "new_subunit_taxonomy_log.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

SYSTEM_PROMPT = """You are designing the topic ontology for a medical clinical-practice-guideline evidence update.

CONTEXT
The source is the 2015 ESMO pancreatic-cancer guideline. A large body of 2015–August 2023 evidence has already been mapped to major chapters and to an initial set of 115 evidence units. The records in this task were specifically flagged because they appear to address NEW guideline-level topics that were not adequately represented by those existing evidence units.

YOUR TASK
Within the ONE fixed major chapter supplied in the request, consolidate the candidate records into a small, medically coherent set of proposed topic clusters.

CRITICAL PRINCIPLES
1. CLUSTER BY MEDICAL QUESTION, NOT BY DESIRED PAPER COUNT.
2. DO NOT create small evidence units merely to make later LLM processing easier.
3. A single evidence unit may legitimately contain HUNDREDS of publications if they address the same guideline-level question.
4. Technical prompt/batch chunking will be handled later and MUST NOT influence medical topic boundaries here.
5. Aggressively merge synonymous, near-synonymous, overlapping, or differently worded candidate topics when they represent the same clinical/scientific question.
6. Keep topics separate only when they would reasonably require a distinct guideline discussion, evidence synthesis, or recommendation/statement.
7. Prefer broad but medically meaningful units over narrow paper-specific niches.
8. Compare every proposed cluster against the EXISTING evidence units supplied for the chapter.
9. If a candidate theme is actually already covered by an existing evidence unit, propose MERGE_INTO_EXISTING_UNIT rather than inventing a new unit.
10. If a candidate theme is clearly not appropriate for the chapter/guideline scope, propose OUT_OF_SCOPE_THEME.
11. NEW_SUBUNIT should be used only for a coherent topic that is substantively distinct from the existing ontology.
12. Do NOT assess evidence quality, treatment efficacy, recommendation strength, or whether the guideline should ultimately change.
13. Do NOT create a new major chapter in this task; the nine new-major-chapter candidates are handled separately.
14. Candidate examples are for ontology design only. Do not attempt final paper-to-cluster assignment in this step.
15. The future evidence-integration step will separately determine whether a new topic has sufficient evidence for guideline text or a recommendation.

OUTPUT GOAL
Produce a compact proposed chapter taxonomy that can subsequently be used to map all candidate papers in STEP 2C.

A good result may contain only a few new subunits even when hundreds of candidate papers are supplied.
Return only the requested structured JSON.
"""


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def log_event(event: str, **payload: Any) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    obj = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chapter_existing_units_text(ontology: dict[str, Any], cid: str) -> str:
    chapter = ontology["chapters"][cid]
    lines = [
        f"FIXED MAJOR CHAPTER: {cid} — {chapter['title']}",
        "",
        "EXISTING EVIDENCE UNITS — DO NOT DUPLICATE THESE:",
    ]
    for unit in chapter["evidence_units"]:
        lines += [
            "",
            f"{unit['id']} — {unit['name']}",
            f"Definition: {unit['definition']}",
            f"Boundary: {unit['boundary']}",
        ]
    return "\n".join(lines)


def compact_candidate_text(rec: dict[str, Any]) -> str:
    proposals = rec.get("candidate_proposals") or []
    proposal_lines = []
    for i, p in enumerate(proposals, 1):
        proposal_lines += [
            f"  Proposal {i}: {clean(p.get('candidate_title'))}",
            f"  Description {i}: {clean(p.get('candidate_description'))}",
        ]
        rationale = clean(p.get("mapping_rationale"))
        if rationale:
            proposal_lines.append(f"  Mapping rationale {i}: {rationale}")

    # Candidate proposals were themselves generated from full title/abstract in
    # an earlier high-reasoning mapping step. Include a short abstract excerpt
    # only as additional disambiguating context, not as the primary clustering key.
    abstract = clean(rec.get("abstract"))
    if len(abstract) > 1200:
        abstract = abstract[:1200].rsplit(" ", 1)[0] + " …"

    return "\n".join([
        f"CANDIDATE KEY: PMID:{clean(rec.get('pmid'))}",
        f"Article title: {clean(rec.get('title'))}",
        f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}",
        *proposal_lines,
        f"Abstract excerpt: {abstract or '[no abstract available]'}",
    ])


def response_schema(existing_unit_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "pdac_new_subunit_taxonomy",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "chapter_id": {"type": "string"},
                "taxonomy_summary": {"type": "string"},
                "proposed_clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "temporary_cluster_id": {"type": "string"},
                            "disposition": {
                                "type": "string",
                                "enum": [
                                    "NEW_SUBUNIT",
                                    "MERGE_INTO_EXISTING_UNIT",
                                    "OUT_OF_SCOPE_THEME",
                                ],
                            },
                            "title": {"type": "string"},
                            "definition": {"type": "string"},
                            "boundary": {"type": "string"},
                            "existing_unit_id": {
                                "type": "string",
                                "enum": [""] + existing_unit_ids,
                            },
                            "candidate_scope_summary": {"type": "string"},
                            "representative_pmids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 0,
                                "maxItems": 8,
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "temporary_cluster_id",
                            "disposition",
                            "title",
                            "definition",
                            "boundary",
                            "existing_unit_id",
                            "candidate_scope_summary",
                            "representative_pmids",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 80,
                },
            },
            "required": [
                "chapter_id",
                "taxonomy_summary",
                "proposed_clusters",
            ],
            "additionalProperties": False,
        },
    }


def build_requests(model: str, effort: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY)
    candidates = load_jsonl(CANDIDATES)

    by_chapter: dict[str, list[dict[str, Any]]] = {
        cid: [] for cid in ontology["chapters"]
    }
    for rec in candidates:
        cid = clean(rec.get("chapter_id"))
        if cid not in by_chapter:
            raise RuntimeError(
                f"Invalid candidate chapter {cid!r} for PMID {rec.get('pmid')}"
            )
        by_chapter[cid].append(rec)

    custom_ids: list[str] = []

    with BATCH_INPUT.open("w", encoding="utf-8") as f:
        for cid, chapter in ontology["chapters"].items():
            records = by_chapter[cid]
            if not records:
                continue

            existing_ids = [
                u["id"] for u in chapter["evidence_units"]
            ]

            candidate_blocks = "\n\n---\n\n".join(
                compact_candidate_text(rec)
                for rec in records
            )

            user_prompt = (
                chapter_existing_units_text(ontology, cid)
                + "\n\n"
                + f"NUMBER OF NEW-SUBUNIT CANDIDATE RECORDS IN THIS CHAPTER: "
                  f"{len(records)}\n\n"
                + "CANDIDATE RECORDS TO CONSOLIDATE:\n\n"
                + candidate_blocks
                + "\n\n"
                + "Design a compact medical topic taxonomy for these candidates. "
                  "Do not optimize cluster count for prompt size. A cluster may "
                  "ultimately contain hundreds of papers."
            )

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema(existing_ids),
                },
                "max_completion_tokens": 12000,
                "reasoning_effort": effort,
            }

            custom_id = f"taxonomy-chapter-{cid.replace('.', '_')}"
            custom_ids.append(custom_id)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    if len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError("Duplicate Batch custom_id detected.")

    size_mb = BATCH_INPUT.stat().st_size / 1024 / 1024
    if size_mb > 190:
        raise RuntimeError(
            f"Batch input is {size_mb:.2f} MB; too close to upload limit."
        )

    chapter_counts = {
        cid: len(rows) for cid, rows in by_chapter.items()
    }

    print("\nSTEP 2B taxonomy batch prepared.")
    print(f"  chapter taxonomy requests: {len(custom_ids)}")
    print(f"  total candidates:          {sum(chapter_counts.values()):,}")
    print(f"  JSONL MB:                  {size_mb:.2f}")
    print(f"  model:                     {model}")
    print(f"  reasoning effort:          {effort}")
    print()
    for cid in ontology["chapters"]:
        print(
            f"  chapter {cid:>3}: "
            f"{chapter_counts[cid]:,} candidate records"
        )
    print(f"\n  batch input: {BATCH_INPUT}")

    summary = {
        "chapter_taxonomy_requests": len(custom_ids),
        "candidate_records": sum(chapter_counts.values()),
        "chapter_candidate_counts": chapter_counts,
        "batch_jsonl_mb": round(size_mb, 2),
        "model": model,
        "reasoning_effort": effort,
    }
    log_event("prepared", **summary)
    return summary


class OpenAIHTTP:
    def __init__(self, api_key: str, retry_wait: int):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty.")
        self.session = requests.Session()
        self.auth = {"Authorization": f"Bearer {api_key}"}
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
            headers = dict(self.auth)
            headers.update(kwargs.pop("headers", {}))
            try:
                response = self.session.request(
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
                    f"WARN: transient OpenAI network/stream error "
                    f"{type(e).__name__}: {e}; retrying in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if response.status_code in {
                408, 409, 429, 500, 502, 503, 504
            }:
                print(
                    f"WARN: transient OpenAI HTTP {response.status_code}; "
                    f"retrying in {self.retry_wait}s"
                )
                time.sleep(self.retry_wait)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {response.status_code}: "
                    f"{response.text[:5000]}"
                )

            return response


def load_state() -> dict[str, Any]:
    if not STATE.exists():
        return {}
    return load_json(STATE)


def save_state(state: dict[str, Any]) -> None:
    STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def submit(
    client: OpenAIHTTP,
    model: str,
    effort: str,
) -> dict[str, Any]:
    state = load_state()
    if state.get("batch_id"):
        print(f"Existing taxonomy batch: {state['batch_id']}")
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
            "task": "design_new_subunit_taxonomy",
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


def download(
    client: OpenAIHTTP,
    file_id: str,
    destination: Path,
) -> None:
    destination.write_bytes(
        client.request(
            "GET",
            f"/files/{file_id}/content",
            timeout=900,
        ).content
    )


def watch(
    client: OpenAIHTTP,
    poll_seconds: int,
) -> dict[str, Any]:
    state = load_state()
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No taxonomy batch_id in state.")

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


def parse_custom_id(value: str) -> str:
    m = re.fullmatch(
        r"taxonomy-chapter-([0-9_]+)", value
    )
    if not m:
        raise ValueError(f"Invalid custom_id: {value}")
    return m.group(1).replace("_", ".")


def normalize_taxonomy(
    parsed: dict[str, Any],
    cid: str,
    ontology: dict[str, Any],
    valid_pmids: set[str],
) -> dict[str, Any]:
    if clean(parsed.get("chapter_id")) != cid:
        raise ValueError(
            f"Model returned chapter_id={parsed.get('chapter_id')} "
            f"for request chapter {cid}"
        )

    existing_ids = {
        u["id"] for u in ontology["chapters"][cid]["evidence_units"]
    }

    clusters = parsed.get("proposed_clusters") or []
    if not clusters:
        raise ValueError("No proposed clusters returned.")

    seen_ids = set()
    normalized = []

    for index, cluster in enumerate(clusters, 1):
        temp_id = clean(cluster.get("temporary_cluster_id"))
        if not temp_id:
            temp_id = f"{cid}-TEMP-{index:02d}"
        if temp_id in seen_ids:
            temp_id = f"{cid}-TEMP-{index:02d}"
        seen_ids.add(temp_id)

        disposition = cluster["disposition"]
        existing_unit_id = clean(
            cluster.get("existing_unit_id")
        )

        normalization = []

        if disposition == "NEW_SUBUNIT":
            if existing_unit_id:
                normalization.append(
                    "cleared_irrelevant_existing_unit_id"
                )
            existing_unit_id = ""

        elif disposition == "MERGE_INTO_EXISTING_UNIT":
            if existing_unit_id not in existing_ids:
                raise ValueError(
                    f"MERGE cluster has invalid existing unit "
                    f"{existing_unit_id}"
                )

        elif disposition == "OUT_OF_SCOPE_THEME":
            if existing_unit_id:
                normalization.append(
                    "cleared_irrelevant_existing_unit_id"
                )
            existing_unit_id = ""

        else:
            raise ValueError(
                f"Unknown disposition: {disposition}"
            )

        representative_pmids = []
        for pmid in cluster.get("representative_pmids") or []:
            pmid = clean(pmid)
            if pmid and pmid in valid_pmids:
                if pmid not in representative_pmids:
                    representative_pmids.append(pmid)

        normalized.append(
            {
                "temporary_cluster_id": temp_id,
                "disposition": disposition,
                "title": clean(cluster.get("title")),
                "definition": clean(cluster.get("definition")),
                "boundary": clean(cluster.get("boundary")),
                "existing_unit_id": existing_unit_id,
                "candidate_scope_summary": clean(
                    cluster.get("candidate_scope_summary")
                ),
                "representative_pmids": representative_pmids,
                "rationale": clean(cluster.get("rationale")),
                "deterministic_normalization":
                    "; ".join(normalization),
            }
        )

    return {
        "chapter_id": cid,
        "chapter_title": ontology["chapters"][cid]["title"],
        "taxonomy_summary": clean(
            parsed.get("taxonomy_summary")
        ),
        "proposed_clusters": normalized,
    }


def merge(model: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY)
    candidates = load_jsonl(CANDIDATES)

    pmids_by_chapter: dict[str, set[str]] = {
        cid: set() for cid in ontology["chapters"]
    }
    candidate_counts = Counter()

    for rec in candidates:
        cid = clean(rec.get("chapter_id"))
        pmid = clean(rec.get("pmid"))
        pmids_by_chapter[cid].add(pmid)
        candidate_counts[cid] += 1

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    with BATCH_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")

            try:
                cid = parse_custom_id(custom_id)
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

                result = normalize_taxonomy(
                    parsed,
                    cid,
                    ontology,
                    pmids_by_chapter[cid],
                )
                results[cid] = result

            except Exception as e:
                failures.append(
                    {
                        "line": line_no,
                        "custom_id": custom_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    ordered_results = [
        results[cid]
        for cid in ontology["chapters"]
        if cid in results
    ]

    output = {
        "ontology_version": "ESMO-PDAC-PoC-new-subunit-clusters-v1",
        "status": (
            "COMPLETE"
            if not failures
            and len(results) == len(
                [cid for cid, n in candidate_counts.items() if n]
            )
            else "INCOMPLETE"
        ),
        "design_principle": (
            "Medical evidence-unit boundaries are independent of paper count. "
            "A unit may contain hundreds of publications. Technical synthesis "
            "chunking will be performed later without changing the ontology."
        ),
        "source_candidate_file": str(CANDIDATES),
        "model": model,
        "chapters": ordered_results,
    }

    OUT_TAXONOMY.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(OUT_FAILURES, failures)

    disposition_counts = Counter()
    total_clusters = 0
    per_chapter_cluster_counts = {}

    for chapter in ordered_results:
        counts = Counter(
            c["disposition"]
            for c in chapter["proposed_clusters"]
        )
        cid = chapter["chapter_id"]
        per_chapter_cluster_counts[cid] = dict(counts)
        disposition_counts.update(counts)
        total_clusters += len(chapter["proposed_clusters"])

    manifest = {
        "candidate_records": len(candidates),
        "chapter_candidate_counts": dict(candidate_counts),
        "chapter_taxonomies_successful": len(results),
        "parse_failures": len(failures),
        "proposed_theme_clusters_total": total_clusters,
        "disposition_counts": dict(disposition_counts),
        "per_chapter_cluster_counts": per_chapter_cluster_counts,
        "important_note": (
            "Cluster counts are ontology-design counts only. "
            "No final paper assignment has yet been performed."
        ),
        "next_step": (
            "STEP 2C: map every candidate PMID+chapter record to exactly one "
            "primary proposed cluster, with optional secondary cluster(s), "
            "then inspect resulting paper counts per medical unit."
        ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSTEP 2B taxonomy merge completed.")
    print(f"  chapter taxonomies:          {len(results)}")
    print(f"  parse failures:              {len(failures)}")
    print(f"  proposed theme clusters:     {total_clusters}")
    print(
        f"  NEW_SUBUNIT:                 "
        f"{disposition_counts['NEW_SUBUNIT']}"
    )
    print(
        f"  MERGE_INTO_EXISTING_UNIT:    "
        f"{disposition_counts['MERGE_INTO_EXISTING_UNIT']}"
    )
    print(
        f"  OUT_OF_SCOPE_THEME:          "
        f"{disposition_counts['OUT_OF_SCOPE_THEME']}"
    )
    print()
    for chapter in ordered_results:
        cid = chapter["chapter_id"]
        print(
            f"  chapter {cid:>3}: "
            f"{candidate_counts[cid]:,} candidate papers -> "
            f"{len(chapter['proposed_clusters'])} proposed themes"
        )
    print()
    print(f"  taxonomy: {OUT_TAXONOMY}")
    print(f"  manifest: {OUT_MANIFEST}")
    print(f"  failures: {OUT_FAILURES}")

    if failures:
        print(
            "\nWARNING: repair taxonomy parse failures before STEP 2C."
        )

    return manifest


def synchronous_test(client: OpenAIHTTP) -> None:
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
        )[:12000]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "STEP 2B: consolidate new-subunit candidates into broad, "
            "medically coherent proposed evidence units."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "prepare",
            "test",
            "submit",
            "watch",
            "merge",
            "all",
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
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
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
        build_requests(
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
            "test",
            "submit",
            "watch",
            "all",
        }
        and not api_key
    ):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this "
            "PowerShell session."
        )

    client = (
        OpenAIHTTP(
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
