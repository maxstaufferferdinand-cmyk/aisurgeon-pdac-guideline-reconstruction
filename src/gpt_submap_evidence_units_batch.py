#!/usr/bin/env python3
"""
GPT topic-oriented evidence-unit submapping for the ESMO PDAC 2015 -> Aug 2023 PoC.

INPUTS
------
data/mapped_evidence_by_chapter.jsonl
data/evidence_unit_ontology.json

TASK
----
Each existing paper-chapter assignment is mapped to one or more precise
evidence units WITHIN THAT ALREADY-ASSIGNED CHAPTER.

This is ONLY semantic submapping. It does NOT:
- assess evidence quality,
- judge efficacy,
- update recommendations,
- rewrite guideline text.

OUTPUTS
-------
data/gpt_evidence_unit_submapping_batch_input.jsonl
data/pubmed_evidence_submapped.csv
data/mapped_evidence_by_evidence_unit.jsonl
data/novel_topic_records.jsonl
data/questionable_chapter_assignments.jsonl
data/gpt_evidence_unit_submapping_manifest.json
data/gpt_evidence_unit_submapping_parse_failures.jsonl
data/gpt_evidence_unit_submapping_state.json

Default model:
    gpt-5.6-sol
Default reasoning:
    medium

For maximum mapping precision, use:
    --model gpt-5.6-sol --reasoning-effort high

The later evidence synthesis / guideline-writing stage should remain a separate
task and can use higher reasoning independently.
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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

INPUT_EVIDENCE = DATA_DIR / "mapped_evidence_by_chapter.jsonl"
ONTOLOGY_PATH = DATA_DIR / "evidence_unit_ontology.json"

BATCH_INPUT = DATA_DIR / "gpt_evidence_unit_submapping_batch_input.jsonl"
BATCH_OUTPUT = DATA_DIR / "gpt_evidence_unit_submapping_batch_output.jsonl"
BATCH_ERRORS = DATA_DIR / "gpt_evidence_unit_submapping_batch_errors.jsonl"
STATE_PATH = DATA_DIR / "gpt_evidence_unit_submapping_state.json"

SUBMAPPED_CSV = DATA_DIR / "pubmed_evidence_submapped.csv"
EXPANDED_JSONL = DATA_DIR / "mapped_evidence_by_evidence_unit.jsonl"
NOVEL_JSONL = DATA_DIR / "novel_topic_records.jsonl"
QUESTIONABLE_JSONL = DATA_DIR / "questionable_chapter_assignments.jsonl"
PARSE_FAILURES = DATA_DIR / "gpt_evidence_unit_submapping_parse_failures.jsonl"
MANIFEST_PATH = DATA_DIR / "gpt_evidence_unit_submapping_manifest.json"
LOG_PATH = LOG_DIR / "gpt_evidence_unit_submapping_log.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

SYSTEM_PROMPT = """You are performing a precise topic-oriented submapping task for a medical guideline evidence-update workflow.

CONTEXT
The source guideline is the 2015 ESMO pancreatic cancer guideline. Each publication has already been assigned to ONE guideline chapter in a previous independent mapping step. Your current task is ONLY to map that publication to the most appropriate predefined EVIDENCE UNIT(S) WITHIN THAT ASSIGNED CHAPTER.

This is a semantic indexing task for a medical guideline update.

STRICT TASK RULES
1. Stay inside the assigned chapter. Do NOT remap the publication to another chapter.
2. Identify the smallest set of evidence units that captures the publication's substantive scientific content.
3. The PRIMARY evidence unit must correspond to the main research question, main exposure/intervention, main diagnostic question, main biological domain, or main synthesis domain.
4. Add SECONDARY evidence units only when the paper substantively studies those topics. Do not map background mentions, introductory statements, incidental covariates, or citations.
5. Do NOT assess evidence quality, risk of bias, treatment efficacy, recommendation strength, or whether the guideline should change.
6. Do NOT infer information that is absent from title/abstract.
7. Systematic reviews/meta-analyses spanning several substantive domains may receive multiple evidence units, but avoid indiscriminate overmapping.
8. If the publication clearly belongs to the assigned chapter but no predefined evidence unit adequately captures the main topic, return NOVEL_TOPIC and give a short, specific, medically meaningful novel-topic label.
9. If the title/abstract does not substantively support the existing chapter assignment, return CHAPTER_ASSIGNMENT_QUESTIONABLE rather than forcing a topic.
10. When a publication is very broad and genuinely covers more distinct evidence units than can reasonably be enumerated, set broad_multidomain_review=true and still return the most important units only.
11. Neuroendocrine pancreatic tumours and guideline/consensus documents were excluded upstream. Do not introduce them here.
12. Use only the article information and the evidence-unit ontology provided in the request.

OUTPUT DISCIPLINE
- primary_evidence_unit_id must be one allowed unit ID for the assigned chapter, NOVEL_TOPIC, or CHAPTER_ASSIGNMENT_QUESTIONABLE.
- secondary_evidence_unit_ids must contain only allowed unit IDs from the assigned chapter.
- Do not repeat the primary unit in secondary units.
- Use at most 6 secondary units.
- rationale must be one concise sentence explaining why the substantive content maps there.
- If mapping_status=novel_topic, novel_topic_label and novel_topic_description must be specific.
- If mapping_status is not novel_topic, both novel-topic fields must be empty strings.
- If mapping_status=chapter_assignment_questionable, use no secondary units.
Return only the requested structured JSON.
"""


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL line {line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_event(event: str, **payload: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_chapter_token(chapter_id: str) -> str:
    return chapter_id.replace(".", "_")


def unit_maps(ontology: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    chapter_unit_ids: dict[str, list[str]] = {}
    unit_meta: dict[str, dict[str, str]] = {}
    for cid, chapter in ontology["chapters"].items():
        ids = []
        for unit in chapter["evidence_units"]:
            uid = unit["id"]
            if uid in unit_meta:
                raise RuntimeError(f"Duplicate evidence unit ID in ontology: {uid}")
            ids.append(uid)
            unit_meta[uid] = {
                "chapter_id": cid,
                "chapter_title": chapter["title"],
                "unit_name": unit["name"],
                "definition": unit["definition"],
                "boundary": unit["boundary"],
            }
        chapter_unit_ids[cid] = ids
    return chapter_unit_ids, unit_meta


def ontology_prompt_for_chapter(ontology: dict[str, Any], cid: str) -> str:
    chapter = ontology["chapters"][cid]
    lines = [
        f"ASSIGNED CHAPTER: {cid} — {chapter['title']}",
        "",
        "ALLOWED EVIDENCE UNITS IN THIS CHAPTER:",
    ]
    for unit in chapter["evidence_units"]:
        lines.extend([
            "",
            f"{unit['id']} — {unit['name']}",
            f"Definition: {unit['definition']}",
            f"Boundary: {unit['boundary']}",
        ])
    return "\n".join(lines)


def response_schema_for_chapter(allowed_ids: list[str]) -> dict[str, Any]:
    primary_enum = allowed_ids + ["NOVEL_TOPIC", "CHAPTER_ASSIGNMENT_QUESTIONABLE"]
    return {
        "name": "pdac_evidence_unit_submapping",
        "description": "Precise evidence-unit submapping within one fixed guideline chapter.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mapping_status": {
                    "type": "string",
                    "enum": ["mapped", "novel_topic", "chapter_assignment_questionable"],
                },
                "primary_evidence_unit_id": {
                    "type": "string",
                    "enum": primary_enum,
                },
                "secondary_evidence_unit_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": allowed_ids,
                    },
                    "minItems": 0,
                    "maxItems": 6,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "broad_multidomain_review": {
                    "type": "boolean",
                },
                "novel_topic_label": {
                    "type": "string",
                },
                "novel_topic_description": {
                    "type": "string",
                },
                "rationale": {
                    "type": "string",
                },
            },
            "required": [
                "mapping_status",
                "primary_evidence_unit_id",
                "secondary_evidence_unit_ids",
                "confidence",
                "broad_multidomain_review",
                "novel_topic_label",
                "novel_topic_description",
                "rationale",
            ],
            "additionalProperties": False,
        },
    }


def make_custom_id(pmid: str, chapter_id: str) -> str:
    return f"pmid-{pmid}-chapter-{safe_chapter_token(chapter_id)}"


def parse_custom_id(custom_id: str) -> tuple[str, str]:
    m = re.fullmatch(r"pmid-(.+)-chapter-([0-9_]+)", custom_id)
    if not m:
        raise ValueError(f"Invalid custom_id: {custom_id}")
    pmid = m.group(1)
    chapter_id = m.group(2).replace("_", ".")
    return pmid, chapter_id


def prepare(model: str, reasoning_effort: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY_PATH)
    evidence = load_jsonl(INPUT_EVIDENCE)
    chapter_unit_ids, _ = unit_maps(ontology)

    seen = set()
    counts = Counter()

    BATCH_INPUT.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_INPUT.open("w", encoding="utf-8", newline="\n") as f:
        for rec in evidence:
            pmid = clean(rec.get("pmid"))
            cid = clean(rec.get("chapter_id"))

            if not pmid or not cid:
                raise RuntimeError("Evidence record missing PMID or chapter_id.")
            if cid not in ontology["chapters"]:
                raise RuntimeError(f"Chapter {cid} missing from ontology.")

            key = (pmid, cid)
            if key in seen:
                raise RuntimeError(f"Duplicate paper-chapter assignment: {key}")
            seen.add(key)

            chapter_ontology = ontology_prompt_for_chapter(ontology, cid)

            article_text = (
                f"{chapter_ontology}\n\n"
                "PUBLICATION TO SUBMAP:\n"
                f"PMID: {pmid}\n"
                f"Title: {clean(rec.get('title')) or '[missing]'}\n"
                f"Abstract: {clean(rec.get('abstract')) or '[no abstract available]'}\n"
                f"Publication types: {clean(rec.get('publication_types')) or '[not available]'}\n"
                f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}\n"
                f"MeSH terms: {clean(rec.get('mesh_terms')) or '[not available]'}\n"
                f"Keywords: {clean(rec.get('keywords')) or '[not available]'}\n"
                f"Publication year: {clean(rec.get('publication_year')) or '[not available]'}\n\n"
                "Map this publication ONLY within the assigned chapter."
            )

            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": article_text},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema_for_chapter(chapter_unit_ids[cid]),
                },
                "max_completion_tokens": 1800,
                "reasoning_effort": reasoning_effort,
            }

            req = {
                "custom_id": make_custom_id(pmid, cid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            counts[cid] += 1

    size_mb = BATCH_INPUT.stat().st_size / 1024 / 1024
    request_count = len(seen)

    if request_count > 50_000:
        raise RuntimeError(f"Batch has {request_count:,} requests; maximum is 50,000.")
    if size_mb > 190:
        raise RuntimeError(
            f"Prepared batch is {size_mb:.1f} MB. Refusing to approach the 200 MB upload limit. "
            "Split the submapping batch by chapter."
        )

    summary = {
        "ontology_version": ontology.get("version"),
        "evidence_unit_count": sum(
            len(ch["evidence_units"]) for ch in ontology["chapters"].values()
        ),
        "paper_chapter_assignments": request_count,
        "chapter_request_counts": dict(counts),
        "batch_jsonl_mb": round(size_mb, 2),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    print("\nEvidence-unit submapping batch prepared.")
    print(f"  ontology version:            {summary['ontology_version']}")
    print(f"  evidence units:              {summary['evidence_unit_count']}")
    print(f"  paper-chapter assignments:   {request_count:,}")
    print(f"  batch JSONL MB:              {size_mb:.2f}")
    print(f"  model:                       {model}")
    print(f"  reasoning effort:            {reasoning_effort}")
    print()
    for cid in ontology["chapters"]:
        print(f"  chapter {cid:>3}: {counts[cid]:,} requests")
    print(f"\n  batch input: {BATCH_INPUT}")

    log_event("prepared", **summary)
    return summary


class OpenAIHTTP:
    def __init__(self, api_key: str, retry_wait: int = 120):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty.")
        self.session = requests.Session()
        self.auth = {"Authorization": f"Bearer {api_key}"}
        self.retry_wait = retry_wait

    def request(self, method: str, path: str, *, timeout: int = 300, **kwargs: Any) -> requests.Response:
        url = path if path.startswith("http") else OPENAI_BASE_URL + path
        attempt = 0

        while True:
            attempt += 1
            headers = dict(self.auth)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.session.request(
                    method, url, headers=headers, timeout=timeout, **kwargs
                )
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.ChunkedEncodingError,
                requests.ContentDecodingError,
            ) as e:
                print(
                    f"WARN: transient OpenAI network/stream error "
                    f"({type(e).__name__}: {e}); retrying in {self.retry_wait}s"
                )
                log_event("transient_network_error", attempt=attempt, error=repr(e))
                time.sleep(self.retry_wait)
                continue

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                print(
                    f"WARN: transient OpenAI HTTP {r.status_code}; attempt {attempt}; "
                    f"retrying in {self.retry_wait}s"
                )
                log_event(
                    "transient_http_error",
                    attempt=attempt,
                    status_code=r.status_code,
                    response=clean(r.text[:1000]),
                )
                time.sleep(self.retry_wait)
                continue

            if r.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {r.status_code} for {path}: {r.text[:5000]}"
                )

            return r


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_and_create_batch(client: OpenAIHTTP, model: str, reasoning_effort: str) -> dict[str, Any]:
    state = load_state()
    if state.get("batch_id"):
        print(f"Existing submapping batch found: {state['batch_id']}")
        print("Resuming existing batch; no duplicate batch submitted.")
        return state

    if not BATCH_INPUT.exists():
        raise RuntimeError("Batch input missing. Run --mode prepare first.")

    print("\nUploading evidence-unit submapping JSONL...")
    with BATCH_INPUT.open("rb") as f:
        r = client.request(
            "POST",
            "/files",
            files={"file": (BATCH_INPUT.name, f, "application/jsonl")},
            data={"purpose": "batch"},
        )
    input_file_id = r.json()["id"]
    print(f"  uploaded file id: {input_file_id}")

    print("Creating OpenAI Batch...")
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "project": "ESMO_PDAC_2015_to_2023_PoC",
            "task": "evidence_unit_submapping",
            "model_requested": model,
            "reasoning_effort": reasoning_effort,
        },
    }
    r = client.request(
        "POST",
        "/batches",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    batch = r.json()

    state = {
        "input_file_id": input_file_id,
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "model_requested": model,
        "reasoning_effort": reasoning_effort,
    }
    save_state(state)
    log_event("batch_created", **state)

    print(f"  batch id: {state['batch_id']}")
    print(f"  status:   {state['status']}")
    return state


def download_file(client: OpenAIHTTP, file_id: str, destination: Path) -> None:
    r = client.request("GET", f"/files/{file_id}/content", timeout=600)
    destination.write_bytes(r.content)


def watch_batch(client: OpenAIHTTP, poll_seconds: int) -> dict[str, Any]:
    state = load_state()
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No batch_id in state.")

    print(f"\nWatching submapping batch {batch_id}...")
    last_status = None

    while True:
        r = client.request("GET", f"/batches/{batch_id}")
        batch = r.json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}

        print(
            f"  status={status}; total={counts.get('total')}; "
            f"completed={counts.get('completed')}; failed={counts.get('failed')}"
        )

        state.update({
            "status": status,
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "request_counts": counts,
        })
        save_state(state)

        if status in TERMINAL_BATCH_STATUSES:
            if batch.get("output_file_id"):
                print("Downloading successful batch output...")
                download_file(client, batch["output_file_id"], BATCH_OUTPUT)
            if batch.get("error_file_id"):
                print("Downloading batch error file...")
                download_file(client, batch["error_file_id"], BATCH_ERRORS)
            return batch

        last_status = status
        time.sleep(poll_seconds)


def extract_message_content(body: dict[str, Any]) -> str:
    return body["choices"][0]["message"]["content"]


def validate_mapping(
    parsed: dict[str, Any],
    cid: str,
    allowed_ids: list[str],
) -> dict[str, Any]:
    status = parsed["mapping_status"]
    primary = parsed["primary_evidence_unit_id"]
    secondary = list(parsed["secondary_evidence_unit_ids"])
    confidence = parsed["confidence"]
    broad = bool(parsed["broad_multidomain_review"])
    novel_label = clean(parsed["novel_topic_label"])
    novel_desc = clean(parsed["novel_topic_description"])
    rationale = clean(parsed["rationale"])

    allowed = set(allowed_ids)

    if status == "mapped":
        if primary not in allowed:
            raise ValueError(f"mapped status with invalid primary {primary}")
        if novel_label or novel_desc:
            raise ValueError("mapped status with non-empty novel-topic fields")
    elif status == "novel_topic":
        if primary != "NOVEL_TOPIC":
            raise ValueError("novel_topic status without NOVEL_TOPIC primary")
        if not novel_label or not novel_desc:
            raise ValueError("novel_topic requires label and description")
    elif status == "chapter_assignment_questionable":
        if primary != "CHAPTER_ASSIGNMENT_QUESTIONABLE":
            raise ValueError("questionable status without matching primary sentinel")
        if secondary:
            raise ValueError("questionable chapter assignment must have no secondary units")
        if novel_label or novel_desc:
            raise ValueError("questionable status with novel-topic fields")
    else:
        raise ValueError(f"invalid status {status}")

    invalid_secondary = [u for u in secondary if u not in allowed]
    if invalid_secondary:
        raise ValueError(f"invalid secondary unit(s): {invalid_secondary}")

    # Deterministic deduplication, preserving ontology order.
    secondary_set = set(secondary)
    if primary in secondary_set:
        secondary_set.remove(primary)
    ordered_secondary = [u for u in allowed_ids if u in secondary_set]

    return {
        "mapping_status": status,
        "primary_evidence_unit_id": primary,
        "secondary_evidence_unit_ids": ordered_secondary,
        "confidence": confidence,
        "broad_multidomain_review": broad,
        "novel_topic_label": novel_label,
        "novel_topic_description": novel_desc,
        "rationale": rationale,
    }


def parse_batch_output() -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    if not BATCH_OUTPUT.exists():
        raise RuntimeError(f"Missing batch output: {BATCH_OUTPUT}")

    ontology = load_json(ONTOLOGY_PATH)
    chapter_unit_ids, _ = unit_maps(ontology)

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
            except Exception as e:
                failures.append({
                    "line": line_no,
                    "custom_id": custom_id,
                    "error": f"custom_id parse error: {e}",
                })
                continue

            response = obj.get("response")
            error = obj.get("error")

            if error or not response:
                failures.append({
                    "pmid": pmid,
                    "chapter_id": cid,
                    "line": line_no,
                    "error": error or "missing response",
                })
                continue

            if response.get("status_code") != 200:
                failures.append({
                    "pmid": pmid,
                    "chapter_id": cid,
                    "line": line_no,
                    "error": f"HTTP {response.get('status_code')}",
                })
                continue

            try:
                content = extract_message_content(response["body"])
                parsed = json.loads(content)
                validated = validate_mapping(parsed, cid, chapter_unit_ids[cid])
            except Exception as e:
                failures.append({
                    "pmid": pmid,
                    "chapter_id": cid,
                    "line": line_no,
                    "error": f"mapping parse/validation error: {type(e).__name__}: {e}",
                })
                continue

            key = (pmid, cid)
            if key in results:
                failures.append({
                    "pmid": pmid,
                    "chapter_id": cid,
                    "line": line_no,
                    "error": "duplicate batch result for paper-chapter assignment",
                })
                continue

            results[key] = validated

    return results, failures


def merge(model: str) -> dict[str, Any]:
    ontology = load_json(ONTOLOGY_PATH)
    evidence = load_jsonl(INPUT_EVIDENCE)
    chapter_unit_ids, unit_meta = unit_maps(ontology)
    results, failures = parse_batch_output()
    state = load_state()

    flat_rows: list[dict[str, str]] = []
    expanded: list[dict[str, Any]] = []
    novel_rows: list[dict[str, Any]] = []
    questionable_rows: list[dict[str, Any]] = []

    status_counts = Counter()
    confidence_counts = Counter()
    unit_counts = Counter()
    broad_count = 0
    missing = []

    all_fields = set()

    for rec in evidence:
        pmid = clean(rec.get("pmid"))
        cid = clean(rec.get("chapter_id"))
        key = (pmid, cid)
        mapping = results.get(key)

        row = {k: clean(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
               for k, v in rec.items()}

        if not mapping:
            missing.append({"pmid": pmid, "chapter_id": cid})
            row.update({
                "submapping_status": "",
                "primary_evidence_unit_id": "",
                "primary_evidence_unit_name": "",
                "secondary_evidence_unit_ids": "",
                "secondary_evidence_unit_names": "",
                "all_evidence_unit_ids": "",
                "all_evidence_unit_names": "",
                "submapping_confidence": "",
                "submapping_rationale": "",
                "broad_multidomain_review": "",
                "novel_topic_label": "",
                "novel_topic_description": "",
                "submapping_model": model,
                "submapping_batch_id": clean(state.get("batch_id")),
            })
            flat_rows.append(row)
            all_fields.update(row.keys())
            continue

        status = mapping["mapping_status"]
        primary = mapping["primary_evidence_unit_id"]
        secondary = mapping["secondary_evidence_unit_ids"]
        confidence = mapping["confidence"]
        broad = mapping["broad_multidomain_review"]

        status_counts[status] += 1
        confidence_counts[confidence] += 1
        broad_count += int(broad)

        if status == "mapped":
            all_ids = [primary] + secondary
            primary_name = unit_meta[primary]["unit_name"]
            secondary_names = [unit_meta[u]["unit_name"] for u in secondary]
            all_names = [unit_meta[u]["unit_name"] for u in all_ids]

            for uid in all_ids:
                unit_counts[uid] += 1
                expanded_rec = dict(rec)
                expanded_rec.update({
                    "evidence_unit_id": uid,
                    "evidence_unit_name": unit_meta[uid]["unit_name"],
                    "evidence_unit_definition": unit_meta[uid]["definition"],
                    "submapping_role": "primary" if uid == primary else "secondary",
                    "submapping_confidence": confidence,
                    "submapping_rationale": mapping["rationale"],
                    "broad_multidomain_review": broad,
                    "submapping_model": model,
                    "submapping_batch_id": clean(state.get("batch_id")),
                })
                expanded.append(expanded_rec)
        else:
            all_ids = []
            primary_name = ""
            secondary_names = []
            all_names = []

            special = dict(rec)
            special.update({
                "submapping_status": status,
                "submapping_confidence": confidence,
                "submapping_rationale": mapping["rationale"],
                "broad_multidomain_review": broad,
                "novel_topic_label": mapping["novel_topic_label"],
                "novel_topic_description": mapping["novel_topic_description"],
                "submapping_model": model,
                "submapping_batch_id": clean(state.get("batch_id")),
            })
            if status == "novel_topic":
                novel_rows.append(special)
            else:
                questionable_rows.append(special)

        row.update({
            "submapping_status": status,
            "primary_evidence_unit_id": primary,
            "primary_evidence_unit_name": primary_name,
            "secondary_evidence_unit_ids": "; ".join(secondary),
            "secondary_evidence_unit_names": "; ".join(secondary_names),
            "all_evidence_unit_ids": "; ".join(all_ids),
            "all_evidence_unit_names": "; ".join(all_names),
            "submapping_confidence": confidence,
            "submapping_rationale": mapping["rationale"],
            "broad_multidomain_review": "1" if broad else "0",
            "novel_topic_label": mapping["novel_topic_label"],
            "novel_topic_description": mapping["novel_topic_description"],
            "submapping_model": model,
            "submapping_batch_id": clean(state.get("batch_id")),
        })
        flat_rows.append(row)
        all_fields.update(row.keys())

    # Deterministic column order.
    preferred = [
        "pmid", "doi", "title", "abstract", "journal", "publication_year",
        "publication_types", "evidence_labels", "chapter_id", "chapter_title",
        "submapping_status",
        "primary_evidence_unit_id", "primary_evidence_unit_name",
        "secondary_evidence_unit_ids", "secondary_evidence_unit_names",
        "all_evidence_unit_ids", "all_evidence_unit_names",
        "submapping_confidence", "submapping_rationale",
        "broad_multidomain_review",
        "novel_topic_label", "novel_topic_description",
        "submapping_model", "submapping_batch_id",
    ]
    fieldnames = [f for f in preferred if f in all_fields]
    fieldnames += sorted(all_fields - set(fieldnames), key=str.casefold)

    with SUBMAPPED_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    write_jsonl(EXPANDED_JSONL, expanded)
    write_jsonl(NOVEL_JSONL, novel_rows)
    write_jsonl(QUESTIONABLE_JSONL, questionable_rows)
    write_jsonl(PARSE_FAILURES, failures)

    manifest = {
        "ontology_version": ontology.get("version"),
        "model": model,
        "reasoning_effort": state.get("reasoning_effort"),
        "batch_id": state.get("batch_id"),
        "input_paper_chapter_assignments": len(evidence),
        "successful_submappings": len(results),
        "missing_submappings": len(missing),
        "parse_failures": len(failures),
        "mapping_status_counts": dict(status_counts),
        "confidence_counts": dict(confidence_counts),
        "broad_multidomain_review_count": broad_count,
        "expanded_paper_unit_assignments": len(expanded),
        "novel_topic_records": len(novel_rows),
        "questionable_chapter_assignments": len(questionable_rows),
        "evidence_unit_counts": {
            uid: {
                "chapter_id": unit_meta[uid]["chapter_id"],
                "unit_name": unit_meta[uid]["unit_name"],
                "paper_assignments": unit_counts[uid],
            }
            for uid in unit_meta
        },
        "missing_records": missing,
        "safety": {
            "cross_chapter_remapping_performed": False,
            "evidence_quality_assessment_performed": False,
            "guideline_rewriting_performed": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nEvidence-unit submapping merge completed.")
    print(f"  input paper-chapter assignments:  {len(evidence):,}")
    print(f"  successful submappings:           {len(results):,}")
    print(f"  missing submappings:              {len(missing):,}")
    print(f"  parse failures:                   {len(failures):,}")
    print(f"  mapped:                           {status_counts['mapped']:,}")
    print(f"  novel topics:                     {status_counts['novel_topic']:,}")
    print(f"  chapter assignment questionable:  {status_counts['chapter_assignment_questionable']:,}")
    print(f"  broad multidomain reviews:        {broad_count:,}")
    print(f"  expanded paper-unit assignments:  {len(expanded):,}")
    print()
    print(f"  submapped CSV:   {SUBMAPPED_CSV}")
    print(f"  expanded JSONL:  {EXPANDED_JSONL}")
    print(f"  novel topics:    {NOVEL_JSONL}")
    print(f"  questionable:    {QUESTIONABLE_JSONL}")
    print(f"  manifest:        {MANIFEST_PATH}")
    print(f"  parse failures:  {PARSE_FAILURES}")

    if missing or failures:
        print("\nWARNING: Do not start evidence synthesis until missing/failed mappings are repaired.")

    return manifest


def synchronous_test(client: OpenAIHTTP) -> None:
    if not BATCH_INPUT.exists():
        raise RuntimeError("Batch input missing. Run --mode prepare first.")

    with BATCH_INPUT.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    print(f"\nTesting first request synchronously: {first['custom_id']}")
    r = client.request(
        "POST",
        "/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(first["body"]),
    )
    print(f"HTTP {r.status_code}")
    body = r.json()
    print(json.dumps(body, ensure_ascii=False, indent=2)[:6000])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precisely submap paper-chapter assignments to thematic evidence units."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "test", "submit", "watch", "merge", "all"],
        default="prepare",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete local submapping batch state before creating a NEW batch.",
    )
    args = parser.parse_args()

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()
        print(f"Deleted previous state: {STATE_PATH}")

    if args.mode in {"prepare", "all"}:
        prepare(args.model, args.reasoning_effort)
        if args.mode == "prepare":
            return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if args.mode in {"test", "submit", "watch", "all"} and not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in this PowerShell session.")

    client = OpenAIHTTP(api_key, retry_wait=args.retry_wait) if api_key else None

    if args.mode == "test":
        synchronous_test(client)
        return 0

    if args.mode in {"submit", "all"}:
        upload_and_create_batch(client, args.model, args.reasoning_effort)
        if args.mode == "submit":
            return 0

    if args.mode in {"watch", "all"}:
        batch = watch_batch(client, args.poll_seconds)
        status = batch.get("status")
        print(f"\nBatch terminal status: {status}")
        if status != "completed":
            print("Batch did not complete successfully. No automatic paid resubmission was performed.")
            return 2
        merge(args.model)
        return 0

    if args.mode == "merge":
        merge(args.model)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
