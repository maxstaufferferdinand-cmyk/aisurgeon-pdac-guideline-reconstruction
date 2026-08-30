#!/usr/bin/env python3
"""
GPT chapter mapping for the ESMO PDAC 2015 -> August 2023 PoC.

Input:
    data/pubmed_selected_evidence.csv

Hard exclusion BEFORE GPT:
    - exclude_guideline_consensus == 1
    - possible_guidance_title_review == 1
    - additional conservative title safety patterns for guideline / consensus /
      position / Delphi guidance documents

GPT task:
    Map each remaining publication to one or more of the 8 predefined ESMO-PDAC
    chapters based ONLY on article title, abstract, publication type, MeSH terms,
    and keywords. The original PubMed search/chapter mapping is NOT sent to GPT.

Uses the OpenAI Batch API:
    - one article per request
    - Structured Outputs (strict JSON schema)
    - resumable via state JSON
    - network/5xx retry with long waits
    - batch results are merged back to CSV

Default model:
    gpt-5.1
Override with:
    --model MODEL_ID
or environment variable:
    OPENAI_MODEL
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

DEFAULT_INPUT = DATA_DIR / "pubmed_selected_evidence.csv"
DEFAULT_MAPPING_INPUT = DATA_DIR / "pubmed_mapping_input.csv"
DEFAULT_EXCLUDED = DATA_DIR / "pubmed_mapping_excluded_guidance.csv"
DEFAULT_BATCH_JSONL = DATA_DIR / "gpt_chapter_mapping_batch_input.jsonl"
DEFAULT_BATCH_OUTPUT = DATA_DIR / "gpt_chapter_mapping_batch_output.jsonl"
DEFAULT_BATCH_ERRORS = DATA_DIR / "gpt_chapter_mapping_batch_errors.jsonl"
DEFAULT_MAPPED_CSV = DATA_DIR / "pubmed_selected_evidence_mapped.csv"
DEFAULT_STATE = DATA_DIR / "gpt_chapter_mapping_state.json"
DEFAULT_LOG = LOG_DIR / "gpt_chapter_mapping_log.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"

TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

CHAPTERS = {
    "1": "Incidence and epidemiology",
    "2": "Diagnosis and pathology/molecular biology",
    "3": "Staging and risk assessment",
    "4.1": "Treatment of localised disease",
    "4.2": "Treatment of non-resectable disease – borderline resectable / locally advanced",
    "4.3": "Treatment of advanced/metastatic disease",
    "5": "Personalised medicine",
    "6": "Follow-up and long-term implications",
}

# The 27 "possible guidance" records are now HARD EXCLUDED by user decision.
# These title patterns are an additional safety net, not a substitute for the
# flags created in the previous deterministic classification step.
GUIDANCE_SAFETY_PATTERNS = [
    re.compile(r"\bclinical practice guidelines?\b", re.I),
    re.compile(r"\bpractice guidelines?\b", re.I),
    re.compile(r"\bconsensus statement\b", re.I),
    re.compile(r"\bexpert consensus\b", re.I),
    re.compile(r"\bconsensus recommendations?\b", re.I),
    re.compile(r"\bconsensus guidelines?\b", re.I),
    re.compile(r"\bconsensus report\b", re.I),
    re.compile(r"\bdelphi consensus\b", re.I),
    re.compile(r"\bposition statement\b", re.I),
    re.compile(r"\bposition paper\b", re.I),
    re.compile(r"\bguideline update\b", re.I),
    re.compile(r"^\s*(?:updated\s+|update\s+of\s+)?guidelines?\s+(?:for|on|of)\b", re.I),
    re.compile(r"\b(?:ESMO|ASCO|NCCN)\b.{0,100}\bguidelines?\b", re.I),
]

SYSTEM_PROMPT = """You are performing a single, narrow evidence-mapping task for a proof-of-concept update of the 2015 ESMO pancreatic cancer guideline.

TASK
Assign the publication to ONE OR MORE of the predefined guideline chapters below based on the actual scientific content of the TITLE and ABSTRACT. Use publication type, MeSH terms, and keywords only as secondary context.

IMPORTANT RULES
1. Do NOT judge study quality, treatment efficacy, recommendation strength, or whether the findings should change a guideline.
2. Do NOT summarize the study beyond the short mapping rationale.
3. Do NOT infer content that is not supported by title/abstract.
4. A paper may map to multiple chapters if it genuinely addresses multiple chapter domains.
5. Do not force a mapping. If the available information is insufficient or none of the chapters fits, set unmappable=true and chapter_ids=[].
6. Do not use search provenance or prior chapter assignments. They are intentionally not provided.
7. Map according to the disease/content studied, not merely words mentioned in background text.
8. Neuroendocrine pancreatic tumors are outside scope; however, the input has already been filtered and your task is chapter mapping only.

CHAPTER DEFINITIONS

1 — Incidence and epidemiology
Population incidence, prevalence, mortality, survival trends, demographics, geographic burden, hereditary/familial predisposition, high-risk populations, screening of genetically/familially high-risk people, and etiologic/risk-factor epidemiology such as smoking, obesity, diabetes, pancreatitis, diet, alcohol, infection, occupational/environmental exposure.

2 — Diagnosis and pathology/molecular biology
Clinical presentation and symptoms; diagnostic recognition; histopathology and morphology; pathological subtypes; precursor and cystic lesions relevant to pancreatic carcinoma; PanIN/IPMN/mucinous lesions; tumor stroma/microenvironment when studied as biology; molecular pathogenesis, carcinogenesis, genomics, mutations, structural variation, and biological subtypes when NOT principally about biomarker-guided therapy.

3 — Staging and risk assessment
TNM/stage, CA 19-9 as staging/prognostic disease-burden marker, CT/MRI/MRCP/EUS/PET/ERCP for staging, biopsy/tissue acquisition, lymph-node/metastasis detection, vascular involvement, resectability assessment/criteria, staging laparoscopy, performance status, nutritional/comorbidity assessment when used to assess treatment eligibility or risk.

4.1 — Treatment of localised disease
Resectable/localised pancreatic cancer. Upfront surgery, pancreatectomy, pancreaticoduodenectomy, distal pancreatectomy, minimally invasive surgery, vascular resection in resectable disease, margins/R0/R1, specimen assessment, lymphadenectomy, perioperative risk, preoperative biliary drainage, adjuvant therapy, postoperative chemotherapy/chemoradiation, and neoadjuvant/perioperative strategies specifically in primarily resectable disease.

4.2 — Treatment of non-resectable disease – borderline resectable / locally advanced
Borderline-resectable or locally advanced non-metastatic pancreatic cancer. Neoadjuvant/induction chemotherapy, chemoradiation/radiotherapy, FOLFIRINOX or other systemic regimens in these stages, conversion/downstaging, secondary resection, local control, SBRT/IMRT, ablative/local therapies, and multimodality treatment.

4.3 — Treatment of advanced/metastatic disease
Metastatic/advanced pancreatic cancer. Palliative/supportive interventions (biliary/duodenal obstruction, stents, pain, celiac plexus procedures, nutrition/enzyme support in advanced disease), systemic first-line or later-line treatment, treatment sequencing, response monitoring, and treatment of rare exocrine forms when the principal context is advanced/metastatic disease.

5 — Personalised medicine
Predictive/prognostic molecular biomarkers used for treatment stratification; germline/somatic testing for treatment; precision oncology; actionable alterations; BRCA/PALB2/ATM/HRD/DNA-repair deficiency; MSI/dMMR/TMB; NTRK/NRG1/BRAF/HER2/KRAS and other targetable alterations; PARP inhibitors, biomarker-selected platinum treatment, immunotherapy or other molecularly targeted therapy; liquid biopsy when used for molecular treatment selection.

6 — Follow-up and long-term implications
Post-curative surveillance/follow-up, recurrence detection, surveillance imaging/CA 19-9, follow-up schedules or intensity, survivorship, and long-term symptom/nutritional/psychosocial care specifically in the post-treatment follow-up setting.

OUTPUT
Return only the structured JSON requested by the schema.
"""


# OpenAI Structured Outputs supports only a subset of JSON Schema.
# Array `uniqueItems` is not supported; uniqueness is enforced deterministically
# after parsing by the merge step.
RESPONSE_SCHEMA = {
    "name": "pdac_chapter_mapping",
    "description": "Chapter mapping of one pancreatic-cancer evidence record.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chapter_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["1", "2", "3", "4.1", "4.2", "4.3", "5", "6"],
                },
                "minItems": 0,
                "maxItems": 8,
            },
            "unmappable": {"type": "boolean"},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "rationale": {
                "type": "string",
                "description": "One concise sentence explaining the content-based mapping.",
            },
        },
        "required": ["chapter_ids", "unmappable", "confidence", "rationale"],
        "additionalProperties": False,
    },
}


def clean(value: str | None) -> str:
    return " ".join((value or "").replace("\ufeff", "").split())


def truthy(value: str | None) -> bool:
    return clean(value).casefold() in {"1", "true", "yes", "y"}


def title_guidance_safety_match(title: str) -> bool:
    return any(p.search(title or "") for p in GUIDANCE_SAFETY_PATTERNS)


def log_event(event: str, **payload: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **payload,
    }
    with DEFAULT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV header in {path}")
        return list(reader), list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def guidance_exclusion_reason(row: dict[str, str]) -> str:
    reasons = []
    if truthy(row.get("exclude_guideline_consensus")):
        reasons.append("exclude_guideline_consensus=1")
    if truthy(row.get("possible_guidance_title_review")):
        reasons.append("possible_guidance_title_review=1")
    if title_guidance_safety_match(clean(row.get("title"))):
        reasons.append("guidance_title_safety_pattern")
    return "; ".join(reasons)


def prepare_mapping_input(input_path: Path, model: str, reasoning_effort: str) -> dict[str, Any]:
    rows, fields = load_csv(input_path)

    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    seen_pmids: set[str] = set()

    for row in rows:
        pmid = clean(row.get("pmid"))
        if not pmid:
            continue
        if pmid in seen_pmids:
            raise RuntimeError(f"Duplicate PMID in selected evidence input: {pmid}")
        seen_pmids.add(pmid)

        reason = guidance_exclusion_reason(row)
        if reason:
            copy = dict(row)
            copy["mapping_exclusion_reason"] = reason
            excluded.append(copy)
        else:
            eligible.append(row)

    excluded_fields = fields + ([] if "mapping_exclusion_reason" in fields else ["mapping_exclusion_reason"])
    write_csv(DEFAULT_MAPPING_INPUT, eligible, fields)
    write_csv(DEFAULT_EXCLUDED, excluded, excluded_fields)

    DEFAULT_BATCH_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with DEFAULT_BATCH_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for row in eligible:
            pmid = clean(row.get("pmid"))
            article_text = (
                f"PMID: {pmid}\n"
                f"Title: {clean(row.get('title')) or '[missing]'}\n"
                f"Abstract: {clean(row.get('abstract')) or '[no abstract available]'}\n"
                f"Publication types: {clean(row.get('publication_types')) or '[not available]'}\n"
                f"MeSH terms: {clean(row.get('mesh_terms')) or '[not available]'}\n"
                f"Keywords: {clean(row.get('keywords')) or '[not available]'}\n"
                f"Publication year: {clean(row.get('publication_year')) or '[not available]'}"
            )

            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": article_text},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": RESPONSE_SCHEMA,
                },
                "max_completion_tokens": 2000,
            }

            # GPT-5.x models support reasoning_effort; users can set "none" for a
            # pure classification pass or "low" for a little more deliberation.
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort

            request_obj = {
                "custom_id": f"pmid-{pmid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    size_bytes = DEFAULT_BATCH_JSONL.stat().st_size
    request_count = len(eligible)

    if request_count > 50_000:
        raise RuntimeError(f"Batch has {request_count:,} requests; OpenAI Batch API max is 50,000.")
    if size_bytes > 200 * 1024 * 1024:
        raise RuntimeError(
            f"Batch JSONL is {size_bytes / 1024 / 1024:.1f} MB; OpenAI Batch API max is 200 MB."
        )

    summary = {
        "source_selected_rows": len(rows),
        "mapping_eligible_rows": len(eligible),
        "guidance_excluded_before_gpt": len(excluded),
        "batch_requests": request_count,
        "batch_jsonl_mb": round(size_bytes / 1024 / 1024, 2),
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    print("\nMapping input prepared.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  mapping input CSV: {DEFAULT_MAPPING_INPUT}")
    print(f"  excluded guidance: {DEFAULT_EXCLUDED}")
    print(f"  batch JSONL:       {DEFAULT_BATCH_JSONL}")

    log_event("prepared", **summary)
    return summary


class OpenAIHTTP:
    def __init__(self, api_key: str, retry_wait: int = 120):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty.")
        self.api_key = api_key
        self.retry_wait = retry_wait
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 180,
        retry_forever: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        url = path if path.startswith("http") else OPENAI_BASE_URL + path
        attempt = 0

        while True:
            attempt += 1
            try:
                r = self.session.request(
                    method,
                    url,
                    headers={**self.headers, **kwargs.pop("headers", {})},
                    timeout=timeout,
                    **kwargs,
                )
            except (requests.Timeout, requests.ConnectionError, requests.ChunkedEncodingError) as e:
                if not retry_forever:
                    raise
                print(
                    f"WARN: transient OpenAI network error: {type(e).__name__}: {e}; "
                    f"retrying in {self.retry_wait}s"
                )
                log_event("transient_network_error", attempt=attempt, error=repr(e))
                time.sleep(self.retry_wait)
                continue

            if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                if not retry_forever:
                    r.raise_for_status()
                body = clean(r.text[:500])
                print(
                    f"WARN: transient OpenAI HTTP {r.status_code}; attempt {attempt}; "
                    f"retrying in {self.retry_wait}s; {body}"
                )
                log_event(
                    "transient_http_error",
                    attempt=attempt,
                    status_code=r.status_code,
                    response=body,
                )
                time.sleep(self.retry_wait)
                continue

            # Authentication, schema, billing, or other permanent request errors
            # should not be retried forever.
            if r.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI HTTP {r.status_code} for {path}: {r.text[:3000]}"
                )

            return r


def save_state(state: dict[str, Any]) -> None:
    DEFAULT_STATE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state() -> dict[str, Any]:
    if not DEFAULT_STATE.exists():
        return {}
    return json.loads(DEFAULT_STATE.read_text(encoding="utf-8"))


def upload_and_create_batch(client: OpenAIHTTP, model: str) -> dict[str, Any]:
    state = load_state()
    if state.get("batch_id"):
        print(f"Existing batch state found: {state['batch_id']}")
        print("Will resume that batch instead of submitting duplicate paid work.")
        return state

    if not DEFAULT_BATCH_JSONL.exists():
        raise RuntimeError("Batch JSONL does not exist. Run prepare first.")

    print("\nUploading Batch JSONL to OpenAI...")
    with DEFAULT_BATCH_JSONL.open("rb") as f:
        r = client.request(
            "POST",
            "/files",
            files={"file": (DEFAULT_BATCH_JSONL.name, f, "application/jsonl")},
            data={"purpose": "batch"},
        )
    file_obj = r.json()
    input_file_id = file_obj["id"]
    print(f"  uploaded file id: {input_file_id}")

    print("Creating OpenAI Batch...")
    r = client.request(
        "POST",
        "/batches",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {
                    "project": "ESMO_PDAC_2015_to_2023_PoC",
                    "task": "chapter_mapping_only",
                    "model_requested": model,
                },
            }
        ),
    )
    batch = r.json()

    state = {
        "input_file_id": input_file_id,
        "batch_id": batch["id"],
        "status": batch.get("status"),
        "created_at": batch.get("created_at"),
        "model_requested": model,
    }
    save_state(state)
    log_event("batch_created", **state)

    print(f"  batch id: {state['batch_id']}")
    print(f"  status:   {state['status']}")
    return state


def download_file_content(client: OpenAIHTTP, file_id: str, destination: Path) -> None:
    r = client.request("GET", f"/files/{file_id}/content", timeout=300)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(r.content)


def watch_batch(client: OpenAIHTTP, poll_seconds: int) -> dict[str, Any]:
    state = load_state()
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No batch_id in state. Submit the batch first.")

    last_status = None
    print(f"\nWatching batch {batch_id}...")

    while True:
        r = client.request("GET", f"/batches/{batch_id}")
        batch = r.json()
        status = batch.get("status")
        counts = batch.get("request_counts") or {}

        if status != last_status:
            print(
                f"  status={status}; total={counts.get('total')}; "
                f"completed={counts.get('completed')}; failed={counts.get('failed')}"
            )
            last_status = status
        else:
            print(
                f"  status={status}; completed={counts.get('completed')}/"
                f"{counts.get('total')}; failed={counts.get('failed')}"
            )

        state.update(
            {
                "status": status,
                "output_file_id": batch.get("output_file_id"),
                "error_file_id": batch.get("error_file_id"),
                "request_counts": counts,
                "usage": batch.get("usage"),
            }
        )
        save_state(state)
        log_event("batch_poll", batch_id=batch_id, status=status, request_counts=counts)

        if status in TERMINAL_BATCH_STATUSES:
            if batch.get("output_file_id"):
                print("Downloading successful batch output...")
                download_file_content(
                    client, batch["output_file_id"], DEFAULT_BATCH_OUTPUT
                )
            if batch.get("error_file_id"):
                print("Downloading batch error file...")
                download_file_content(
                    client, batch["error_file_id"], DEFAULT_BATCH_ERRORS
                )
            return batch

        time.sleep(poll_seconds)


def parse_batch_output() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not DEFAULT_BATCH_OUTPUT.exists():
        raise RuntimeError(f"Batch output file missing: {DEFAULT_BATCH_OUTPUT}")

    mappings: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    with DEFAULT_BATCH_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            pmid = custom_id.removeprefix("pmid-")
            response = obj.get("response")
            error = obj.get("error")

            if error or not response:
                failures.append({"pmid": pmid, "line": line_no, "error": error or "missing response"})
                continue

            if response.get("status_code") != 200:
                failures.append(
                    {
                        "pmid": pmid,
                        "line": line_no,
                        "error": f"HTTP {response.get('status_code')}",
                    }
                )
                continue

            try:
                body = response["body"]
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except Exception as e:
                failures.append(
                    {
                        "pmid": pmid,
                        "line": line_no,
                        "error": f"parse error: {type(e).__name__}: {e}",
                    }
                )
                continue

            chapter_ids = parsed.get("chapter_ids", [])
            unmappable = bool(parsed.get("unmappable"))

            # Deterministic consistency repair/validation.
            if unmappable:
                chapter_ids = []
            elif not chapter_ids:
                failures.append(
                    {
                        "pmid": pmid,
                        "line": line_no,
                        "error": "model returned unmappable=false but empty chapter_ids",
                    }
                )
                continue

            invalid = [x for x in chapter_ids if x not in CHAPTERS]
            if invalid:
                failures.append(
                    {
                        "pmid": pmid,
                        "line": line_no,
                        "error": f"invalid chapter ids: {invalid}",
                    }
                )
                continue

            # Preserve chapter order according to guideline, not arbitrary model order.
            ordered = [cid for cid in CHAPTERS if cid in set(chapter_ids)]

            mappings[pmid] = {
                "gpt_chapter_ids": "; ".join(ordered),
                "gpt_chapter_titles": "; ".join(CHAPTERS[c] for c in ordered),
                "gpt_mapping_confidence": parsed.get("confidence", ""),
                "gpt_mapping_rationale": clean(parsed.get("rationale")),
                "gpt_unmappable": "1" if unmappable else "0",
            }

    return mappings, failures


def merge_results(model: str) -> None:
    input_rows, fields = load_csv(DEFAULT_MAPPING_INPUT)
    mappings, failures = parse_batch_output()

    state = load_state()
    batch_id = state.get("batch_id", "")

    added_fields = [
        "gpt_chapter_ids",
        "gpt_chapter_titles",
        "gpt_mapping_confidence",
        "gpt_mapping_rationale",
        "gpt_unmappable",
        "gpt_model",
        "gpt_batch_id",
    ]
    out_fields = fields + [x for x in added_fields if x not in fields]

    mapped_rows = []
    missing = []

    for row in input_rows:
        pmid = clean(row.get("pmid"))
        result = mappings.get(pmid)
        out = dict(row)
        if result:
            out.update(result)
            out["gpt_model"] = model
            out["gpt_batch_id"] = batch_id
        else:
            missing.append(pmid)
            for field in added_fields:
                out.setdefault(field, "")
        mapped_rows.append(out)

    write_csv(DEFAULT_MAPPED_CSV, mapped_rows, out_fields)

    qc = {
        "mapping_input_rows": len(input_rows),
        "successful_mappings": len(mappings),
        "missing_or_failed_mappings": len(missing),
        "batch_parse_failures": len(failures),
        "unmappable": sum(1 for r in mappings.values() if r["gpt_unmappable"] == "1"),
        "high_confidence": sum(1 for r in mappings.values() if r["gpt_mapping_confidence"] == "high"),
        "medium_confidence": sum(1 for r in mappings.values() if r["gpt_mapping_confidence"] == "medium"),
        "low_confidence": sum(1 for r in mappings.values() if r["gpt_mapping_confidence"] == "low"),
    }

    failure_path = DATA_DIR / "gpt_chapter_mapping_parse_failures.jsonl"
    with failure_path.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    qc_path = DATA_DIR / "gpt_chapter_mapping_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nGPT chapter mapping merge completed.")
    for k, v in qc.items():
        print(f"  {k}: {v}")
    print(f"  mapped CSV: {DEFAULT_MAPPED_CSV}")
    print(f"  QC JSON:    {qc_path}")
    print(f"  parse failures: {failure_path}")

    if missing:
        print(
            "\nWARNING: Some PMIDs have no successful mapping result. "
            "Do NOT proceed to downstream synthesis until these are repaired."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, watch, and merge GPT chapter mapping via OpenAI Batch API."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "submit", "watch", "merge", "all"],
        default="all",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Selected evidence CSV.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.1"),
        help="OpenAI model ID; default OPENAI_MODEL or gpt-5.1.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for GPT-5.x; default low.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="Batch status polling interval; default 300 seconds.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=120,
        help="Retry wait for transient OpenAI/network errors; default 120 seconds.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete local batch state before preparing/submitting a NEW paid batch.",
    )
    args = parser.parse_args()

    if args.reset_state and DEFAULT_STATE.exists():
        DEFAULT_STATE.unlink()
        print(f"Deleted state: {DEFAULT_STATE}")

    if args.mode in {"prepare", "all"}:
        prepare_mapping_input(args.input, args.model, args.reasoning_effort)
        if args.mode == "prepare":
            return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if args.mode in {"submit", "watch", "all"} and not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in this PowerShell session before submitting/watching."
        )

    client = OpenAIHTTP(api_key, retry_wait=args.retry_wait) if api_key else None

    if args.mode in {"submit", "all"}:
        upload_and_create_batch(client, args.model)
        if args.mode == "submit":
            return 0

    if args.mode in {"watch", "all"}:
        batch = watch_batch(client, args.poll_seconds)
        status = batch.get("status")
        print(f"\nBatch terminal status: {status}")

        if status != "completed":
            print(
                "Batch did not complete successfully. State/output/error files were preserved. "
                "No automatic paid resubmission was performed."
            )
            return 2

        merge_results(args.model)
        return 0

    if args.mode == "merge":
        merge_results(args.model)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
