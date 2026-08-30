#!/usr/bin/env python3
"""
FINAL STAGE A — Evidence synthesis by frozen evidence unit.

Model default:
    gpt-5.6-sol
Reasoning:
    high

This runner DOES NOT rewrite the guideline yet.
It performs the final evidence appraisal and synthesis that will feed the
subsequent guideline-update/rewrite stage.

Core design
-----------
- Medical evidence-unit boundaries are FROZEN and never split for paper count.
- Large units are split only into TECHNICAL chunks.
- Every mapped paper remains visible to GPT.
- GPT may classify a paper as MAIN_SYNTHESIS, CONTEXT_ONLY, APPENDIX, or REJECT.
- No deterministic pre-filter removes OTHER_REVIEW records.
- Every paper decision is audit-preserved.

USER-DEFINED EVIDENCE HIERARCHY
-------------------------------
Tier 1: Meta-analysis of HUMAN randomized controlled trials.
Tier 2: Meta-analysis of HUMAN retrospective/non-randomized studies, including
        mixed meta-analyses that are not demonstrably restricted to RCTs.
Tier 3: Systematic review.
Tier 4: Other review OR standalone randomized controlled trial.

Within any tier, directness, human clinical relevance, endpoint relevance,
population fit, consistency and clinical translatability still matter.

Important special rule:
- A standalone RCT can directly support clinical synthesis when clinically
  relevant, despite being Tier 4 in the user-defined hierarchy.
- An OTHER_REVIEW may provide useful context/support but should not, by itself,
  justify ADD/MODIFY/REMOVE of a clinical recommendation.

THERAPEUTIC ENDPOINT POLICY
---------------------------
For therapeutic/interventional evidence, recommendation-driving evidence should
have clinically meaningful patient-relevant outcomes. Surrogate-only,
mechanistic, preclinical, non-human, indirect, or insufficiently translatable
evidence can be placed in CONTEXT_ONLY / APPENDIX / REJECT.

Non-therapeutic domains use domain-appropriate clinical logic:
- epidemiology/risk/screening
- diagnosis/pathology
- staging/prognosis/resectability
- personalised medicine
- follow-up/survivorship

Inputs
------
data/guideline_integration_master_v2.jsonl

Outputs — chunk stage
---------------------
data/stageA_evidence_chunk_plan.jsonl
data/stageA_evidence_chunk_batch_input.jsonl
data/stageA_evidence_chunk_batch_output.jsonl
data/stageA_evidence_chunk_results.jsonl
data/stageA_evidence_chunk_parse_failures.jsonl
data/stageA_evidence_chunk_manifest.json
data/stageA_evidence_chunk_state.json

Outputs — unit reducer
----------------------
data/stageA_unit_reducer_batch_input.jsonl
data/stageA_unit_reducer_batch_output.jsonl
data/stageA_unit_evidence_synthesis.jsonl
data/stageA_unit_reducer_parse_failures.jsonl
data/stageA_evidence_synthesis_manifest.json
data/stageA_unit_reducer_state.json

Outputs — audit views
---------------------
data/stageA_main_synthesis_papers.jsonl
data/stageA_context_only_papers.jsonl
data/stageA_appendix_papers.jsonl
data/stageA_rejected_papers.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MASTER = DATA / "guideline_integration_master_v2.jsonl"

CHUNK_PLAN = DATA / "stageA_evidence_chunk_plan.jsonl"
CHUNK_INPUT = DATA / "stageA_evidence_chunk_batch_input.jsonl"
CHUNK_OUTPUT = DATA / "stageA_evidence_chunk_batch_output.jsonl"
CHUNK_ERRORS = DATA / "stageA_evidence_chunk_batch_errors.jsonl"
CHUNK_RESULTS = DATA / "stageA_evidence_chunk_results.jsonl"
CHUNK_FAILURES = DATA / "stageA_evidence_chunk_parse_failures.jsonl"
CHUNK_MANIFEST = DATA / "stageA_evidence_chunk_manifest.json"
CHUNK_STATE = DATA / "stageA_evidence_chunk_state.json"

REDUCER_INPUT = DATA / "stageA_unit_reducer_batch_input.jsonl"
REDUCER_OUTPUT = DATA / "stageA_unit_reducer_batch_output.jsonl"
REDUCER_ERRORS = DATA / "stageA_unit_reducer_batch_errors.jsonl"
REDUCER_RESULTS = DATA / "stageA_unit_evidence_synthesis.jsonl"
REDUCER_FAILURES = DATA / "stageA_unit_reducer_parse_failures.jsonl"
FINAL_MANIFEST = DATA / "stageA_evidence_synthesis_manifest.json"
REDUCER_STATE = DATA / "stageA_unit_reducer_state.json"

MAIN_PAPERS = DATA / "stageA_main_synthesis_papers.jsonl"
CONTEXT_PAPERS = DATA / "stageA_context_only_papers.jsonl"
APPENDIX_PAPERS = DATA / "stageA_appendix_papers.jsonl"
REJECTED_PAPERS = DATA / "stageA_rejected_papers.jsonl"

OPENAI_BASE_URL = "https://api.openai.com/v1"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
TRANSIENT = {408, 409, 429, 500, 502, 503, 504}

CHAPTER_ORDER = ["1", "2", "3", "4.1", "4.2", "4.3", "5", "6"]


CHUNK_SYSTEM_PROMPT = r"""
You are conducting final evidence appraisal for an evidence unit in a
pancreatic-cancer living-guideline update (ESMO 2015 -> evidence through
August 2023).

This is an ABSTRACT-LEVEL evidence synthesis. Do not invent methods, endpoints,
effect estimates, populations, subgroup findings, or certainty judgments that
are not supported by the supplied title/abstract/metadata.

MEDICAL UNIT BOUNDARY
The evidence-unit definition is fixed. Technical chunking is only for context
management. Do not create, split, merge, rename, or reassign the medical unit.

USER-DEFINED EVIDENCE HIERARCHY — FOLLOW EXACTLY
Tier 1:
  Meta-analysis of HUMAN randomized controlled trials.
Tier 2:
  Meta-analysis of HUMAN retrospective/non-randomized studies; use this tier
  also for mixed meta-analyses that are not demonstrably restricted to RCTs.
Tier 3:
  Systematic review.
Tier 4:
  Other review OR standalone randomized controlled trial.

Do not silently replace this hierarchy with another conventional hierarchy.

Interpretation within the hierarchy:
- A standalone RCT may still directly support clinical synthesis when it has a
  relevant human population and clinically meaningful endpoint.
- An OTHER_REVIEW may inform background, interpretation, implementation,
  supportive care, rare questions, or emerging evidence, but should NOT by
  itself justify ADD/MODIFY/REMOVE of a clinical recommendation.
- Publication-type metadata are hints, not permission to invent study design.
- If a meta-analysis does not clearly establish that included studies were RCTs,
  do not call it Tier 1.

FINAL PAPER STATUS
Classify EVERY supplied paper into exactly one:
1. MAIN_SYNTHESIS
   Clinically usable evidence for the evidence-unit synthesis.
2. CONTEXT_ONLY
   Relevant clinical/contextual evidence that can inform interpretation but
   should not independently drive a recommendation change.
3. APPENDIX
   Relevant emerging, surrogate, mechanistic, weakly translatable, or otherwise
   non-recommendation-driving evidence worth retaining and discussing.
4. REJECT
   Wrong population/topic/design, non-human/preclinical when clinical evidence
   is required, abstract insufficient for any reliable contribution, or
   otherwise unusable.

THERAPEUTIC / INTERVENTIONAL ENDPOINT RULE
For treatment evidence, recommendation-driving evidence should address
clinically meaningful patient-relevant outcomes. Examples can include survival,
recurrence/disease-control outcomes when clinically meaningful, morbidity,
major complications, treatment toxicity, quality of life, symptom relief,
hospitalization, or other direct patient-level outcomes appropriate to the
question. Do not automatically treat a biomarker, pathway signal, radiographic
response, technical feasibility metric, pharmacodynamic measure, immune marker,
or other surrogate as sufficient for clinical translation.

If a therapeutic paper is only surrogate/mechanistic and does not establish a
clinically meaningful benefit, use CONTEXT_ONLY or APPENDIX rather than allowing
it to drive recommendation change.

DOMAIN-SPECIFIC EXCEPTIONS
Do not impose a therapeutic RCT/mortality standard on questions where it is
methodologically inappropriate.

For epidemiology/risk/screening:
  clinically relevant incidence, mortality, risk, screening performance,
  population outcomes, and validated risk associations can be appropriate.

For diagnosis/pathology:
  diagnostic accuracy, clinically validated discrimination, management impact,
  pathology classification, and clinically validated diagnostic biomarkers can
  be appropriate. Pure model-system/pathway biology without clinical validation
  is generally context/appendix.

For staging/prognosis/resectability:
  clinically validated staging accuracy, resectability classification,
  recurrence, survival, prognostic discrimination, or management impact can be
  appropriate. Unvalidated biomarker associations are not recommendation-driving.

For personalised medicine:
  molecular association alone is not enough for a treatment recommendation.
  Require clinical validation/translatability for recommendation-driving use.

For follow-up/survivorship:
  recurrence detection, patient outcomes, quality of life, symptoms,
  survivorship, long-term morbidity and clinically meaningful surveillance
  outcomes can be appropriate.

ABSTRACT-LEVEL LIMITATION
If the abstract does not provide enough information to determine a claim, say
UNCLEAR. Do not infer full-text details.

OUTPUT REQUIREMENTS
- Return one paper_decision for EVERY PMID supplied in the request.
- Do not add PMIDs that were not supplied.
- Keep per-paper notes concise.
- The chunk synthesis must cite PMIDs explicitly.
- Separate recommendation-driving findings from contextual/emerging evidence.
- Do not rewrite the guideline or make a final guideline recommendation here.
"""


REDUCER_SYSTEM_PROMPT = r"""
You are the final evidence-unit reducer for a pancreatic-cancer living-guideline
update. You receive already appraised technical chunk results belonging to ONE
frozen medical evidence unit.

The medical evidence unit must remain intact regardless of the number of papers.
Do not create new subunits.

USER-DEFINED EVIDENCE HIERARCHY
Tier 1: Meta-analysis of HUMAN randomized controlled trials.
Tier 2: Meta-analysis of HUMAN retrospective/non-randomized studies, including
        mixed meta-analyses not demonstrably restricted to RCTs.
Tier 3: Systematic review.
Tier 4: Other review OR standalone randomized controlled trial.

Do not replace this hierarchy with a different hierarchy.

Important:
- Standalone RCTs may directly support clinical synthesis when clinically
  relevant, despite being Tier 4 in this user-defined hierarchy.
- OTHER_REVIEW evidence can contextualize/support the synthesis but must not,
  by itself, justify ADD/MODIFY/REMOVE of a clinical recommendation.
- Give preference to clinically meaningful human evidence and directness.
- Surrogate-only or weakly translatable findings must not be upgraded merely
  because they are numerous.
- A large literature does not imply a clinically actionable conclusion.
- A small number of high-value clinically relevant studies can matter more than
  a large volume of weak contextual literature.
- Preserve disagreements and uncertainty.
- Do not invent effect sizes or claims absent from the chunk evidence.
- This is still Stage A: synthesize the evidence, but DO NOT rewrite the
  guideline text and DO NOT issue the final guideline recommendation.

You must preserve the paper classifications from the chunk appraisal. Your task
is to reduce them into one coherent evidence memo and one recommendation-
readiness signal for the later guideline-update stage.
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
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path.name} line {n}: {e}") from e
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chapter_sort_key(cid: str) -> int:
    try:
        return CHAPTER_ORDER.index(cid)
    except ValueError:
        return 999


def metadata_hint(p: dict[str, Any]) -> str:
    text = (
        clean(p.get("evidence_labels")) + " | " +
        clean(p.get("publication_types"))
    ).lower()
    if "meta-analysis" in text or "meta analysis" in text:
        return "META_ANALYSIS"
    if "systematic review" in text:
        return "SYSTEMATIC_REVIEW"
    if (
        "randomized controlled trial" in text
        or "randomised controlled trial" in text
        or "rct" in clean(p.get("evidence_labels")).lower()
    ):
        return "RCT"
    if "review" in text:
        return "OTHER_REVIEW"
    return "OTHER_OR_UNCLEAR"


def paper_sort_key(p: dict[str, Any]):
    order = {
        "META_ANALYSIS": 0,
        "SYSTEMATIC_REVIEW": 1,
        "RCT": 2,
        "OTHER_REVIEW": 3,
        "OTHER_OR_UNCLEAR": 4,
    }
    year = clean(p.get("publication_year"))
    pmid = clean(p.get("pmid"))
    return (
        order[metadata_hint(p)],
        year,
        int(pmid) if pmid.isdigit() else pmid,
    )


def paper_prompt_record(p: dict[str, Any]) -> str:
    return "\n".join([
        f"PMID: {clean(p.get('pmid'))}",
        f"Metadata class hint: {metadata_hint(p)}",
        f"Title: {clean(p.get('title'))}",
        f"Publication types: {clean(p.get('publication_types')) or '[not available]'}",
        f"Evidence labels: {clean(p.get('evidence_labels')) or '[not available]'}",
        f"Year: {clean(p.get('publication_year')) or '[not available]'}",
        f"MeSH: {clean(p.get('mesh_terms')) or '[not available]'}",
        f"Keywords: {clean(p.get('keywords')) or '[not available]'}",
        f"Abstract: {clean(p.get('abstract')) or '[no abstract available]'}",
    ])


def estimated_chars(p: dict[str, Any]) -> int:
    return len(paper_prompt_record(p)) + 250


def make_chunks(
    papers: list[dict[str, Any]],
    max_papers: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    papers = sorted(papers, key=paper_sort_key)
    chunks = []
    current = []
    chars = 0

    for p in papers:
        pchars = estimated_chars(p)
        if current and (
            len(current) >= max_papers
            or chars + pchars > max_chars
        ):
            chunks.append(current)
            current = []
            chars = 0

        current.append(p)
        chars += pchars

    if current:
        chunks.append(current)

    return chunks


def chunk_schema(pmids: list[str]) -> dict[str, Any]:
    return {
        "name": "pdac_stageA_chunk_appraisal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "paper_decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pmid": {
                                "type": "string",
                                "enum": pmids,
                            },
                            "evidence_tier": {
                                "type": "string",
                                "enum": [
                                    "TIER_1_MA_OF_RCTS",
                                    "TIER_2_MA_NONRANDOMIZED_OR_MIXED",
                                    "TIER_3_SYSTEMATIC_REVIEW",
                                    "TIER_4_RCT_OR_OTHER_REVIEW",
                                    "NOT_ELIGIBLE_DESIGN",
                                    "UNCLEAR",
                                ],
                            },
                            "paper_status": {
                                "type": "string",
                                "enum": [
                                    "MAIN_SYNTHESIS",
                                    "CONTEXT_ONLY",
                                    "APPENDIX",
                                    "REJECT",
                                ],
                            },
                            "human_clinical_relevance": {
                                "type": "string",
                                "enum": ["YES", "NO", "UNCLEAR"],
                            },
                            "endpoint_relevance": {
                                "type": "string",
                                "enum": [
                                    "HARD_OR_PATIENT_RELEVANT_CLINICAL",
                                    "DOMAIN_APPROPRIATE_CLINICAL",
                                    "MIXED_CLINICAL_AND_SURROGATE",
                                    "SURROGATE_ONLY",
                                    "MECHANISTIC_OR_PRECLINICAL",
                                    "UNCLEAR",
                                ],
                            },
                            "clinical_translatability": {
                                "type": "string",
                                "enum": [
                                    "HIGH",
                                    "MODERATE",
                                    "LOW",
                                    "NONE",
                                    "UNCLEAR",
                                ],
                            },
                            "can_drive_recommendation_change": {
                                "type": "boolean",
                            },
                            "reason_code": {
                                "type": "string",
                                "enum": [
                                    "ELIGIBLE_CLINICAL_EVIDENCE",
                                    "DOMAIN_APPROPRIATE_NONTHERAPEUTIC_EVIDENCE",
                                    "CONTEXTUAL_REVIEW",
                                    "SURROGATE_ONLY",
                                    "INSUFFICIENT_CLINICAL_TRANSLATION",
                                    "WRONG_POPULATION",
                                    "WRONG_TOPIC",
                                    "WRONG_STUDY_DESIGN",
                                    "NON_HUMAN_OR_PRECLINICAL",
                                    "DUPLICATIVE_OR_LOW_VALUE_REVIEW",
                                    "ABSTRACT_INSUFFICIENT",
                                    "OTHER",
                                ],
                            },
                            "concise_finding": {"type": "string"},
                            "concise_reason": {"type": "string"},
                        },
                        "required": [
                            "pmid",
                            "evidence_tier",
                            "paper_status",
                            "human_clinical_relevance",
                            "endpoint_relevance",
                            "clinical_translatability",
                            "can_drive_recommendation_change",
                            "reason_code",
                            "concise_finding",
                            "concise_reason",
                        ],
                        "additionalProperties": False,
                    },
                },
                "chunk_key_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "supporting_pmids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": pmids,
                                },
                            },
                            "highest_supporting_tier": {
                                "type": "string",
                                "enum": [
                                    "TIER_1_MA_OF_RCTS",
                                    "TIER_2_MA_NONRANDOMIZED_OR_MIXED",
                                    "TIER_3_SYSTEMATIC_REVIEW",
                                    "TIER_4_RCT_OR_OTHER_REVIEW",
                                    "UNCLEAR",
                                ],
                            },
                            "clinical_relevance": {
                                "type": "string",
                                "enum": ["HIGH", "MODERATE", "LOW", "UNCLEAR"],
                            },
                            "consistency": {
                                "type": "string",
                                "enum": [
                                    "CONSISTENT",
                                    "MOSTLY_CONSISTENT",
                                    "MIXED",
                                    "CONFLICTING",
                                    "INSUFFICIENT",
                                ],
                            },
                        },
                        "required": [
                            "finding",
                            "supporting_pmids",
                            "highest_supporting_tier",
                            "clinical_relevance",
                            "consistency",
                        ],
                        "additionalProperties": False,
                    },
                },
                "chunk_conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string"},
                            "pmids": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": pmids,
                                },
                            },
                        },
                        "required": ["issue", "pmids"],
                        "additionalProperties": False,
                    },
                },
                "chunk_main_synthesis_summary": {"type": "string"},
                "chunk_context_summary": {"type": "string"},
                "chunk_appendix_summary": {"type": "string"},
                "chunk_limitations": {"type": "string"},
            },
            "required": [
                "paper_decisions",
                "chunk_key_findings",
                "chunk_conflicts",
                "chunk_main_synthesis_summary",
                "chunk_context_summary",
                "chunk_appendix_summary",
                "chunk_limitations",
            ],
            "additionalProperties": False,
        },
    }


def reducer_schema() -> dict[str, Any]:
    return {
        "name": "pdac_stageA_unit_evidence_synthesis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "evidence_unit_id": {"type": "string"},
                "main_synthesis_pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "context_only_pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "appendix_pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rejected_pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "tier_counts_main_synthesis": {
                    "type": "object",
                    "properties": {
                        "tier_1_ma_of_rcts": {"type": "integer"},
                        "tier_2_ma_nonrandomized_or_mixed": {"type": "integer"},
                        "tier_3_systematic_review": {"type": "integer"},
                        "tier_4_rct_or_other_review": {"type": "integer"},
                        "unclear_or_other": {"type": "integer"},
                    },
                    "required": [
                        "tier_1_ma_of_rcts",
                        "tier_2_ma_nonrandomized_or_mixed",
                        "tier_3_systematic_review",
                        "tier_4_rct_or_other_review",
                        "unclear_or_other",
                    ],
                    "additionalProperties": False,
                },
                "key_clinical_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "supporting_pmids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "highest_supporting_tier": {
                                "type": "string",
                                "enum": [
                                    "TIER_1_MA_OF_RCTS",
                                    "TIER_2_MA_NONRANDOMIZED_OR_MIXED",
                                    "TIER_3_SYSTEMATIC_REVIEW",
                                    "TIER_4_RCT_OR_OTHER_REVIEW",
                                    "UNCLEAR",
                                ],
                            },
                            "clinical_relevance": {
                                "type": "string",
                                "enum": ["HIGH", "MODERATE", "LOW", "UNCLEAR"],
                            },
                            "consistency": {
                                "type": "string",
                                "enum": [
                                    "CONSISTENT",
                                    "MOSTLY_CONSISTENT",
                                    "MIXED",
                                    "CONFLICTING",
                                    "INSUFFICIENT",
                                ],
                            },
                            "interpretation": {"type": "string"},
                        },
                        "required": [
                            "finding",
                            "supporting_pmids",
                            "highest_supporting_tier",
                            "clinical_relevance",
                            "consistency",
                            "interpretation",
                        ],
                        "additionalProperties": False,
                    },
                },
                "evidence_conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string"},
                            "pmids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "interpretation": {"type": "string"},
                        },
                        "required": ["issue", "pmids", "interpretation"],
                        "additionalProperties": False,
                    },
                },
                "recommendation_readiness": {
                    "type": "string",
                    "enum": [
                        "CLINICALLY_ACTIONABLE_EVIDENCE_PRESENT",
                        "CLINICALLY_RELEVANT_BUT_NOT_ACTIONABLE",
                        "CONTEXT_OR_EMERGING_EVIDENCE_ONLY",
                        "INSUFFICIENT_CLINICAL_EVIDENCE",
                    ],
                },
                "potential_guideline_implication": {
                    "type": "string",
                    "enum": [
                        "POTENTIAL_CONFIRMATION",
                        "POTENTIAL_MODIFICATION",
                        "POTENTIAL_ADDITION",
                        "POTENTIAL_REMOVAL",
                        "NO_CLINICAL_CHANGE_SIGNAL",
                        "CANNOT_DETERMINE_WITHOUT_OLD_GUIDELINE_TEXT",
                    ],
                },
                "recommendation_driving_evidence_summary": {"type": "string"},
                "context_evidence_summary": {"type": "string"},
                "appendix_evidence_summary": {"type": "string"},
                "clinical_translation_summary": {"type": "string"},
                "limitations": {"type": "string"},
                "evidence_memo": {"type": "string"},
            },
            "required": [
                "evidence_unit_id",
                "main_synthesis_pmids",
                "context_only_pmids",
                "appendix_pmids",
                "rejected_pmids",
                "tier_counts_main_synthesis",
                "key_clinical_findings",
                "evidence_conflicts",
                "recommendation_readiness",
                "potential_guideline_implication",
                "recommendation_driving_evidence_summary",
                "context_evidence_summary",
                "appendix_evidence_summary",
                "clinical_translation_summary",
                "limitations",
                "evidence_memo",
            ],
            "additionalProperties": False,
        },
    }


def sanitize_unit_id(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", uid).strip("_")


def parse_chunk_custom_id(cid: str) -> tuple[str, int]:
    m = re.fullmatch(r"stageA-chunk-(.+)-(\d{3})", cid)
    if not m:
        raise ValueError(f"Invalid chunk custom_id: {cid}")
    return m.group(1).replace("_DOT_", "."), int(m.group(2))


def custom_unit_token(uid: str) -> str:
    return uid.replace(".", "_DOT_")


def prepare_chunks(
    model: str,
    effort: str,
    max_papers: int,
    max_chars: int,
) -> dict[str, Any]:
    units = load_jsonl(MASTER)

    plan_rows = []
    request_rows = []
    total_papers = 0
    chapter_counts = Counter()
    max_unit_papers = 0

    for unit in units:
        uid = clean(unit.get("evidence_unit_id"))
        cid = clean(unit.get("chapter_id"))
        papers = unit.get("mapped_evidence") or []

        if not uid or not cid:
            raise RuntimeError("Integration master contains unit without ID/chapter.")
        if not papers:
            raise RuntimeError(f"Frozen unit {uid} has zero mapped evidence.")

        pmids = [clean(p.get("pmid")) for p in papers]
        if not all(pmids):
            raise RuntimeError(f"Unit {uid} has mapped paper without PMID.")
        if len(pmids) != len(set(pmids)):
            raise RuntimeError(
                f"Unit {uid} contains duplicate PMID records before chunking."
            )

        chunks = make_chunks(papers, max_papers, max_chars)
        total_papers += len(papers)
        max_unit_papers = max(max_unit_papers, len(papers))

        for chunk_index, chunk in enumerate(chunks, 1):
            chunk_pmids = [clean(p.get("pmid")) for p in chunk]
            custom_id = (
                f"stageA-chunk-{custom_unit_token(uid)}-{chunk_index:03d}"
            )

            domain_policy = unit.get("final_evidence_policy") or {}

            user_prompt = "\n".join([
                f"CHAPTER: {cid} — {clean(unit.get('chapter_title'))}",
                f"EVIDENCE UNIT ID: {uid}",
                f"EVIDENCE UNIT: {clean(unit.get('evidence_unit_name'))}",
                f"DEFINITION: {clean(unit.get('evidence_unit_definition'))}",
                f"BOUNDARY: {clean(unit.get('evidence_unit_boundary'))}",
                f"UNIT ORIGIN: {clean(unit.get('evidence_unit_origin'))}",
                "",
                "DOMAIN POLICY FROM THE FROZEN MASTER:",
                json.dumps(domain_policy, ensure_ascii=False, indent=2),
                "",
                f"TECHNICAL CHUNK: {chunk_index}/{len(chunks)}",
                f"PAPERS IN THIS CHUNK: {len(chunk)}",
                "",
                "PAPERS:",
                "\n\n--- PAPER ---\n\n".join(
                    paper_prompt_record(p) for p in chunk
                ),
                "",
                "Appraise every paper and synthesize this technical chunk. "
                "Do not rewrite the guideline.",
            ])

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": chunk_schema(chunk_pmids),
                },
                "max_completion_tokens": 50000,
                "reasoning_effort": effort,
            }

            request_rows.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

            plan_rows.append({
                "custom_id": custom_id,
                "chapter_id": cid,
                "evidence_unit_id": uid,
                "evidence_unit_name": clean(unit.get("evidence_unit_name")),
                "chunk_index": chunk_index,
                "chunk_count_for_unit": len(chunks),
                "paper_count": len(chunk),
                "pmids": chunk_pmids,
                "estimated_prompt_chars_from_papers": sum(
                    estimated_chars(p) for p in chunk
                ),
            })
            chapter_counts[cid] += 1

    if len(request_rows) != len({r["custom_id"] for r in request_rows}):
        raise RuntimeError("Duplicate chunk custom IDs.")

    write_jsonl(CHUNK_PLAN, plan_rows)
    write_jsonl(CHUNK_INPUT, request_rows)

    size_mb = CHUNK_INPUT.stat().st_size / 1024 / 1024
    if len(request_rows) > 50_000:
        raise RuntimeError("Chunk batch exceeds 50,000 requests.")
    if size_mb > 190:
        raise RuntimeError(
            f"Chunk Batch JSONL is {size_mb:.2f} MB. "
            "Reduce chunk prompt duplication or split Batch files."
        )

    manifest = {
        "status": "PREPARED",
        "frozen_units": len(units),
        "mapped_paper_unit_assignments": total_papers,
        "technical_chunks": len(plan_rows),
        "max_papers_in_any_medical_unit": max_unit_papers,
        "max_papers_per_technical_chunk": max_papers,
        "max_chars_per_technical_chunk": max_chars,
        "batch_input_mb": round(size_mb, 2),
        "model": model,
        "reasoning_effort": effort,
        "chapter_chunk_counts": dict(chapter_counts),
        "medical_unit_split_policy": (
            "Technical chunking only. No medical evidence-unit boundary changes."
        ),
    }
    CHUNK_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL STAGE A — chunk Batch prepared.")
    print(f"  frozen medical evidence units:        {len(units):,}")
    print(f"  PMID-unit assignments to appraise:    {total_papers:,}")
    print(f"  technical chunks / Batch requests:    {len(plan_rows):,}")
    print(f"  largest medical unit:                 {max_unit_papers:,} papers")
    print(f"  max papers per technical chunk:       {max_papers:,}")
    print(f"  max chars per technical chunk:        {max_chars:,}")
    print(f"  chunk Batch JSONL:                    {size_mb:.2f} MB")
    print(f"  model:                                {model}")
    print(f"  reasoning effort:                     {effort}")
    print()
    for cid in CHAPTER_ORDER:
        print(f"  chapter {cid:>3}: {chapter_counts[cid]:,} technical chunks")
    print()
    print(f"  plan:        {CHUNK_PLAN}")
    print(f"  batch input: {CHUNK_INPUT}")
    return manifest


class Client:
    def __init__(self, key: str, retry_wait: int):
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.s = requests.Session()
        self.headers = {"Authorization": f"Bearer {key}"}
        self.retry_wait = retry_wait

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: int = 900,
        **kwargs: Any,
    ) -> requests.Response:
        url = path if path.startswith("http") else OPENAI_BASE_URL + path

        while True:
            headers = dict(self.headers)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.s.request(
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

            if r.status_code in TRANSIENT:
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


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def submit_batch(
    client: Client,
    input_path: Path,
    state_path: Path,
    task_name: str,
    model: str,
    effort: str,
) -> dict[str, Any]:
    state = load_state(state_path)
    if state.get("batch_id"):
        print(f"Existing {task_name} Batch: {state['batch_id']}")
        print("Resuming; no duplicate Batch submitted.")
        return state

    with input_path.open("rb") as f:
        upload = client.request(
            "POST",
            "/files",
            files={
                "file": (input_path.name, f, "application/jsonl")
            },
            data={"purpose": "batch"},
        ).json()

    payload = {
        "input_file_id": upload["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "project": "ESMO_PDAC_2015_to_2023_PoC",
            "task": task_name,
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
    save_state(state_path, state)

    print(f"Batch id: {batch['id']}")
    print(f"Status:   {batch.get('status')}")
    return state


def download(client: Client, file_id: str, destination: Path) -> None:
    destination.write_bytes(
        client.request(
            "GET",
            f"/files/{file_id}/content",
            timeout=1200,
        ).content
    )


def watch_batch(
    client: Client,
    state_path: Path,
    output_path: Path,
    error_path: Path,
    poll_seconds: int,
) -> dict[str, Any]:
    state = load_state(state_path)
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError(f"No batch_id in {state_path.name}")

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

        state.update({
            "status": status,
            "output_file_id": batch.get("output_file_id"),
            "error_file_id": batch.get("error_file_id"),
            "request_counts": counts,
        })
        save_state(state_path, state)

        if status in TERMINAL:
            if batch.get("output_file_id"):
                download(
                    client,
                    batch["output_file_id"],
                    output_path,
                )
            if batch.get("error_file_id"):
                download(
                    client,
                    batch["error_file_id"],
                    error_path,
                )
            return batch

        time.sleep(poll_seconds)


def extract_batch_content(obj: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if obj.get("error"):
        raise ValueError(f"Batch error: {obj['error']}")

    response = obj.get("response")
    if not response:
        raise ValueError("Missing response object.")
    if response.get("status_code") != 200:
        raise ValueError(f"HTTP {response.get('status_code')}")

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("No choices in response.")

    choice = choices[0]
    finish_reason = clean(choice.get("finish_reason"))
    msg = choice.get("message") or {}
    content = msg.get("content") or ""

    if finish_reason != "stop":
        raise ValueError(
            f"finish_reason={finish_reason}; content_length={len(content)}"
        )
    if not content.strip():
        raise ValueError("Empty response content.")

    return content, finish_reason, body.get("usage") or {}


def merge_chunks() -> dict[str, Any]:
    plan = load_jsonl(CHUNK_PLAN)
    plan_by_id = {r["custom_id"]: r for r in plan}

    results = []
    failures = []
    seen_custom_ids = set()

    with CHUNK_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = clean(obj.get("custom_id"))

            try:
                if cid not in plan_by_id:
                    raise ValueError(f"Unknown custom_id {cid}")
                if cid in seen_custom_ids:
                    raise ValueError(f"Duplicate output custom_id {cid}")
                seen_custom_ids.add(cid)

                content, finish_reason, usage = extract_batch_content(obj)
                parsed = json.loads(content)

                expected_pmids = plan_by_id[cid]["pmids"]
                decisions = parsed.get("paper_decisions") or []
                returned_pmids = [clean(x.get("pmid")) for x in decisions]

                if len(returned_pmids) != len(set(returned_pmids)):
                    raise ValueError("Duplicate PMID in paper_decisions.")
                if set(returned_pmids) != set(expected_pmids):
                    missing = sorted(set(expected_pmids) - set(returned_pmids))
                    extra = sorted(set(returned_pmids) - set(expected_pmids))
                    raise ValueError(
                        f"PMID completeness mismatch; missing={missing}; extra={extra}"
                    )

                # Deterministic guard: OTHER_REVIEW alone must not be marked
                # recommendation-driving. We can identify metadata hints from master
                # later, but here preserve model output and enforce in audit merge.
                result = {
                    **plan_by_id[cid],
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "result": parsed,
                }
                results.append(result)

            except Exception as e:
                failures.append({
                    "line": line_no,
                    "custom_id": cid,
                    "error": f"{type(e).__name__}: {e}",
                })

    missing_custom_ids = sorted(set(plan_by_id) - seen_custom_ids)
    for cid in missing_custom_ids:
        failures.append({
            "line": None,
            "custom_id": cid,
            "error": "Missing output row",
        })

    results.sort(
        key=lambda r: (
            chapter_sort_key(r["chapter_id"]),
            r["evidence_unit_id"],
            r["chunk_index"],
        )
    )
    write_jsonl(CHUNK_RESULTS, results)
    write_jsonl(CHUNK_FAILURES, failures)

    status_counts = Counter()
    tier_counts = Counter()
    endpoint_counts = Counter()

    for r in results:
        for d in r["result"]["paper_decisions"]:
            status_counts[d["paper_status"]] += 1
            tier_counts[d["evidence_tier"]] += 1
            endpoint_counts[d["endpoint_relevance"]] += 1

    manifest = load_json(CHUNK_MANIFEST)
    manifest.update({
        "status": "COMPLETE" if not failures and len(results) == len(plan) else "INCOMPLETE",
        "successful_chunks": len(results),
        "failed_or_missing_chunks": len(failures),
        "paper_status_counts_across_chunk_decisions": dict(status_counts),
        "evidence_tier_counts_across_chunk_decisions": dict(tier_counts),
        "endpoint_relevance_counts": dict(endpoint_counts),
    })
    CHUNK_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSTAGE A chunk merge completed.")
    print(f"  planned chunks:            {len(plan):,}")
    print(f"  successful chunks:         {len(results):,}")
    print(f"  failed/missing chunks:     {len(failures):,}")
    print()
    print("Paper statuses:")
    for s in ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"]:
        print(f"  {s:<18} {status_counts[s]:,}")
    print()
    print(f"  results:   {CHUNK_RESULTS}")
    print(f"  failures:  {CHUNK_FAILURES}")
    print(f"  manifest:  {CHUNK_MANIFEST}")

    return manifest


def build_paper_metadata_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    for unit in load_jsonl(MASTER):
        uid = clean(unit["evidence_unit_id"])
        for p in unit.get("mapped_evidence") or []:
            pmid = clean(p.get("pmid"))
            lookup[(uid, pmid)] = p
    return lookup


def prepare_reducers(
    model: str,
    effort: str,
) -> dict[str, Any]:
    chunk_manifest = load_json(CHUNK_MANIFEST)
    if chunk_manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            "Chunk stage is not COMPLETE. Repair chunk failures before reducers."
        )

    units = load_jsonl(MASTER)
    chunks = load_jsonl(CHUNK_RESULTS)
    chunks_by_unit = defaultdict(list)

    for c in chunks:
        chunks_by_unit[c["evidence_unit_id"]].append(c)

    requests_out = []

    for unit in units:
        uid = clean(unit["evidence_unit_id"])
        unit_chunks = sorted(
            chunks_by_unit.get(uid, []),
            key=lambda x: x["chunk_index"],
        )
        if not unit_chunks:
            raise RuntimeError(f"No chunk results for unit {uid}")

        # Ensure all original unit PMIDs are represented once in chunk decisions.
        expected_pmids = {
            clean(p.get("pmid"))
            for p in unit.get("mapped_evidence") or []
        }
        decided_pmids = []
        for c in unit_chunks:
            decided_pmids.extend(
                clean(d["pmid"])
                for d in c["result"]["paper_decisions"]
            )
        if set(decided_pmids) != expected_pmids:
            raise RuntimeError(
                f"Reducer prep PMID mismatch for unit {uid}: "
                f"expected {len(expected_pmids)}, found {len(set(decided_pmids))}"
            )
        if len(decided_pmids) != len(set(decided_pmids)):
            raise RuntimeError(
                f"Reducer prep duplicate PMID decisions for unit {uid}"
            )

        compact_chunks = []
        for c in unit_chunks:
            compact_chunks.append({
                "chunk_index": c["chunk_index"],
                "paper_decisions": c["result"]["paper_decisions"],
                "chunk_key_findings": c["result"]["chunk_key_findings"],
                "chunk_conflicts": c["result"]["chunk_conflicts"],
                "chunk_main_synthesis_summary": c["result"]["chunk_main_synthesis_summary"],
                "chunk_context_summary": c["result"]["chunk_context_summary"],
                "chunk_appendix_summary": c["result"]["chunk_appendix_summary"],
                "chunk_limitations": c["result"]["chunk_limitations"],
            })

        user_prompt = "\n".join([
            f"CHAPTER: {clean(unit.get('chapter_id'))} — {clean(unit.get('chapter_title'))}",
            f"EVIDENCE UNIT ID: {uid}",
            f"EVIDENCE UNIT: {clean(unit.get('evidence_unit_name'))}",
            f"DEFINITION: {clean(unit.get('evidence_unit_definition'))}",
            f"BOUNDARY: {clean(unit.get('evidence_unit_boundary'))}",
            f"UNIT ORIGIN: {clean(unit.get('evidence_unit_origin'))}",
            "",
            "DOMAIN POLICY:",
            json.dumps(
                unit.get("final_evidence_policy") or {},
                ensure_ascii=False,
                indent=2,
            ),
            "",
            f"TOTAL UNIQUE PAPERS IN UNIT: {len(expected_pmids)}",
            f"TECHNICAL CHUNKS: {len(unit_chunks)}",
            "",
            "APPRAISED CHUNK RESULTS:",
            json.dumps(compact_chunks, ensure_ascii=False, indent=2),
            "",
            "Reduce these chunk appraisals into ONE evidence-unit synthesis. "
            "Preserve paper classifications and do not rewrite the guideline.",
        ])

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": REDUCER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": reducer_schema(),
            },
            "max_completion_tokens": 45000,
            "reasoning_effort": effort,
        }

        requests_out.append({
            "custom_id": f"stageA-reduce-{custom_unit_token(uid)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    if len(requests_out) != 184:
        raise RuntimeError(
            f"Expected 184 reducer requests, found {len(requests_out)}"
        )
    if len(requests_out) != len({r["custom_id"] for r in requests_out}):
        raise RuntimeError("Duplicate reducer custom IDs.")

    write_jsonl(REDUCER_INPUT, requests_out)
    size_mb = REDUCER_INPUT.stat().st_size / 1024 / 1024
    if size_mb > 190:
        raise RuntimeError(
            f"Reducer Batch JSONL is {size_mb:.2f} MB; split reducers if needed."
        )

    summary = {
        "status": "PREPARED",
        "reducer_requests": len(requests_out),
        "batch_input_mb": round(size_mb, 2),
        "model": model,
        "reasoning_effort": effort,
    }

    print("\nSTAGE A unit reducers prepared.")
    print(f"  frozen units / reducer requests: {len(requests_out):,}")
    print(f"  reducer Batch JSONL:             {size_mb:.2f} MB")
    print(f"  model:                           {model}")
    print(f"  reasoning effort:                {effort}")
    print(f"  input:                           {REDUCER_INPUT}")
    return summary


def parse_reducer_custom_id(cid: str) -> str:
    prefix = "stageA-reduce-"
    if not cid.startswith(prefix):
        raise ValueError(f"Invalid reducer custom_id: {cid}")
    return cid[len(prefix):].replace("_DOT_", ".")


def deterministic_paper_views(
    reducer_results: list[dict[str, Any]],
) -> None:
    meta_lookup = build_paper_metadata_lookup()

    main_rows = []
    context_rows = []
    appendix_rows = []
    reject_rows = []

    # Build concise appraisal lookup from chunk results.
    appraisal_lookup = {}
    for c in load_jsonl(CHUNK_RESULTS):
        uid = c["evidence_unit_id"]
        for d in c["result"]["paper_decisions"]:
            appraisal_lookup[(uid, clean(d["pmid"]))] = d

    for unit_result in reducer_results:
        uid = unit_result["evidence_unit_id"]
        result = unit_result["result"]

        mapping = {
            "MAIN_SYNTHESIS": result["main_synthesis_pmids"],
            "CONTEXT_ONLY": result["context_only_pmids"],
            "APPENDIX": result["appendix_pmids"],
            "REJECT": result["rejected_pmids"],
        }

        for status, pmids in mapping.items():
            for pmid in pmids:
                p = deepcopy(meta_lookup.get((uid, pmid), {}))
                row = {
                    "evidence_unit_id": uid,
                    "paper_status": status,
                    **p,
                    "stageA_appraisal": appraisal_lookup.get((uid, pmid), {}),
                }
                if status == "MAIN_SYNTHESIS":
                    main_rows.append(row)
                elif status == "CONTEXT_ONLY":
                    context_rows.append(row)
                elif status == "APPENDIX":
                    appendix_rows.append(row)
                else:
                    reject_rows.append(row)

    write_jsonl(MAIN_PAPERS, main_rows)
    write_jsonl(CONTEXT_PAPERS, context_rows)
    write_jsonl(APPENDIX_PAPERS, appendix_rows)
    write_jsonl(REJECTED_PAPERS, reject_rows)


def merge_reducers(model: str) -> dict[str, Any]:
    units = load_jsonl(MASTER)
    unit_by_id = {
        clean(u["evidence_unit_id"]): u
        for u in units
    }

    chunk_results = load_jsonl(CHUNK_RESULTS)
    chunk_decisions_by_unit = defaultdict(dict)

    for c in chunk_results:
        uid = c["evidence_unit_id"]
        for d in c["result"]["paper_decisions"]:
            pmid = clean(d["pmid"])
            chunk_decisions_by_unit[uid][pmid] = d

    results = []
    failures = []
    seen = set()

    with REDUCER_OUTPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = clean(obj.get("custom_id"))

            try:
                uid = parse_reducer_custom_id(cid)
                if uid not in unit_by_id:
                    raise ValueError(f"Unknown evidence unit {uid}")
                if uid in seen:
                    raise ValueError(f"Duplicate reducer result for {uid}")
                seen.add(uid)

                content, finish_reason, usage = extract_batch_content(obj)
                parsed = json.loads(content)

                if clean(parsed.get("evidence_unit_id")) != uid:
                    raise ValueError(
                        f"Returned unit ID {parsed.get('evidence_unit_id')} != {uid}"
                    )

                expected = set(chunk_decisions_by_unit[uid])
                status_lists = {
                    "MAIN_SYNTHESIS": [
                        clean(x) for x in parsed["main_synthesis_pmids"]
                    ],
                    "CONTEXT_ONLY": [
                        clean(x) for x in parsed["context_only_pmids"]
                    ],
                    "APPENDIX": [
                        clean(x) for x in parsed["appendix_pmids"]
                    ],
                    "REJECT": [
                        clean(x) for x in parsed["rejected_pmids"]
                    ],
                }

                flattened = []
                for status, pmids in status_lists.items():
                    if len(pmids) != len(set(pmids)):
                        raise ValueError(
                            f"Duplicate PMIDs inside reducer status {status}"
                        )
                    flattened.extend(pmids)

                if len(flattened) != len(set(flattened)):
                    raise ValueError(
                        "A PMID appears in more than one reducer status list."
                    )
                if set(flattened) != expected:
                    missing = sorted(expected - set(flattened))
                    extra = sorted(set(flattened) - expected)
                    raise ValueError(
                        f"Reducer PMID partition mismatch; "
                        f"missing={missing}; extra={extra}"
                    )

                # Critical preservation rule:
                # reducer may not change the per-paper statuses assigned by chunks.
                for status, pmids in status_lists.items():
                    for pmid in pmids:
                        chunk_status = (
                            chunk_decisions_by_unit[uid][pmid]["paper_status"]
                        )
                        if chunk_status != status:
                            raise ValueError(
                                f"Reducer altered PMID {pmid} status "
                                f"{chunk_status} -> {status}"
                            )

                # Deterministically recompute main-synthesis tier counts and
                # overwrite only if model count is wrong.
                tier_counter = Counter(
                    chunk_decisions_by_unit[uid][pmid]["evidence_tier"]
                    for pmid in status_lists["MAIN_SYNTHESIS"]
                )
                deterministic_counts = {
                    "tier_1_ma_of_rcts":
                        tier_counter["TIER_1_MA_OF_RCTS"],
                    "tier_2_ma_nonrandomized_or_mixed":
                        tier_counter["TIER_2_MA_NONRANDOMIZED_OR_MIXED"],
                    "tier_3_systematic_review":
                        tier_counter["TIER_3_SYSTEMATIC_REVIEW"],
                    "tier_4_rct_or_other_review":
                        tier_counter["TIER_4_RCT_OR_OTHER_REVIEW"],
                    "unclear_or_other":
                        tier_counter["UNCLEAR"]
                        + tier_counter["NOT_ELIGIBLE_DESIGN"],
                }
                parsed["tier_counts_main_synthesis"] = deterministic_counts

                enriched = {
                    "chapter_id": clean(unit_by_id[uid]["chapter_id"]),
                    "chapter_title": clean(unit_by_id[uid]["chapter_title"]),
                    "evidence_unit_id": uid,
                    "evidence_unit_name": clean(
                        unit_by_id[uid]["evidence_unit_name"]
                    ),
                    "evidence_unit_definition": clean(
                        unit_by_id[uid]["evidence_unit_definition"]
                    ),
                    "evidence_unit_origin": clean(
                        unit_by_id[uid]["evidence_unit_origin"]
                    ),
                    "mapped_evidence_count": len(expected),
                    "model": model,
                    "reasoning_effort": load_state(
                        REDUCER_STATE
                    ).get("reasoning_effort", "high"),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "result": parsed,
                }
                results.append(enriched)

            except Exception as e:
                failures.append({
                    "line": line_no,
                    "custom_id": cid,
                    "error": f"{type(e).__name__}: {e}",
                })

    for uid in unit_by_id:
        if uid not in seen:
            failures.append({
                "line": None,
                "custom_id": f"stageA-reduce-{custom_unit_token(uid)}",
                "error": "Missing reducer output",
            })

    results.sort(
        key=lambda r: (
            chapter_sort_key(r["chapter_id"]),
            r["evidence_unit_id"],
        )
    )
    write_jsonl(REDUCER_RESULTS, results)
    write_jsonl(REDUCER_FAILURES, failures)

    deterministic_paper_views(results)

    status_counts = Counter()
    readiness_counts = Counter()
    implication_counts = Counter()
    tier_counts = Counter()

    for r in results:
        x = r["result"]
        status_counts["MAIN_SYNTHESIS"] += len(x["main_synthesis_pmids"])
        status_counts["CONTEXT_ONLY"] += len(x["context_only_pmids"])
        status_counts["APPENDIX"] += len(x["appendix_pmids"])
        status_counts["REJECT"] += len(x["rejected_pmids"])
        readiness_counts[x["recommendation_readiness"]] += 1
        implication_counts[x["potential_guideline_implication"]] += 1
        for k, v in x["tier_counts_main_synthesis"].items():
            tier_counts[k] += int(v)

    manifest = {
        "status": (
            "READY_FOR_STAGE_B_GUIDELINE_UPDATE"
            if not failures and len(results) == 184
            else "INCOMPLETE"
        ),
        "frozen_evidence_units": 184,
        "successful_unit_syntheses": len(results),
        "failed_or_missing_unit_syntheses": len(failures),
        "paper_status_counts": dict(status_counts),
        "main_synthesis_tier_counts": dict(tier_counts),
        "recommendation_readiness_counts": dict(readiness_counts),
        "potential_guideline_implication_counts": dict(implication_counts),
        "evidence_hierarchy": [
            "Tier 1: meta-analysis of human RCTs",
            "Tier 2: meta-analysis of human retrospective/non-randomized or mixed studies",
            "Tier 3: systematic review",
            "Tier 4: other review OR standalone RCT",
        ],
        "important_constraint": (
            "OTHER_REVIEW records were retained. They can support context but "
            "must not alone justify ADD/MODIFY/REMOVE."
        ),
        "next_step": (
            "Stage B: compare each Stage-A evidence-unit memo with the original "
            "ESMO-2015 text/context and generate CONFIRM/MODIFY/ADD/REMOVE/"
            "INSUFFICIENT decisions and updated English guideline text."
        ),
    }
    FINAL_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL STAGE A unit synthesis merge completed.")
    print(f"  successful unit syntheses:    {len(results):,}/184")
    print(f"  failures:                     {len(failures):,}")
    print()
    print("Paper disposition across PMID-unit assignments:")
    for s in ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"]:
        print(f"  {s:<18} {status_counts[s]:,}")
    print()
    print("Recommendation readiness:")
    for k, v in readiness_counts.items():
        print(f"  {k}: {v:,}")
    print()
    print(f"  unit syntheses: {REDUCER_RESULTS}")
    print(f"  main papers:    {MAIN_PAPERS}")
    print(f"  context papers: {CONTEXT_PAPERS}")
    print(f"  appendix:       {APPENDIX_PAPERS}")
    print(f"  rejected audit: {REJECTED_PAPERS}")
    print(f"  failures:       {REDUCER_FAILURES}")
    print(f"  manifest:       {FINAL_MANIFEST}")

    if failures:
        print(
            "\nWARNING: repair reducer failures before Stage B guideline update."
        )

    return manifest


def test_first_request(
    client: Client,
    input_path: Path,
) -> None:
    if not input_path.exists():
        raise RuntimeError(f"Missing input file: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        req = json.loads(f.readline())

    print(f"Testing synchronously: {req['custom_id']}")
    r = client.request(
        "POST",
        "/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(req["body"]),
        timeout=1800,
    )
    print(f"HTTP {r.status_code}")

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    print(f"finish_reason: {choice.get('finish_reason')}")
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    print(f"content length: {len(content):,}")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:16000])


def run_chunk_batch(
    client: Client,
    model: str,
    effort: str,
    poll_seconds: int,
) -> dict[str, Any]:
    submit_batch(
        client,
        CHUNK_INPUT,
        CHUNK_STATE,
        "stageA_evidence_chunk_appraisal",
        model,
        effort,
    )
    batch = watch_batch(
        client,
        CHUNK_STATE,
        CHUNK_OUTPUT,
        CHUNK_ERRORS,
        poll_seconds,
    )
    print(f"Chunk Batch terminal status: {batch.get('status')}")
    if batch.get("status") != "completed":
        raise RuntimeError(
            f"Chunk Batch ended with status {batch.get('status')}"
        )
    manifest = merge_chunks()
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            "Chunk merge incomplete. Repair failures before reducer stage."
        )
    return manifest


def run_reducer_batch(
    client: Client,
    model: str,
    effort: str,
    poll_seconds: int,
) -> dict[str, Any]:
    submit_batch(
        client,
        REDUCER_INPUT,
        REDUCER_STATE,
        "stageA_evidence_unit_reducer",
        model,
        effort,
    )
    batch = watch_batch(
        client,
        REDUCER_STATE,
        REDUCER_OUTPUT,
        REDUCER_ERRORS,
        poll_seconds,
    )
    print(f"Reducer Batch terminal status: {batch.get('status')}")
    if batch.get("status") != "completed":
        raise RuntimeError(
            f"Reducer Batch ended with status {batch.get('status')}"
        )
    return merge_reducers(model)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final Stage A evidence synthesis: chunk appraisal + "
            "evidence-unit reduction, without guideline rewriting."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "prepare-chunks",
            "test-chunk",
            "run-chunks",
            "merge-chunks",
            "prepare-reducers",
            "test-reducer",
            "run-reducers",
            "merge-reducers",
            "all",
        ],
        default="prepare-chunks",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    parser.add_argument(
        "--max-papers-per-chunk",
        type=int,
        default=75,
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=350000,
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--reset-chunk-state", action="store_true")
    parser.add_argument("--reset-reducer-state", action="store_true")
    args = parser.parse_args()

    if args.reset_chunk_state and CHUNK_STATE.exists():
        CHUNK_STATE.unlink()
        print(f"Deleted: {CHUNK_STATE}")
    if args.reset_reducer_state and REDUCER_STATE.exists():
        REDUCER_STATE.unlink()
        print(f"Deleted: {REDUCER_STATE}")

    if args.mode in {"prepare-chunks", "all"}:
        prepare_chunks(
            args.model,
            args.reasoning_effort,
            args.max_papers_per_chunk,
            args.max_chars_per_chunk,
        )
        if args.mode == "prepare-chunks":
            return 0

    if args.mode == "merge-chunks":
        merge_chunks()
        return 0

    if args.mode == "prepare-reducers":
        prepare_reducers(
            args.model,
            args.reasoning_effort,
        )
        return 0

    if args.mode == "merge-reducers":
        merge_reducers(args.model)
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this PowerShell session."
        )
    client = Client(api_key, args.retry_wait)

    if args.mode == "test-chunk":
        test_first_request(client, CHUNK_INPUT)
        return 0

    if args.mode == "run-chunks":
        run_chunk_batch(
            client,
            args.model,
            args.reasoning_effort,
            args.poll_seconds,
        )
        return 0

    if args.mode == "test-reducer":
        test_first_request(client, REDUCER_INPUT)
        return 0

    if args.mode == "run-reducers":
        run_reducer_batch(
            client,
            args.model,
            args.reasoning_effort,
            args.poll_seconds,
        )
        return 0

    if args.mode == "all":
        run_chunk_batch(
            client,
            args.model,
            args.reasoning_effort,
            args.poll_seconds,
        )
        prepare_reducers(
            args.model,
            args.reasoning_effort,
        )
        run_reducer_batch(
            client,
            args.model,
            args.reasoning_effort,
            args.poll_seconds,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
