#!/usr/bin/env python3
"""
Build the canonical chapter-wise integration master for the ESMO PDAC 2015
-> August 2023 proof-of-concept.

This step DOES NOT call OpenAI and DOES NOT update the guideline.
It only joins:
  A) the frozen ESMO 2015 baseline by chapter
  B) the completed GPT-mapped new PubMed evidence by chapter

INPUTS
------
data/esmo2015_baseline_by_chapter.json
data/mapped_evidence_by_chapter.jsonl
data/mapped_evidence_chapter_manifest.json

OUTPUTS
-------
data/guideline_integration_master.jsonl
    Exactly 8 JSON objects, one per guideline chapter. Each object contains the
    full old ESMO chapter baseline plus all newly mapped evidence records.

data/guideline_integration_master_manifest.json
    Deterministic QC counts and source file hashes.

No evidence is discarded in this join.
Guidelines/consensus statements must already have been excluded upstream.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

BASELINE_PATH = DATA_DIR / "esmo2015_baseline_by_chapter.json"
EVIDENCE_PATH = DATA_DIR / "mapped_evidence_by_chapter.jsonl"
EVIDENCE_MANIFEST_PATH = DATA_DIR / "mapped_evidence_chapter_manifest.json"

OUTPUT_PATH = DATA_DIR / "guideline_integration_master.jsonl"
OUTPUT_MANIFEST_PATH = DATA_DIR / "guideline_integration_master_manifest.json"

CHAPTER_ORDER = ["1", "2", "3", "4.1", "4.2", "4.3", "5", "6"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def main():
    baseline = load_json(BASELINE_PATH)
    evidence_manifest = load_json(EVIDENCE_MANIFEST_PATH)
    evidence_rows = load_jsonl(EVIDENCE_PATH)

    baseline_by_id = {str(ch["chapter_id"]): ch for ch in baseline["chapters"]}
    if set(baseline_by_id) != set(CHAPTER_ORDER):
        raise RuntimeError(
            f"Baseline chapter IDs mismatch. Found: {sorted(baseline_by_id)}"
        )

    evidence_by_chapter = defaultdict(list)
    seen_assignments = set()

    for rec in evidence_rows:
        cid = str(rec.get("chapter_id", "")).strip()
        pmid = str(rec.get("pmid", "")).strip()

        if cid not in CHAPTER_ORDER:
            raise RuntimeError(f"Invalid evidence chapter_id: {cid}")
        if not pmid:
            raise RuntimeError("Evidence record without PMID encountered.")

        # Strong safety check: the evidence-builder deliberately omits old PubMed
        # search provenance and guidance flags. We additionally reject obvious
        # guidance status if such a field somehow reappears.
        for flag in ("exclude_guideline_consensus", "possible_guidance_title_review"):
            value = str(rec.get(flag, "")).strip().lower()
            if value in {"1", "true", "yes", "y"}:
                raise RuntimeError(
                    f"HARD FAIL: guidance record reached integration master: "
                    f"PMID {pmid}, field {flag}"
                )

        key = (cid, pmid)
        if key in seen_assignments:
            raise RuntimeError(
                f"Duplicate paper-chapter assignment in mapped evidence: {key}"
            )
        seen_assignments.add(key)
        evidence_by_chapter[cid].append(rec)

    # Stable deterministic ordering: evidence class priority, newest first, PMID.
    def evidence_priority(rec):
        # Lower number = higher priority.
        if str(rec.get("is_meta_analysis", "")).strip() == "1":
            p = 0
        elif str(rec.get("is_systematic_review", "")).strip() == "1":
            p = 1
        elif str(rec.get("is_rct", "")).strip() == "1":
            p = 2
        elif str(rec.get("is_review", "")).strip() == "1":
            p = 3
        else:
            p = 4
        try:
            year = int(str(rec.get("publication_year", "")).strip() or 0)
        except ValueError:
            year = 0
        try:
            pmid_num = int(str(rec.get("pmid", "")).strip())
        except ValueError:
            pmid_num = 0
        return (p, -year, -pmid_num)

    master_rows = []
    chapter_counts = {}

    for cid in CHAPTER_ORDER:
        old = baseline_by_id[cid]
        new_evidence = sorted(evidence_by_chapter[cid], key=evidence_priority)

        chapter_record = {
            "chapter_id": cid,
            "chapter_title": old["title"],
            "integration_status": "INPUT_ONLY_NOT_YET_UPDATED",
            "baseline_2015": {
                "source_document": baseline["source"],
                "source_pages_pdf": old.get("source_pages_pdf", []),
                "source_text": old.get(
                    "source_text_normalized_from_original_pdf", ""
                ),
                "summary_table_4_items": old.get("summary_table_4_items", {}),
                "tables": old.get("tables", []),
                "figures": old.get("figures", []),
                "cited_reference_numbers": old.get(
                    "cited_reference_numbers", []
                ),
                "cited_references": old.get("cited_references", []),
                "grading_system": baseline.get("grading_system", {}),
                "known_source_internal_inconsistencies": baseline.get(
                    "known_source_internal_inconsistencies_to_preserve_for_review",
                    [],
                ),
            },
            "new_evidence_2015_to_2023_08": new_evidence,
            "new_evidence_count": len(new_evidence),
            "input_policy": {
                "language_for_future_integration": "English",
                "guidelines_and_consensus_statements_excluded": True,
                "possible_guidance_titles_excluded": True,
                "unmappable_evidence_excluded_from_chapter_integration": True,
                "old_guideline_content_must_not_be_silently_corrected": True,
                "new_evidence_must_be_discussed_before_change_decisions": True,
                "guideline_content_should_change_only_when_new_evidence_is_sufficient": True,
                "reviews_are_contextual_unless_they_provide_high_level_synthesis": True,
                "rcts_systematic_reviews_meta_analyses_have_priority_for_change_decisions": True,
                "abstract_level_limitations_must_be_respected": True,
            },
        }
        master_rows.append(chapter_record)
        chapter_counts[cid] = len(new_evidence)

    # Cross-check against the prior manifest.
    expected_counts = {
        cid: int(
            evidence_manifest["chapter_counts"][cid]["evidence_records"]
        )
        for cid in CHAPTER_ORDER
    }
    if chapter_counts != expected_counts:
        raise RuntimeError(
            f"Chapter counts changed during join.\n"
            f"Expected: {expected_counts}\nObserved: {chapter_counts}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in master_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "document_type": "guideline_evidence_integration_master_manifest",
        "status": "INPUT_ONLY_NOT_YET_UPDATED",
        "baseline_file": str(BASELINE_PATH),
        "baseline_sha256": sha256_file(BASELINE_PATH),
        "mapped_evidence_file": str(EVIDENCE_PATH),
        "mapped_evidence_sha256": sha256_file(EVIDENCE_PATH),
        "mapped_evidence_manifest_file": str(EVIDENCE_MANIFEST_PATH),
        "mapped_evidence_manifest_sha256": sha256_file(
            EVIDENCE_MANIFEST_PATH
        ),
        "output_file": str(OUTPUT_PATH),
        "chapter_count": len(master_rows),
        "paper_chapter_assignments": sum(chapter_counts.values()),
        "chapter_counts": {
            cid: {
                "title": baseline_by_id[cid]["title"],
                "new_evidence_records": chapter_counts[cid],
            }
            for cid in CHAPTER_ORDER
        },
        "global_baseline_reference_count": len(
            baseline.get("full_original_reference_list", [])
        ),
        "integration_policy": {
            "guidelines_consensus_excluded": True,
            "integration_not_yet_performed": True,
            "future_model": "gpt-5.6-sol",
            "future_reasoning_effort": "high",
            "future_api": "Responses API",
        },
    }
    OUTPUT_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Guideline integration master created.")
    print(f"  chapters:                    {len(master_rows)}")
    print(
        f"  paper-chapter assignments:   "
        f"{sum(chapter_counts.values()):,}"
    )
    print(
        f"  original references:         "
        f"{len(baseline.get('full_original_reference_list', []))}"
    )
    print()
    for cid in CHAPTER_ORDER:
        print(
            f"  chapter {cid:>3}: "
            f"{chapter_counts[cid]:,} mapped evidence records"
        )
    print()
    print(f"  master JSONL: {OUTPUT_PATH}")
    print(f"  manifest:     {OUTPUT_MANIFEST_PATH}")
    print()
    print("No GPT/API call was made. No guideline text was updated.")


if __name__ == "__main__":
    main()
