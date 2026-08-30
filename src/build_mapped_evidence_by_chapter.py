#!/usr/bin/env python3
"""
Build the second integration input file from the completed GPT chapter-mapped
PubMed evidence dataset.

INPUT
-----
data/pubmed_selected_evidence_mapped.csv

OUTPUT
------
data/mapped_evidence_by_chapter.jsonl
    One JSON object per paper-chapter assignment. A paper mapped to multiple
    chapters appears once for each assigned chapter.

data/mapped_evidence_chapter_manifest.json
    Counts and QC by chapter.

data/unmappable_evidence_records.csv
    Records GPT explicitly marked unmappable; these are NOT sent into chapter
    evidence integration.

Safety policy
-------------
- Any record still carrying guideline/consensus guidance flags causes a hard
  failure. Guidance must never enter the evidence integration set.
- Only GPT chapter mappings are used for chapter assignment; the original
  PubMed search provenance is not used as the integration assignment.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT = DATA_DIR / "pubmed_selected_evidence_mapped.csv"
OUTPUT_JSONL = DATA_DIR / "mapped_evidence_by_chapter.jsonl"
MANIFEST = DATA_DIR / "mapped_evidence_chapter_manifest.json"
UNMAPPABLE_CSV = DATA_DIR / "unmappable_evidence_records.csv"

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

def clean(v):
    return " ".join((v or "").replace("\ufeff", "").split())

def truthy(v):
    return clean(v).casefold() in {"1", "true", "yes", "y"}

def split_semicolon(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    required = {
        "pmid", "title", "abstract", "publication_types",
        "gpt_chapter_ids", "gpt_mapping_confidence",
        "gpt_mapping_rationale", "gpt_unmappable",
    }
    missing = required - set(fields)
    if missing:
        raise RuntimeError("Missing required columns: " + ", ".join(sorted(missing)))

    # Hard safety check: guidance must not enter evidence integration.
    bad_guidance = []
    for r in rows:
        if truthy(r.get("exclude_guideline_consensus")) or truthy(r.get("possible_guidance_title_review")):
            bad_guidance.append(clean(r.get("pmid")))
    if bad_guidance:
        raise RuntimeError(
            f"HARD FAIL: {len(bad_guidance)} guidance records remain in mapped evidence. "
            f"Example PMIDs: {bad_guidance[:10]}"
        )

    unmappable = []
    output_records = []
    chapter_counts = Counter()
    confidence_counts = Counter()
    evidence_type_counts = Counter()
    unique_mapped_pmids = set()

    for r in rows:
        pmid = clean(r.get("pmid"))
        if not pmid:
            continue

        if truthy(r.get("gpt_unmappable")):
            unmappable.append(r)
            continue

        chapter_ids = split_semicolon(r.get("gpt_chapter_ids"))
        invalid = [c for c in chapter_ids if c not in CHAPTERS]
        if invalid:
            raise RuntimeError(f"Invalid chapter IDs for PMID {pmid}: {invalid}")
        if not chapter_ids:
            raise RuntimeError(f"Mapped record has no chapter_ids: PMID {pmid}")

        unique_mapped_pmids.add(pmid)
        confidence = clean(r.get("gpt_mapping_confidence"))
        confidence_counts[confidence or "missing"] += 1

        labels = split_semicolon(r.get("evidence_labels"))
        for label in labels:
            evidence_type_counts[label] += 1

        base = {
            "pmid": pmid,
            "doi": clean(r.get("doi")),
            "pmcid": clean(r.get("pmcid")),
            "title": clean(r.get("title")),
            "abstract": clean(r.get("abstract")),
            "authors": clean(r.get("authors")),
            "journal": clean(r.get("journal")),
            "publication_date": clean(r.get("publication_date")),
            "publication_year": clean(r.get("publication_year")),
            "publication_types": clean(r.get("publication_types")),
            "mesh_terms": clean(r.get("mesh_terms")),
            "keywords": clean(r.get("keywords")),
            "evidence_labels": clean(r.get("evidence_labels")),
            "is_rct": clean(r.get("is_rct")),
            "is_meta_analysis": clean(r.get("is_meta_analysis")),
            "is_systematic_review": clean(r.get("is_systematic_review")),
            "is_review": clean(r.get("is_review")),
            "gpt_mapping_confidence": confidence,
            "gpt_mapping_rationale": clean(r.get("gpt_mapping_rationale")),
            "gpt_model": clean(r.get("gpt_model")),
            "gpt_batch_id": clean(r.get("gpt_batch_id")),
        }

        # Deliberately do not copy original matched_chapter_ids/search provenance.
        for cid in chapter_ids:
            rec = dict(base)
            rec["chapter_id"] = cid
            rec["chapter_title"] = CHAPTERS[cid]
            output_records.append(rec)
            chapter_counts[cid] += 1

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with UNMAPPABLE_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unmappable)

    manifest = {
        "input_file": str(INPUT),
        "output_file": str(OUTPUT_JSONL),
        "input_rows": len(rows),
        "unique_mapped_pmids": len(unique_mapped_pmids),
        "unmappable_records_excluded": len(unmappable),
        "paper_chapter_assignments": len(output_records),
        "chapter_counts": {
            cid: {
                "title": CHAPTERS[cid],
                "evidence_records": chapter_counts[cid],
            }
            for cid in CHAPTERS
        },
        "mapping_confidence_counts_on_mapped_pmids": dict(confidence_counts),
        "evidence_label_counts_on_mapped_pmids": dict(evidence_type_counts),
        "safety": {
            "guideline_consensus_records_allowed": False,
            "possible_guidance_records_allowed": False,
            "original_pubmed_search_chapter_assignment_used": False,
            "gpt_mapping_only": True,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("Mapped evidence integration file created.")
    print(f"  input rows:                  {len(rows):,}")
    print(f"  unique mapped PMIDs:         {len(unique_mapped_pmids):,}")
    print(f"  unmappable excluded:         {len(unmappable):,}")
    print(f"  paper-chapter assignments:   {len(output_records):,}")
    print()
    for cid in CHAPTERS:
        print(f"  chapter {cid:>3}: {chapter_counts[cid]:,} evidence records")
    print()
    print(f"  JSONL:    {OUTPUT_JSONL}")
    print(f"  manifest: {MANIFEST}")
    print(f"  unmappable: {UNMAPPABLE_CSV}")

if __name__ == "__main__":
    main()
