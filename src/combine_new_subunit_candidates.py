#!/usr/bin/env python3
"""
STEP 2A — Consolidate all NEW_SUBUNIT_CANDIDATE records before clustering.

This step makes NO OpenAI call.

Inputs
------
data/recovery_new_subunit_candidates.jsonl
    Novel-topic recovery records classified as NEW_SUBUNIT_CANDIDATE.

data/recovered_questionables_new_subunit_candidates.jsonl
    New-subunit candidates generated after re-mapping QUESTIONABLE records.

data/manual_evidence_exclusions.jsonl
    Manual QC exclusions. Any matching PMID is removed from the candidate pool.

Outputs
-------
data/new_subunit_candidates_combined.jsonl
    Canonical combined candidate pool, one row per PMID + chapter_id.

data/new_subunit_candidates_combined_manifest.json
    QC counts by chapter and provenance.

Important
---------
- No candidate is silently discarded except explicit manual exclusions.
- Duplicate PMID + chapter_id records are merged while preserving all candidate
  labels/descriptions/rationales and provenance.
- This is only preparation for the later semantic clustering step.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCE_NOVEL = DATA / "recovery_new_subunit_candidates.jsonl"
SOURCE_RECOVERED = DATA / "recovered_questionables_new_subunit_candidates.jsonl"
MANUAL_EXCLUSIONS = DATA / "manual_evidence_exclusions.jsonl"

OUTPUT = DATA / "new_subunit_candidates_combined.jsonl"
MANIFEST = DATA / "new_subunit_candidates_combined_manifest.json"

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


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


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
                    f"Invalid JSONL in {path.name} line {line_no}: {e}"
                ) from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def manual_excluded_pmids() -> set[str]:
    return {
        clean(row.get("pmid"))
        for row in load_jsonl(MANUAL_EXCLUSIONS, optional=True)
        if clean(row.get("pmid"))
    }


def candidate_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    Normalize the different field names produced by the two upstream workflows.
    Returns: candidate_title, candidate_description, rationale, confidence
    """
    title = clean(
        row.get("candidate_title")
        or row.get("recovered_candidate_title")
        or row.get("novel_topic_label")
    )
    description = clean(
        row.get("candidate_description")
        or row.get("recovered_candidate_description")
        or row.get("novel_topic_description")
    )
    rationale = clean(
        row.get("rationale")
        or row.get("recovered_submapping_rationale")
        or row.get("submapping_rationale")
    )
    confidence = clean(
        row.get("confidence")
        or row.get("recovered_submapping_confidence")
        or row.get("submapping_confidence")
    )
    return title, description, rationale, confidence


def main() -> None:
    excluded_pmids = manual_excluded_pmids()

    sources = [
        ("novel_topic_recovery", SOURCE_NOVEL),
        ("recovered_questionable_submapping", SOURCE_RECOVERED),
    ]

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    proposals: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    provenance: dict[tuple[str, str], set[str]] = defaultdict(set)

    source_counts = Counter()
    source_excluded = Counter()
    duplicate_rows = 0

    for source_name, source_path in sources:
        rows = load_jsonl(source_path)
        source_counts[source_name] = len(rows)

        for row in rows:
            pmid = clean(row.get("pmid"))
            chapter_id = clean(row.get("chapter_id"))

            if not pmid:
                raise RuntimeError(f"{source_name}: candidate row without PMID")
            if chapter_id not in CHAPTERS:
                raise RuntimeError(
                    f"{source_name}: invalid chapter_id={chapter_id!r} for PMID {pmid}"
                )

            if pmid in excluded_pmids:
                source_excluded[source_name] += 1
                continue

            title, description, rationale, confidence = candidate_fields(row)

            if not title:
                raise RuntimeError(
                    f"{source_name}: NEW_SUBUNIT_CANDIDATE without candidate title "
                    f"for PMID {pmid}, chapter {chapter_id}"
                )
            if not description:
                raise RuntimeError(
                    f"{source_name}: NEW_SUBUNIT_CANDIDATE without candidate description "
                    f"for PMID {pmid}, chapter {chapter_id}"
                )

            key = (pmid, chapter_id)

            if key not in grouped:
                grouped[key] = {
                    "pmid": pmid,
                    "chapter_id": chapter_id,
                    "chapter_title": CHAPTERS[chapter_id],
                    "doi": clean(row.get("doi")),
                    "pmcid": clean(row.get("pmcid")),
                    "title": clean(row.get("title")),
                    "abstract": clean(row.get("abstract")),
                    "authors": clean(row.get("authors")),
                    "journal": clean(row.get("journal")),
                    "publication_date": clean(row.get("publication_date")),
                    "publication_year": clean(row.get("publication_year")),
                    "publication_types": clean(row.get("publication_types")),
                    "evidence_labels": clean(row.get("evidence_labels")),
                    "mesh_terms": clean(row.get("mesh_terms")),
                    "keywords": clean(row.get("keywords")),
                }
            else:
                duplicate_rows += 1
                # Keep the richer scientific metadata if the upstream copies differ.
                for field in [
                    "doi", "pmcid", "title", "abstract", "authors", "journal",
                    "publication_date", "publication_year", "publication_types",
                    "evidence_labels", "mesh_terms", "keywords",
                ]:
                    existing = clean(grouped[key].get(field))
                    new = clean(row.get(field))
                    if len(new) > len(existing):
                        grouped[key][field] = new

            proposal = {
                "candidate_title": title,
                "candidate_description": description,
                "mapping_rationale": rationale,
                "mapping_confidence": confidence,
                "source": source_name,
            }

            # Preserve distinct proposals only.
            signature = json.dumps(proposal, sort_keys=True, ensure_ascii=False)
            existing_signatures = {
                json.dumps(x, sort_keys=True, ensure_ascii=False)
                for x in proposals[key]
            }
            if signature not in existing_signatures:
                proposals[key].append(proposal)

            provenance[key].add(source_name)

    combined: list[dict[str, Any]] = []
    chapter_counts = Counter()
    unique_pmids = set()
    multi_proposal_records = 0

    for key in sorted(
        grouped,
        key=lambda x: (
            list(CHAPTERS).index(x[1]),
            int(x[0]) if x[0].isdigit() else x[0],
        ),
    ):
        row = grouped[key]
        row["candidate_proposals"] = proposals[key]
        row["candidate_proposal_count"] = len(proposals[key])
        row["candidate_sources"] = sorted(provenance[key])

        if len(proposals[key]) > 1:
            multi_proposal_records += 1

        combined.append(row)
        chapter_counts[row["chapter_id"]] += 1
        unique_pmids.add(row["pmid"])

    write_jsonl(OUTPUT, combined)

    manifest = {
        "status": "READY_FOR_NEW_SUBUNIT_CLUSTERING",
        "source_files": {
            "novel_topic_recovery": str(SOURCE_NOVEL),
            "recovered_questionable_submapping": str(SOURCE_RECOVERED),
            "manual_exclusions": (
                str(MANUAL_EXCLUSIONS) if MANUAL_EXCLUSIONS.exists() else None
            ),
        },
        "source_row_counts": dict(source_counts),
        "source_rows_removed_by_manual_exclusion": dict(source_excluded),
        "input_candidate_rows_total": sum(source_counts.values()),
        "duplicate_source_rows_merged": duplicate_rows,
        "combined_unique_pmid_chapter_candidates": len(combined),
        "combined_unique_pmids": len(unique_pmids),
        "records_with_multiple_preserved_candidate_proposals": multi_proposal_records,
        "chapter_counts": {
            cid: {
                "title": CHAPTERS[cid],
                "candidate_records": chapter_counts[cid],
            }
            for cid in CHAPTERS
        },
        "next_step": (
            "Cluster candidates semantically WITHIN each major chapter. "
            "Do not create final ontology units until cross-paper clusters have "
            "been consolidated and reviewed."
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("STEP 2A complete — new-subunit candidates consolidated.")
    print(f"  source novel-topic candidates:        {source_counts['novel_topic_recovery']:,}")
    print(f"  source recovered-questionable:        {source_counts['recovered_questionable_submapping']:,}")
    print(f"  total source candidate rows:          {sum(source_counts.values()):,}")
    print(f"  manual-exclusion rows removed:        {sum(source_excluded.values()):,}")
    print(f"  duplicate source rows merged:         {duplicate_rows:,}")
    print(f"  unique PMID+chapter candidates:       {len(combined):,}")
    print(f"  unique PMIDs:                         {len(unique_pmids):,}")
    print(f"  records with >1 candidate proposal:   {multi_proposal_records:,}")
    print()
    for cid in CHAPTERS:
        print(f"  chapter {cid:>3}: {chapter_counts[cid]:,} candidates")
    print()
    print(f"  combined JSONL: {OUTPUT}")
    print(f"  manifest:       {MANIFEST}")
    print()
    print("No OpenAI/API call was made.")


if __name__ == "__main__":
    main()
