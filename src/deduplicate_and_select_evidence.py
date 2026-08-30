#!/usr/bin/env python3
"""
Deduplicate PubMed results globally by PMID and select evidence types for the
ESMO PDAC 2015 -> August 2023 proof-of-concept.

INPUT
-----
data/pubmed_results.csv

OUTPUTS
-------
data/pubmed_unique_classified.csv
data/pubmed_selected_evidence.csv
data/pubmed_excluded_guidelines_consensus.csv
data/pubmed_unselected_other.csv
data/evidence_selection_summary.csv

Policy:
- RCTs, meta-analyses, systematic reviews, and reviews are marked.
- Guidelines / practice guidelines / consensus statements / consensus
  development conferences are explicitly excluded.
- Guideline/consensus exclusion overrides evidence-type classification.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT = Path("data/pubmed_results.csv")
DEFAULT_OUTDIR = Path("data")

MULTIVALUE_SOURCE_FIELDS = {
    "chapter_id": "matched_chapter_ids",
    "chapter_title": "matched_chapter_titles",
    "slice_start": "matched_slice_starts",
    "slice_end": "matched_slice_ends",
    "query_sha256": "matched_query_sha256",
}

GUIDANCE_PUBLICATION_TYPES = {
    "guideline",
    "practice guideline",
    "consensus statement",
    "consensus development conference",
    "consensus development conference, nih",
}

STRONG_GUIDANCE_TITLE_PATTERNS = [
    re.compile(r"\bclinical practice guidelines?\b", re.I),
    re.compile(r"\bpractice guidelines?\b", re.I),
    re.compile(r"\bconsensus statement\b", re.I),
    re.compile(r"\bexpert consensus\b", re.I),
    re.compile(r"\bconsensus recommendations?\b", re.I),
    re.compile(r"\bconsensus guidelines?\b", re.I),
    re.compile(r"\bguideline update\b", re.I),
    re.compile(r"^\s*(?:updated\s+|update\s+of\s+)?guidelines?\s+(?:for|on|of)\b", re.I),
    re.compile(r"\bESMO\b.{0,80}\bguidelines?\b", re.I),
    re.compile(r"\bASCO\b.{0,80}\bguidelines?\b", re.I),
    re.compile(r"\bNCCN\b.{0,80}\bguidelines?\b", re.I),
]

POSSIBLE_GUIDANCE_TITLE_PATTERNS = [
    re.compile(r"\bposition statement\b", re.I),
    re.compile(r"\bposition paper\b", re.I),
    re.compile(r"\bDelphi consensus\b", re.I),
    re.compile(r"\bconsensus report\b", re.I),
    re.compile(r"\brecommendations? from\b", re.I),
]

META_TITLE_RE = re.compile(r"\bmeta[\s-]?analys(?:is|es)\b", re.I)
SYSTEMATIC_REVIEW_TITLE_RE = re.compile(r"\bsystematic review\b", re.I)
RANDOMIZED_TITLE_RE = re.compile(
    r"\b(?:randomized|randomised)\b.{0,120}\b(?:trial|study)\b"
    r"|\b(?:trial|study)\b.{0,120}\b(?:randomized|randomised)\b",
    re.I,
)
PROTOCOL_TITLE_RE = re.compile(r"\bprotocol\b", re.I)


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\ufeff", "").split())


def split_multi(value: str | None) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [x.strip() for x in re.split(r"\s*[;|]\s*", text) if x.strip()]


def join_sorted(values: Iterable[str]) -> str:
    return "; ".join(sorted({normalize_text(v) for v in values if normalize_text(v)}, key=str.casefold))


def choose_richest(old: str, new: str) -> str:
    old_n = normalize_text(old)
    new_n = normalize_text(new)
    if not old_n:
        return new_n
    if not new_n:
        return old_n
    return new_n if len(new_n) > len(old_n) else old_n


def canonical_pt_set(value: str | None) -> set[str]:
    return {normalize_text(x).casefold() for x in split_multi(value)}


def publication_type_flags(publication_types: str, title: str) -> dict[str, bool]:
    pts = canonical_pt_set(publication_types)
    title_n = normalize_text(title)

    is_rct_pt = "randomized controlled trial" in pts
    is_meta_pt = "meta-analysis" in pts
    is_systematic_pt = "systematic review" in pts
    is_review_pt = "review" in pts

    is_rct_title = bool(RANDOMIZED_TITLE_RE.search(title_n)) and not bool(PROTOCOL_TITLE_RE.search(title_n))
    is_meta_title = bool(META_TITLE_RE.search(title_n))
    is_systematic_title = bool(SYSTEMATIC_REVIEW_TITLE_RE.search(title_n))

    return {
        "is_rct": is_rct_pt or (is_rct_title and not (is_review_pt or is_meta_pt or is_systematic_pt)),
        "is_meta_analysis": is_meta_pt or is_meta_title,
        "is_systematic_review": is_systematic_pt or is_systematic_title,
        "is_review": is_review_pt or is_systematic_pt or is_meta_pt or is_systematic_title or is_meta_title,
        "rct_by_publication_type": is_rct_pt,
        "rct_by_title_fallback": is_rct_title and not is_rct_pt,
        "meta_by_publication_type": is_meta_pt,
        "meta_by_title_fallback": is_meta_title and not is_meta_pt,
        "systematic_review_by_publication_type": is_systematic_pt,
        "systematic_review_by_title_fallback": is_systematic_title and not is_systematic_pt,
        "review_by_publication_type": is_review_pt,
    }


def guidance_flags(publication_types: str, title: str) -> dict[str, str | bool]:
    pts = canonical_pt_set(publication_types)
    title_n = normalize_text(title)

    matched_pt = sorted({pt for pt in pts if pt in GUIDANCE_PUBLICATION_TYPES}, key=str.casefold)
    matched_strong_title = [pat.pattern for pat in STRONG_GUIDANCE_TITLE_PATTERNS if pat.search(title_n)]
    matched_possible_title = [pat.pattern for pat in POSSIBLE_GUIDANCE_TITLE_PATTERNS if pat.search(title_n)]

    hard_exclude = bool(matched_pt or matched_strong_title)

    reasons = []
    if matched_pt:
        reasons.append("publication_type:" + "|".join(matched_pt))
    if matched_strong_title:
        reasons.append("strong_title_guidance_pattern")

    return {
        "exclude_guideline_consensus": hard_exclude,
        "guidance_exclusion_reason": "; ".join(reasons),
        "guidance_publication_type_matches": "; ".join(matched_pt),
        "strong_guidance_title_match": bool(matched_strong_title),
        "possible_guidance_title_review": bool(matched_possible_title),
    }


def load_and_deduplicate(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header.")

        required = {"pmid", "title", "abstract", "publication_types"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError("Input CSV is missing required columns: " + ", ".join(sorted(missing)))

        by_pmid = {}
        source_sets = {}
        rows_read = 0
        missing_pmid_rows = 0
        union_fields = {"publication_types", "mesh_terms", "keywords", "languages"}

        for row in reader:
            rows_read += 1
            pmid = normalize_text(row.get("pmid"))
            if not pmid:
                missing_pmid_rows += 1
                continue

            if pmid not in by_pmid:
                base = {k: normalize_text(v) for k, v in row.items() if k is not None}
                by_pmid[pmid] = base
                source_sets[pmid] = {out_name: set() for out_name in MULTIVALUE_SOURCE_FIELDS.values()}
                by_pmid[pmid]["source_row_count"] = "0"
            else:
                base = by_pmid[pmid]
                for key, value in row.items():
                    if key is None or key in MULTIVALUE_SOURCE_FIELDS or key in union_fields or key == "pmid":
                        continue
                    base[key] = choose_richest(base.get(key, ""), value or "")

                for key in union_fields:
                    base[key] = join_sorted(split_multi(base.get(key, "")) + split_multi(row.get(key, "")))

            base = by_pmid[pmid]
            base["source_row_count"] = str(int(base.get("source_row_count", "0") or 0) + 1)

            for key in union_fields:
                base[key] = join_sorted(split_multi(base.get(key, "")))

            for input_name, output_name in MULTIVALUE_SOURCE_FIELDS.items():
                value = normalize_text(row.get(input_name))
                if value:
                    source_sets[pmid][output_name].add(value)

        unique_rows = []
        for pmid, row in by_pmid.items():
            for output_name, vals in source_sets[pmid].items():
                row[output_name] = join_sorted(vals)
            unique_rows.append(row)

    unique_rows.sort(key=lambda r: int(r["pmid"]) if r["pmid"].isdigit() else r["pmid"])

    stats = {
        "source_rows": rows_read,
        "unique_pmids": len(unique_rows),
        "duplicate_source_rows_removed": rows_read - len(unique_rows) - missing_pmid_rows,
        "missing_pmid_rows_skipped": missing_pmid_rows,
    }
    return unique_rows, stats


def classify_rows(rows):
    counts = Counter()

    for row in rows:
        pflags = publication_type_flags(row.get("publication_types", ""), row.get("title", ""))
        gflags = guidance_flags(row.get("publication_types", ""), row.get("title", ""))

        for key, value in {**pflags, **gflags}.items():
            row[key] = "1" if value is True else "0" if value is False else str(value)

        labels = []
        if pflags["is_rct"]:
            labels.append("RCT")
        if pflags["is_meta_analysis"]:
            labels.append("META_ANALYSIS")
        if pflags["is_systematic_review"]:
            labels.append("SYSTEMATIC_REVIEW")
        elif pflags["is_review"]:
            labels.append("REVIEW")

        excluded_guidance = bool(gflags["exclude_guideline_consensus"])

        row["evidence_labels"] = "; ".join(labels)
        row["selected_evidence"] = "1" if (labels and not excluded_guidance) else "0"

        if excluded_guidance:
            row["selection_status"] = "EXCLUDED_GUIDELINE_OR_CONSENSUS"
        elif labels:
            row["selection_status"] = "SELECTED"
        else:
            row["selection_status"] = "NOT_SELECTED_OTHER_PUBLICATION_TYPE"

        counts["rct_unique"] += int(pflags["is_rct"])
        counts["meta_analysis_unique"] += int(pflags["is_meta_analysis"])
        counts["systematic_review_unique"] += int(pflags["is_systematic_review"])
        counts["review_any_unique"] += int(pflags["is_review"])
        counts["guideline_consensus_excluded_unique"] += int(excluded_guidance)
        counts["possible_guidance_title_review_unique"] += int(bool(gflags["possible_guidance_title_review"]))
        counts["selected_unique"] += int(row["selected_evidence"] == "1")

    return rows, counts


def output_fieldnames(rows):
    preferred = [
        "pmid", "doi", "pmcid", "title", "abstract", "authors", "affiliations",
        "journal", "publication_date", "publication_year", "publication_types",
        "mesh_terms", "keywords", "languages", "source_row_count",
        "matched_chapter_ids", "matched_chapter_titles", "matched_slice_starts",
        "matched_slice_ends", "matched_query_sha256",
        "is_rct", "is_meta_analysis", "is_systematic_review", "is_review",
        "evidence_labels", "exclude_guideline_consensus",
        "guidance_exclusion_reason", "guidance_publication_type_matches",
        "strong_guidance_title_match", "possible_guidance_title_review",
        "selected_evidence", "selection_status",
        "rct_by_publication_type", "rct_by_title_fallback",
        "meta_by_publication_type", "meta_by_title_fallback",
        "systematic_review_by_publication_type",
        "systematic_review_by_title_fallback", "review_by_publication_type",
    ]
    existing = set()
    for row in rows:
        existing.update(row.keys())
    result = [x for x in preferred if x in existing]
    result.extend(sorted(existing - set(result), key=str.casefold))
    return result


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, stats, counts):
    summary_rows = [
        ("source_rows", stats["source_rows"]),
        ("unique_pmids", stats["unique_pmids"]),
        ("duplicate_source_rows_removed", stats["duplicate_source_rows_removed"]),
        ("missing_pmid_rows_skipped", stats["missing_pmid_rows_skipped"]),
        ("rct_unique", counts["rct_unique"]),
        ("meta_analysis_unique", counts["meta_analysis_unique"]),
        ("systematic_review_unique", counts["systematic_review_unique"]),
        ("review_any_unique", counts["review_any_unique"]),
        ("guideline_consensus_excluded_unique", counts["guideline_consensus_excluded_unique"]),
        ("possible_guidance_title_review_unique", counts["possible_guidance_title_review_unique"]),
        ("selected_unique", counts["selected_unique"]),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count"])
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate PubMed by PMID, classify RCT/meta-analysis/systematic-review/review, and explicitly exclude guidelines/consensus statements."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rows, stats = load_and_deduplicate(args.input)
    rows, counts = classify_rows(rows)
    fields = output_fieldnames(rows)

    selected = [r for r in rows if r["selected_evidence"] == "1"]
    excluded_guidance = [r for r in rows if r["exclude_guideline_consensus"] == "1"]
    unselected_other = [
        r for r in rows
        if r["selected_evidence"] != "1" and r["exclude_guideline_consensus"] != "1"
    ]

    outdir = args.outdir
    write_csv(outdir / "pubmed_unique_classified.csv", rows, fields)
    write_csv(outdir / "pubmed_selected_evidence.csv", selected, fields)
    write_csv(outdir / "pubmed_excluded_guidelines_consensus.csv", excluded_guidance, fields)
    write_csv(outdir / "pubmed_unselected_other.csv", unselected_other, fields)
    write_summary(outdir / "evidence_selection_summary.csv", stats, counts)

    print()
    print("PubMed deduplication / evidence selection completed.")
    print(f"  source rows:                         {stats['source_rows']:,}")
    print(f"  unique PMIDs:                        {stats['unique_pmids']:,}")
    print(f"  duplicate source rows removed:       {stats['duplicate_source_rows_removed']:,}")
    print(f"  missing-PMID rows skipped:           {stats['missing_pmid_rows_skipped']:,}")
    print()
    print(f"  RCTs marked:                         {counts['rct_unique']:,}")
    print(f"  meta-analyses marked:                {counts['meta_analysis_unique']:,}")
    print(f"  systematic reviews marked:           {counts['systematic_review_unique']:,}")
    print(f"  reviews (any review class) marked:   {counts['review_any_unique']:,}")
    print()
    print(f"  guideline/consensus EXCLUDED:        {counts['guideline_consensus_excluded_unique']:,}")
    print(f"  possible guidance titles for QC:     {counts['possible_guidance_title_review_unique']:,}")
    print(f"  FINAL SELECTED evidence records:     {counts['selected_unique']:,}")
    print()
    print(f"  all unique classified: {outdir / 'pubmed_unique_classified.csv'}")
    print(f"  selected evidence:     {outdir / 'pubmed_selected_evidence.csv'}")
    print(f"  excluded guidance:     {outdir / 'pubmed_excluded_guidelines_consensus.csv'}")
    print(f"  other/unselected:      {outdir / 'pubmed_unselected_other.csv'}")
    print(f"  QC summary:            {outdir / 'evidence_selection_summary.csv'}")
    print()
    print("POLICY: guideline/consensus exclusion overrides RCT/review/meta-analysis flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
