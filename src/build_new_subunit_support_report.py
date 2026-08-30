#!/usr/bin/env python3
"""
STEP 2D1 — Deterministic support/QC report for the STEP 2B/2C proposed taxonomy.

NO OpenAI/API call.

Why this step exists
--------------------
STEP 2B proposed 115 broad themes and STEP 2C assigned all 1,191 candidate
PMID+chapter records to them. Before freezing ontology v2, we should inspect
how much evidence actually landed in each proposed theme.

IMPORTANT:
Paper count is a QC/support descriptor, NOT a rule for splitting evidence units
and NOT an automatic acceptance/rejection threshold. A medically coherent unit
may contain hundreds of papers. Conversely, a clinically important new topic
may be supported by only a small number of high-value studies and still merit
a unit after review.

Inputs
------
data/new_subunit_cluster_taxonomy_v1.json
data/new_subunit_candidate_assignments_expanded.jsonl

Outputs
-------
data/new_subunit_cluster_support_report.csv
data/new_subunit_cluster_support_report.json
data/new_subunit_cluster_support_manifest.json
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TAXONOMY = DATA / "new_subunit_cluster_taxonomy_v1.json"
ASSIGNMENTS = DATA / "new_subunit_candidate_assignments_expanded.jsonl"

OUT_CSV = DATA / "new_subunit_cluster_support_report.csv"
OUT_JSON = DATA / "new_subunit_cluster_support_report.json"
OUT_MANIFEST = DATA / "new_subunit_cluster_support_manifest.json"

CHAPTER_ORDER = ["1", "2", "3", "4.1", "4.2", "4.3", "5", "6"]


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
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


def text_has(text: str, *needles: str) -> bool:
    t = clean(text).lower()
    return any(n.lower() in t for n in needles)


def evidence_flags(row: dict[str, Any]) -> dict[str, bool]:
    labels = clean(row.get("evidence_labels"))
    ptypes = clean(row.get("publication_types"))
    combined = f"{labels} | {ptypes}"

    is_rct = text_has(
        combined,
        "randomized controlled trial",
        "randomised controlled trial",
        "controlled clinical trial",
        " rct",
        "rct ",
        "rct;",
        "[rct]",
    )
    is_meta = text_has(
        combined,
        "meta-analysis",
        "meta analysis",
        "meta_analysis",
    )
    is_sr = text_has(
        combined,
        "systematic review",
        "systematic_review",
    )
    is_review = text_has(
        combined,
        "review",
    )
    return {
        "rct": is_rct,
        "meta_analysis": is_meta,
        "systematic_review": is_sr,
        "review_any": is_review,
    }


def support_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 9:
        return "5-9"
    if n <= 24:
        return "10-24"
    if n <= 49:
        return "25-49"
    if n <= 99:
        return "50-99"
    return "100+"


def chapter_sort_key(cid: str) -> int:
    try:
        return CHAPTER_ORDER.index(cid)
    except ValueError:
        return 999


def main() -> None:
    taxonomy = load_json(TAXONOMY)
    assignments = load_jsonl(ASSIGNMENTS)

    # Canonical cluster metadata from STEP 2B.
    clusters: dict[tuple[str, str], dict[str, Any]] = {}
    chapter_titles: dict[str, str] = {}

    for ch in taxonomy.get("chapters", []):
        cid = clean(ch.get("chapter_id"))
        chapter_titles[cid] = clean(ch.get("chapter_title"))
        for c in ch.get("proposed_clusters", []):
            cluster_id = clean(c.get("temporary_cluster_id"))
            key = (cid, cluster_id)
            if key in clusters:
                raise RuntimeError(f"Duplicate cluster key in taxonomy: {key}")
            clusters[key] = {
                "chapter_id": cid,
                "chapter_title": chapter_titles[cid],
                "cluster_id": cluster_id,
                "title": clean(c.get("title")),
                "disposition": clean(c.get("disposition")),
                "definition": clean(c.get("definition")),
                "boundary": clean(c.get("boundary")),
                "existing_unit_id": clean(c.get("existing_unit_id")),
                "rationale": clean(c.get("rationale")),
            }

    if not clusters:
        raise RuntimeError("No clusters found in taxonomy.")

    # Deduplicate exact paper-cluster assignments defensively.
    seen_assignments = set()
    per_cluster_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    duplicate_assignment_rows = 0

    for row in assignments:
        pmid = clean(row.get("pmid"))
        cid = clean(row.get("chapter_id"))
        cluster_id = clean(row.get("assigned_cluster_id"))
        role = clean(row.get("assignment_role"))

        key = (cid, cluster_id)
        if key not in clusters:
            raise RuntimeError(
                f"Assignment refers to unknown cluster {key}, PMID {pmid}"
            )
        if role not in {"primary", "secondary"}:
            raise RuntimeError(
                f"Invalid assignment_role={role!r} for PMID {pmid}, cluster {key}"
            )

        assignment_key = (pmid, cid, cluster_id, role)
        if assignment_key in seen_assignments:
            duplicate_assignment_rows += 1
            continue
        seen_assignments.add(assignment_key)
        per_cluster_rows[key].append(row)

    report = []

    for key, meta in clusters.items():
        rows = per_cluster_rows.get(key, [])

        all_pmids = {clean(r.get("pmid")) for r in rows if clean(r.get("pmid"))}
        primary_pmids = {
            clean(r.get("pmid"))
            for r in rows
            if clean(r.get("assignment_role")) == "primary"
        }
        secondary_pmids = {
            clean(r.get("pmid"))
            for r in rows
            if clean(r.get("assignment_role")) == "secondary"
        }

        rct_pmids = set()
        ma_pmids = set()
        sr_pmids = set()
        review_pmids = set()
        high_conf_pmids = set()
        medium_conf_pmids = set()
        low_conf_pmids = set()

        for r in rows:
            pmid = clean(r.get("pmid"))
            flags = evidence_flags(r)
            if flags["rct"]:
                rct_pmids.add(pmid)
            if flags["meta_analysis"]:
                ma_pmids.add(pmid)
            if flags["systematic_review"]:
                sr_pmids.add(pmid)
            if flags["review_any"]:
                review_pmids.add(pmid)

            conf = clean(r.get("assignment_confidence")).lower()
            if conf == "high":
                high_conf_pmids.add(pmid)
            elif conf == "medium":
                medium_conf_pmids.add(pmid)
            elif conf == "low":
                low_conf_pmids.add(pmid)

        n = len(all_pmids)

        item = {
            **meta,
            "unique_supporting_pmids": n,
            "primary_pmids": len(primary_pmids),
            "secondary_pmids": len(secondary_pmids),
            "rct_pmids": len(rct_pmids),
            "meta_analysis_pmids": len(ma_pmids),
            "systematic_review_pmids": len(sr_pmids),
            "review_any_pmids": len(review_pmids),
            "high_confidence_pmids": len(high_conf_pmids),
            "medium_confidence_pmids": len(medium_conf_pmids),
            "low_confidence_pmids": len(low_conf_pmids),
            "support_bin": support_bin(n),
            "zero_assignment_flag": n == 0,
            "single_paper_flag": n == 1,
            "high_volume_100_plus_flag": n >= 100,
            "qc_note": (
                "COUNT IS DESCRIPTIVE ONLY — do not split or reject a medically "
                "coherent unit solely because of paper count."
            ),
        }
        report.append(item)

    report.sort(
        key=lambda x: (
            chapter_sort_key(x["chapter_id"]),
            x["disposition"],
            -x["unique_supporting_pmids"],
            x["title"].casefold(),
        )
    )

    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "chapter_id",
        "chapter_title",
        "cluster_id",
        "title",
        "disposition",
        "existing_unit_id",
        "unique_supporting_pmids",
        "primary_pmids",
        "secondary_pmids",
        "rct_pmids",
        "meta_analysis_pmids",
        "systematic_review_pmids",
        "review_any_pmids",
        "high_confidence_pmids",
        "medium_confidence_pmids",
        "low_confidence_pmids",
        "support_bin",
        "zero_assignment_flag",
        "single_paper_flag",
        "high_volume_100_plus_flag",
        "definition",
        "boundary",
        "rationale",
        "qc_note",
    ]

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(report)

    disposition_counts = Counter(x["disposition"] for x in report)
    zero_counts = Counter(
        x["disposition"] for x in report if x["unique_supporting_pmids"] == 0
    )
    support_bins_new = Counter(
        x["support_bin"]
        for x in report
        if x["disposition"] == "NEW_SUBUNIT"
    )

    new_counts = [
        x["unique_supporting_pmids"]
        for x in report
        if x["disposition"] == "NEW_SUBUNIT"
    ]

    manifest = {
        "taxonomy_clusters_total": len(report),
        "disposition_counts": dict(disposition_counts),
        "duplicate_assignment_rows_deduplicated": duplicate_assignment_rows,
        "clusters_with_zero_assignments_by_disposition": dict(zero_counts),
        "new_subunit_support_bins": dict(support_bins_new),
        "new_subunit_support_distribution": {
            "n_units": len(new_counts),
            "min": min(new_counts) if new_counts else None,
            "median": statistics.median(new_counts) if new_counts else None,
            "mean": round(statistics.mean(new_counts), 2) if new_counts else None,
            "max": max(new_counts) if new_counts else None,
        },
        "principle": (
            "Support counts are QC descriptors only. They must not be used as "
            "automatic split/merge/accept/reject thresholds. Medical coherence "
            "defines evidence-unit boundaries."
        ),
        "next_steps": [
            "Review zero/very-low-support proposed NEW_SUBUNIT themes for true distinctness.",
            "Review high-volume NEW_SUBUNIT themes without splitting them solely by count.",
            "Resolve the 9 NEW_MAJOR_CHAPTER_CANDIDATE records separately.",
            "Then freeze final ontology v2 and build the canonical all-evidence-by-unit dataset.",
        ],
    }

    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    new_units = [x for x in report if x["disposition"] == "NEW_SUBUNIT"]
    smallest = sorted(
        new_units,
        key=lambda x: (
            x["unique_supporting_pmids"],
            chapter_sort_key(x["chapter_id"]),
            x["title"].casefold(),
        ),
    )[:15]
    largest = sorted(
        new_units,
        key=lambda x: x["unique_supporting_pmids"],
        reverse=True,
    )[:15]

    print()
    print("STEP 2D1 support/QC report completed.")
    print(f"  taxonomy clusters total:       {len(report):,}")
    print(f"  NEW_SUBUNIT:                   {disposition_counts['NEW_SUBUNIT']:,}")
    print(f"  MERGE_INTO_EXISTING_UNIT:      {disposition_counts['MERGE_INTO_EXISTING_UNIT']:,}")
    print(f"  OUT_OF_SCOPE_THEME:            {disposition_counts['OUT_OF_SCOPE_THEME']:,}")
    print(f"  duplicate assignment rows:     {duplicate_assignment_rows:,}")
    print()

    if new_counts:
        print("NEW_SUBUNIT support distribution:")
        print(f"  min:                           {min(new_counts):,}")
        print(f"  median:                        {statistics.median(new_counts)}")
        print(f"  mean:                          {statistics.mean(new_counts):.2f}")
        print(f"  max:                           {max(new_counts):,}")
        print()
        print("Support bins:")
        for b in ["0", "1", "2-4", "5-9", "10-24", "25-49", "50-99", "100+"]:
            print(f"  {b:>5}: {support_bins_new[b]:,}")
        print()

    print("Smallest 15 NEW_SUBUNIT themes:")
    for x in smallest:
        print(
            f"  chapter {x['chapter_id']:>3} | "
            f"{x['unique_supporting_pmids']:>4} papers | "
            f"RCT {x['rct_pmids']:>3} | MA {x['meta_analysis_pmids']:>3} | "
            f"SR {x['systematic_review_pmids']:>3} | {x['title']}"
        )

    print()
    print("Largest 15 NEW_SUBUNIT themes:")
    for x in largest:
        print(
            f"  chapter {x['chapter_id']:>3} | "
            f"{x['unique_supporting_pmids']:>4} papers | "
            f"RCT {x['rct_pmids']:>3} | MA {x['meta_analysis_pmids']:>3} | "
            f"SR {x['systematic_review_pmids']:>3} | {x['title']}"
        )

    print()
    print(f"  CSV:      {OUT_CSV}")
    print(f"  JSON:     {OUT_JSON}")
    print(f"  manifest: {OUT_MANIFEST}")
    print()
    print("No OpenAI/API call was made.")


if __name__ == "__main__":
    main()
