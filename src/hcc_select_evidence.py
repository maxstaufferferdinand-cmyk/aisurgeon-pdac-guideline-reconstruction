from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def lower_blob(row: dict[str, str]) -> str:
    return " ".join(
        [
            row.get("title", ""),
            row.get("abstract", ""),
            row.get("publication_types", ""),
            row.get("mesh_terms", ""),
        ]
    ).lower()


GUIDANCE_RE = re.compile(
    r"\b("
    r"clinical practice guideline|practice guideline|guideline|guidelines|treatment guideline|"
    r"consensus statement|expert consensus|consensus conference|consensus development|"
    r"position statement|society recommendation|recommendations from|recommendation statement|"
    r"white paper|appropriateness criteria"
    r")\b",
    re.I,
)
AMBIGUOUS_GUIDANCE_RE = re.compile(r"\b(recommendation|recommendations|consensus|guidance)\b", re.I)


def classify(row: dict[str, str]) -> tuple[str, str, bool, bool]:
    blob = lower_blob(row)
    pubtypes = row.get("publication_types", "").lower()
    title = row.get("title", "").lower()
    is_guidance = bool(GUIDANCE_RE.search(blob))
    is_ambiguous_guidance = bool(AMBIGUOUS_GUIDANCE_RE.search(blob)) and not is_guidance

    is_meta = "meta-analysis" in pubtypes or "meta analysis" in title or "meta-analysis" in title
    is_systematic = "systematic review" in pubtypes or "systematic review" in title
    is_rct = (
        "randomized controlled trial" in pubtypes
        or "controlled clinical trial" in pubtypes
        or bool(re.search(r"\brandomi[sz]ed\b", title))
        or bool(re.search(r"\brandomi[sz]ed\b", row.get("abstract", "").lower()))
    )
    is_review = "review" in pubtypes or "review" in title

    if is_meta:
        evidence_type = "META_ANALYSIS"
    elif is_systematic:
        evidence_type = "SYSTEMATIC_REVIEW"
    elif is_rct:
        evidence_type = "RANDOMIZED_CONTROLLED_TRIAL"
    elif is_review:
        evidence_type = "OTHER_REVIEW"
    else:
        evidence_type = "NON_SELECTED"

    if is_guidance:
        decision = "EXCLUDE_GUIDELINE_CONSENSUS"
    elif is_ambiguous_guidance:
        decision = "AMBIGUOUS_GUIDANCE_QC"
    elif evidence_type == "NON_SELECTED":
        decision = "EXCLUDE_NON_TARGET_EVIDENCE_TYPE"
    else:
        decision = "SELECT"
    return decision, evidence_type, is_guidance, is_ambiguous_guidance


def build_provenance(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"query_ids": set(), "chapter_ids": set(), "unit_ids": set(), "slices": set()})
    for row in rows:
        pmid = row["pmid"]
        grouped[pmid]["query_ids"].add(row["query_id"])
        grouped[pmid]["chapter_ids"].add(row["chapter_id"])
        grouped[pmid]["unit_ids"].add(row["unit_id"])
        grouped[pmid]["slices"].add(f"{row['slice_start']}:{row['slice_end']}")
    return {
        pmid: {
            "query_ids": sorted(value["query_ids"]),
            "chapter_ids": sorted(value["chapter_ids"]),
            "unit_ids": sorted(value["unit_ids"]),
            "date_slices": sorted(value["slices"]),
        }
        for pmid, value in grouped.items()
    }


def enrich(row: dict[str, str], decision: str, evidence_type: str, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "selection_decision": decision,
        "evidence_type": evidence_type,
        "provenance_query_ids": "|".join(provenance.get("query_ids", [])),
        "provenance_chapter_ids": "|".join(provenance.get("chapter_ids", [])),
        "provenance_unit_ids": "|".join(provenance.get("unit_ids", [])),
        "provenance_date_slices": "|".join(provenance.get("date_slices", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically select HCC PubMed evidence.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    records = read_csv(hcc_root / "data" / "pubmed_unique_records_v2.csv")
    provenance = build_provenance(read_csv(hcc_root / "data" / "pubmed_query_provenance_v2.csv"))

    selected: list[dict[str, Any]] = []
    excluded_guidance: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    excluded_non: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()

    seen_pmids: set[str] = set()
    duplicate_pmids = 0
    for row in records:
        pmid = row["pmid"]
        if pmid in seen_pmids:
            duplicate_pmids += 1
            continue
        seen_pmids.add(pmid)
        decision, evidence_type, _is_guidance, _is_ambiguous = classify(row)
        enriched = enrich(row, decision, evidence_type, provenance.get(pmid, {}))
        decision_counts[decision] += 1
        evidence_counts[evidence_type] += 1
        if decision == "SELECT":
            selected.append(enriched)
        elif decision == "EXCLUDE_GUIDELINE_CONSENSUS":
            excluded_guidance.append(enriched)
        elif decision == "AMBIGUOUS_GUIDANCE_QC":
            ambiguous.append(enriched)
        else:
            excluded_non.append(enriched)

    fieldnames = list(records[0].keys()) + [
        "selection_decision",
        "evidence_type",
        "provenance_query_ids",
        "provenance_chapter_ids",
        "provenance_unit_ids",
        "provenance_date_slices",
    ]
    write_csv(hcc_root / "data" / "selected_evidence_v2.csv", selected, fieldnames)
    write_csv(hcc_root / "data" / "excluded_guideline_consensus_v2.csv", excluded_guidance, fieldnames)
    write_csv(hcc_root / "data" / "ambiguous_guidance_qc_v2.csv", ambiguous, fieldnames)
    write_csv(hcc_root / "data" / "excluded_non_target_evidence_v2.csv", excluded_non, fieldnames)
    summary = {
        "created_at": utc_now(),
        "raw_pubmed_row_count": len(records),
        "unique_pmid_count": len(seen_pmids),
        "duplicate_unique_record_rows_skipped": duplicate_pmids,
        "selected_count": len(selected),
        "excluded_guideline_consensus_count": len(excluded_guidance),
        "ambiguous_guidance_count": len(ambiguous),
        "excluded_non_target_count": len(excluded_non),
        "decision_counts": dict(decision_counts),
        "evidence_type_counts_before_exclusion": dict(evidence_counts),
        "selected_evidence_type_counts": dict(Counter(row["evidence_type"] for row in selected)),
    }
    atomic_write_json(hcc_root / "data" / "evidence_selection_summary_v2.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
