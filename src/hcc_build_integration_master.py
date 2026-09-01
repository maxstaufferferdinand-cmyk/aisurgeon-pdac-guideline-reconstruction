from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} line {line_no}: invalid JSON") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def unit_maps(ontology: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered: list[dict[str, Any]] = []
    by_unit: dict[str, dict[str, Any]] = {}
    by_chapter: dict[str, dict[str, Any]] = {}
    for chapter in ontology["chapters"]:
        chapter_id = clean(chapter["chapter_id"])
        by_chapter[chapter_id] = chapter
        for order, unit in enumerate(chapter["evidence_units"], 1):
            unit_id = clean(unit["unit_id"])
            record = {
                "unit_id": unit_id,
                "title": clean(unit["title"]),
                "chapter_id": chapter_id,
                "chapter_title": clean(chapter["title"]),
                "chapter_order_index": chapter.get("order_index"),
                "unit_order_index": order,
                "source_heading": clean(chapter.get("source_heading")),
                "origin": "SOURCE_DERIVED_HCC_2012",
            }
            ordered.append(record)
            by_unit[unit_id] = record
    return ordered, by_unit, by_chapter


def source_context_by_chapter(chapters: dict[str, dict[str, Any]], chronology: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    heading_to_chapter = {
        clean(ch.get("source_heading")).lower(): cid for cid, ch in chapters.items()
    }
    for item in chronology:
        heading = clean(item.get("heading_path")).split("/")[0].strip().lower()
        chapter_id = heading_to_chapter.get(heading)
        if not chapter_id:
            continue
        context[chapter_id].append(item)
    return context


def selected_metadata(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        pmid = clean(row.get("pmid"))
        if pmid and pmid not in meta:
            meta[pmid] = row
    return meta


def build_assignments(
    appraisals: list[dict[str, str]],
    selected: dict[str, dict[str, str]],
    units: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in appraisals:
        pmid = clean(row.get("pmid"))
        unit_id = clean(row.get("unit_id"))
        role = clean(row.get("role")) or "PRIMARY"
        if not pmid or not unit_id:
            rejected_rows.append({"reason": "missing_pmid_or_unit", **row})
            continue
        if unit_id not in units:
            rejected_rows.append({"reason": "unknown_unit_id", **row})
            continue
        key = (pmid, unit_id, role)
        if key in seen:
            continue
        seen.add(key)
        meta = selected.get(pmid, {})
        assignments.append(
            {
                "pmid": pmid,
                "unit_id": unit_id,
                "role": role,
                "appraisal_status": clean(row.get("appraisal_status")),
                "tier": clean(row.get("tier")),
                "study_design": clean(row.get("study_design")),
                "human_clinical_relevance": clean(row.get("human_clinical_relevance")),
                "population_directness": clean(row.get("population_directness")),
                "endpoint_strength": clean(row.get("endpoint_strength")),
                "can_support_guideline_narrative": clean(row.get("can_support_guideline_narrative")),
                "can_support_recommendation_change": clean(row.get("can_support_recommendation_change")),
                "rejection_reason": clean(row.get("rejection_reason")),
                "mapping_rationale": clean(row.get("rationale")),
                "model": clean(row.get("model")),
                "title": clean(meta.get("title")),
                "abstract": clean(meta.get("abstract")),
                "journal": clean(meta.get("journal")),
                "pub_year": clean(meta.get("pub_year")),
                "pub_date": clean(meta.get("pub_date")),
                "publication_types": clean(meta.get("publication_types")),
                "mesh_terms": clean(meta.get("mesh_terms")),
                "doi": clean(meta.get("doi")),
                "evidence_type": clean(meta.get("evidence_type")),
                "provenance_query_ids": clean(meta.get("provenance_query_ids")),
                "provenance_chapter_ids": clean(meta.get("provenance_chapter_ids")),
                "provenance_unit_ids": clean(meta.get("provenance_unit_ids")),
            }
        )
    return assignments, rejected_rows


def build_novel_topic_audit(merged_rows: list[dict[str, str]]) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in merged_rows:
        if clean(row.get("gpt_novel_topic")).lower() != "true":
            continue
        label = clean(row.get("gpt_novel_topic_label")) or "unspecified novel topic"
        unit_id = clean(row.get("gpt_primary_unit_id"))
        key = (unit_id + "::" + label).lower()
        cluster = clusters.setdefault(
            key,
            {
                "label": label,
                "primary_unit_id": unit_id,
                "pmids": [],
                "decision": "AUDIT_WITHIN_EXISTING_SOURCE_DERIVED_UNIT",
                "new_major_chapter_candidate": False,
                "rationale": (
                    "The topic was flagged as novel after 2012, but it maps to a "
                    "source-derived HCC evidence unit and does not require a new "
                    "major chapter for this reconstruction."
                ),
            },
        )
        pmid = clean(row.get("pmid"))
        if pmid and pmid not in cluster["pmids"]:
            cluster["pmids"].append(pmid)
    ordered = sorted(clusters.values(), key=lambda x: (-len(x["pmids"]), x["primary_unit_id"], x["label"].lower()))
    return {
        "created_at": utc_now(),
        "policy": {
            "original_unit_ids_frozen": True,
            "new_major_chapters_accepted": 0,
            "new_subunits_accepted": 0,
            "novel_topics_retained_as_audit_taxonomy": True,
        },
        "cluster_count": len(ordered),
        "clusters": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HCC frozen ontology and integration master.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    args = parser.parse_args()
    root = Path(args.hcc_root)
    data = root / "data"
    source = data / "source_extraction"
    map_dir = data / "gpt_mapping_appraisal_direct"

    ontology = read_json(data / "ontology_v1.json")
    ordered_units, units, chapters = unit_maps(ontology)
    chronology = read_jsonl(source / "source_chronology.jsonl")
    formal_items = read_jsonl(source / "formal_items.jsonl")
    original_refs = read_json(source / "original_references.json")
    grading = read_json(source / "grading_systems.json")
    doc_map = read_json(source / "document_map.json")
    merged = read_csv(map_dir / "selected_evidence_mapping_appraisal_merged.csv")
    appraisals = read_csv(map_dir / "pmid_unit_appraisals.csv")
    selected = selected_metadata(read_csv(data / "selected_evidence_v2.csv"))

    assignments, rejected_assignment_rows = build_assignments(appraisals, selected, units)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_unit[row["unit_id"]].append(row)

    source_context = source_context_by_chapter(chapters, chronology)
    formal_by_heading: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in formal_items:
        formal_by_heading[clean(item.get("heading_path")).split("/")[0].strip().lower()].append(item)

    frozen = {
        "created_at": utc_now(),
        "ontology_version": "hcc_2012_source_derived_v2_frozen",
        "source_ontology_version": ontology.get("ontology_version"),
        "original_unit_ids_frozen": True,
        "original_evidence_unit_count": len(ordered_units),
        "accepted_new_evidence_unit_count": 0,
        "accepted_new_major_chapter_count": 0,
        "chapters": ontology["chapters"],
        "evidence_units": ordered_units,
        "policies": {
            **ontology.get("policies", {}),
            "new_topics": "Novel labels are retained in audit taxonomy unless unmappable to source-derived HCC units.",
            "appendix_or_reject_not_recommendation_driving": True,
            "other_review_not_independently_recommendation_driving": True,
        },
    }
    write_json(data / "ontology_v2_frozen.json", frozen)
    write_json(
        data / "evidence_unit_id_crosswalk_v2.json",
        {
            "created_at": utc_now(),
            "status": "FROZEN",
            "crosswalk": [
                {"source_unit_id": u["unit_id"], "final_unit_id": u["unit_id"], "change": "UNCHANGED"}
                for u in ordered_units
            ],
        },
    )
    write_json(data / "new_topic_taxonomy_audit.json", build_novel_topic_audit(merged))
    if rejected_assignment_rows:
        write_jsonl(data / "integration_assignment_rejections.jsonl", rejected_assignment_rows)

    master: list[dict[str, Any]] = []
    empty_units: list[str] = []
    status_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for unit in ordered_units:
        evidence = sorted(
            by_unit.get(unit["unit_id"], []),
            key=lambda r: (
                0 if r["appraisal_status"] == "MAIN_SYNTHESIS" else 1 if r["appraisal_status"] == "CONTEXT_ONLY" else 2,
                0 if r["tier"] == "TIER_1" else 1 if r["tier"] == "TIER_2" else 2 if r["tier"] == "TIER_3" else 3,
                r.get("pub_year") or "9999",
                r["pmid"],
            ),
        )
        if not evidence:
            empty_units.append(unit["unit_id"])
        for row in evidence:
            status_counts[row["appraisal_status"]] += 1
            tier_counts[row["tier"]] += 1
        heading_key = unit["source_heading"].lower()
        master.append(
            {
                "chapter_id": unit["chapter_id"],
                "chapter_title": unit["chapter_title"],
                "chapter_order_index": unit["chapter_order_index"],
                "evidence_unit_id": unit["unit_id"],
                "evidence_unit_title": unit["title"],
                "unit_order_index": unit["unit_order_index"],
                "ontology_version": "hcc_2012_source_derived_v2_frozen",
                "source_context": source_context.get(unit["chapter_id"], []),
                "source_formal_items_for_chapter": formal_by_heading.get(heading_key, []),
                "source_document_map": doc_map,
                "source_grading_system": grading,
                "original_reference_range": [1, 38],
                "mapped_evidence_count": len(evidence),
                "mapped_evidence": evidence,
                "evidence_status_counts": dict(Counter(row["appraisal_status"] for row in evidence)),
                "evidence_tier_counts": dict(Counter(row["tier"] for row in evidence)),
            }
        )

    write_jsonl(data / "guideline_integration_master_v2.jsonl", master)
    manifest = {
        "created_at": utc_now(),
        "status": "READY_FOR_UNIT_SYNTHESIS",
        "final_evidence_unit_count": len(ordered_units),
        "new_subunit_count": 0,
        "new_major_chapter_count": 0,
        "selected_evidence_pmids": len({clean(r.get("pmid")) for r in merged}),
        "pmid_unit_assignment_rows": len(assignments),
        "unique_assigned_pmids": len({r["pmid"] for r in assignments}),
        "empty_final_units": empty_units,
        "empty_final_units_count": len(empty_units),
        "assignment_status_counts": dict(status_counts),
        "assignment_tier_counts": dict(tier_counts),
        "original_reference_count": len(original_refs),
        "grading_system_count": len(grading) if isinstance(grading, list) else len(grading.keys()),
        "outputs": {
            "ontology": str(data / "ontology_v2_frozen.json"),
            "crosswalk": str(data / "evidence_unit_id_crosswalk_v2.json"),
            "novel_topic_audit": str(data / "new_topic_taxonomy_audit.json"),
            "integration_master": str(data / "guideline_integration_master_v2.jsonl"),
        },
    }
    write_json(data / "guideline_integration_master_v2_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
