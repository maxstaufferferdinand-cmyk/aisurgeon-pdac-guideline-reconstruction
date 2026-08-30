#!/usr/bin/env python3
"""
STEP 2E — Freeze ontology v2 and build the canonical Integration Master.

NO OpenAI/API call.

This script performs two deterministic operations:

A) Freeze ontology v2
---------------------
- preserve all 115 original evidence units and their IDs
- accept all 69 manually reviewed STEP-2B NEW_SUBUNIT themes
- assign stable final IDs to those new units
- encode the 24 MERGE_INTO_EXISTING_UNIT directives
- retain the 22 OUT_OF_SCOPE_THEME clusters in the audit section only
- require the 9 NEW_MAJOR_CHAPTER_CANDIDATE records to have been fully resolved
- accept ZERO new major chapters

B) Build canonical final paper-to-unit evidence assignments
-----------------------------------------------------------
Combine every in-scope evidence stream into one deduplicated assignment set:
- initial evidence-unit mappings
- recovered questionable -> existing-unit mappings
- recovered novel-topic -> existing-unit mappings
- STEP-2C assignments to accepted NEW_SUBUNIT or existing units
- resolved 9 former NEW_MAJOR_CHAPTER_CANDIDATE records

Then build one Integration-Master JSON object per final evidence unit containing:
- final ontology metadata
- the exact/raw ESMO-2015 chapter baseline object
- all mapped evidence records
- mapping provenance
- placeholders for the later final clinical eligibility/appraisal run

IMPORTANT EVIDENCE POLICY FOR THE LATER SYNTHESIS
-------------------------------------------------
Ontology inclusion does NOT imply that every mapped paper is clinically eligible.

For therapeutic/interventional questions, main guideline synthesis should be
driven by HUMAN RCTs, meta-analyses and systematic reviews with clinically
meaningful patient-relevant endpoints. Papers that are surrogate-only,
preclinical/mechanistic, non-human, indirect, or insufficiently clinically
translatable may be rejected from the main synthesis and retained in a
separate appendix/context evidence stream.

Epidemiology, diagnosis, staging, prognosis, screening, risk stratification
and follow-up require domain-appropriate endpoint logic rather than an
automatic mortality/RCT requirement.

Expected inputs
---------------
data/evidence_unit_ontology.json
data/new_subunit_cluster_taxonomy_v1.json
data/new_major_chapter_candidates_resolution_manifest.json
data/new_major_chapter_candidates_resolved.jsonl
data/esmo2015_baseline_by_chapter.json

Evidence assignment inputs (if present)
---------------------------------------
data/mapped_evidence_by_evidence_unit.jsonl
data/recovered_questionables_by_evidence_unit.jsonl
data/recovery_existing_unit_assignments.jsonl
data/new_subunit_candidate_assignments_expanded.jsonl

Audit exclusion input
---------------------
data/manual_evidence_exclusions.jsonl

Outputs
-------
data/evidence_unit_ontology_v2_frozen.json
data/evidence_unit_id_crosswalk_v2.json
data/final_evidence_assignments_v2.jsonl
data/guideline_integration_master_v2.jsonl
data/guideline_integration_master_v2_manifest.json
data/ontology_v2_audit.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OLD_ONTOLOGY = DATA / "evidence_unit_ontology.json"
NEW_TAXONOMY = DATA / "new_subunit_cluster_taxonomy_v1.json"
MAJOR_MANIFEST = DATA / "new_major_chapter_candidates_resolution_manifest.json"
MAJOR_RESOLVED = DATA / "new_major_chapter_candidates_resolved.jsonl"
BASELINE = DATA / "esmo2015_baseline_by_chapter.json"
MANUAL_EXCLUSIONS = DATA / "manual_evidence_exclusions.jsonl"

INITIAL_EXISTING = DATA / "mapped_evidence_by_evidence_unit.jsonl"
RECOVERED_QUESTIONABLE_EXISTING = DATA / "recovered_questionables_by_evidence_unit.jsonl"
RECOVERY_NOVEL_EXISTING = DATA / "recovery_existing_unit_assignments.jsonl"
STEP2C_EXPANDED = DATA / "new_subunit_candidate_assignments_expanded.jsonl"

OUT_ONTOLOGY = DATA / "evidence_unit_ontology_v2_frozen.json"
OUT_CROSSWALK = DATA / "evidence_unit_id_crosswalk_v2.json"
OUT_ASSIGNMENTS = DATA / "final_evidence_assignments_v2.jsonl"
OUT_MASTER = DATA / "guideline_integration_master_v2.jsonl"
OUT_MANIFEST = DATA / "guideline_integration_master_v2_manifest.json"
OUT_AUDIT = DATA / "ontology_v2_audit.json"

CHAPTER_ORDER = ["1", "2", "3", "4.1", "4.2", "4.3", "5", "6"]


def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
                    f"{path.name} line {line_no}: {e}"
                ) from e
    return rows


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


def manual_excluded_pmids() -> set[str]:
    return {
        clean(r.get("pmid"))
        for r in load_jsonl(MANUAL_EXCLUSIONS, optional=True)
        if clean(r.get("pmid"))
    }


def normalize_old_ontology(old: dict[str, Any]) -> dict[str, Any]:
    if "chapters" not in old or not isinstance(old["chapters"], dict):
        raise RuntimeError(
            "Expected evidence_unit_ontology.json to contain a dict field 'chapters'."
        )
    return old


def validate_major_resolution() -> list[dict[str, Any]]:
    manifest = load_json(MAJOR_MANIFEST)
    resolved = load_jsonl(MAJOR_RESOLVED)

    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            f"Major-chapter resolution is not COMPLETE: {manifest}"
        )
    if int(manifest.get("accepted_new_major_chapters", -1)) != 0:
        raise RuntimeError(
            "Ontology freeze requires accepted_new_major_chapters = 0."
        )
    if int(manifest.get("resolved_candidates", -1)) != 9:
        raise RuntimeError(
            f"Expected 9 resolved major candidates, got "
            f"{manifest.get('resolved_candidates')}"
        )
    if len(resolved) != 9:
        raise RuntimeError(
            f"Expected 9 rows in {MAJOR_RESOLVED.name}, found {len(resolved)}"
        )

    for row in resolved:
        if clean(row.get("major_chapter_resolution")) != "NO_NEW_MAJOR_CHAPTER":
            raise RuntimeError(
                f"Unexpected major chapter resolution for PMID {row.get('pmid')}: "
                f"{row.get('major_chapter_resolution')}"
            )
    return resolved


def freeze_ontology():
    old = normalize_old_ontology(load_json(OLD_ONTOLOGY))
    new = load_json(NEW_TAXONOMY)

    if new.get("status") != "COMPLETE":
        raise RuntimeError(
            f"new_subunit_cluster_taxonomy_v1.json is not COMPLETE."
        )

    # Preserve original chapter order and original evidence-unit IDs.
    chapters = {}
    existing_ids = set()
    original_unit_count = 0

    for cid in CHAPTER_ORDER:
        if cid not in old["chapters"]:
            raise RuntimeError(f"Missing original ontology chapter {cid}")
        chapter = deepcopy(old["chapters"][cid])

        frozen_units = []
        for unit in chapter["evidence_units"]:
            uid = clean(unit.get("id"))
            if not uid:
                raise RuntimeError(f"Original unit without id in chapter {cid}")
            if uid in existing_ids:
                raise RuntimeError(f"Duplicate original unit ID: {uid}")
            existing_ids.add(uid)
            original_unit_count += 1

            frozen = deepcopy(unit)
            frozen["final_unit_id"] = uid
            frozen["unit_origin"] = "ORIGINAL_ONTOLOGY_V1"
            frozen["ontology_status"] = "FROZEN"
            frozen_units.append(frozen)

        chapter["evidence_units"] = frozen_units
        chapters[cid] = chapter

    # STEP 2B new taxonomy indexed by chapter.
    new_by_chapter = {
        clean(ch["chapter_id"]): ch
        for ch in new["chapters"]
    }

    crosswalk = []
    merge_directives = []
    out_of_scope_themes = []
    accepted_new_count = 0

    for cid in CHAPTER_ORDER:
        chapter_tax = new_by_chapter.get(cid)
        if not chapter_tax:
            raise RuntimeError(f"Missing STEP 2B taxonomy chapter {cid}")

        accepted = [
            c for c in chapter_tax["proposed_clusters"]
            if clean(c.get("disposition")) == "NEW_SUBUNIT"
        ]

        # Stable IDs: preserve all old IDs; append chapter-local N-series.
        # Once this v2 file is frozen, these IDs must not be regenerated or renumbered.
        accepted_sorted = sorted(
            accepted,
            key=lambda c: (
                clean(c.get("title")).casefold(),
                clean(c.get("temporary_cluster_id")),
            ),
        )

        for index, c in enumerate(accepted_sorted, 1):
            final_id = f"{cid}.N{index:02d}"
            if final_id in existing_ids:
                raise RuntimeError(f"Generated new unit ID collision: {final_id}")
            existing_ids.add(final_id)
            accepted_new_count += 1

            unit = {
                "id": final_id,
                "final_unit_id": final_id,
                "name": clean(c.get("title")),
                "definition": clean(c.get("definition")),
                "boundary": clean(c.get("boundary")),
                "unit_origin": "NEW_SUBUNIT_STEP2B_ACCEPTED",
                "ontology_status": "FROZEN",
                "source_temporary_cluster_id": clean(c.get("temporary_cluster_id")),
                "source_candidate_scope_summary": clean(
                    c.get("candidate_scope_summary")
                ),
                "source_rationale": clean(c.get("rationale")),
                "accepted_after_manual_review": True,
            }
            chapters[cid]["evidence_units"].append(unit)

            crosswalk.append({
                "chapter_id": cid,
                "source_temporary_cluster_id": clean(c.get("temporary_cluster_id")),
                "disposition": "NEW_SUBUNIT",
                "final_unit_id": final_id,
                "existing_unit_id": "",
                "title": clean(c.get("title")),
            })

        for c in chapter_tax["proposed_clusters"]:
            disposition = clean(c.get("disposition"))
            temp_id = clean(c.get("temporary_cluster_id"))

            if disposition == "MERGE_INTO_EXISTING_UNIT":
                target = clean(c.get("existing_unit_id"))
                if target not in existing_ids:
                    # It should be an original unit, not a new N-series unit.
                    raise RuntimeError(
                        f"MERGE target {target!r} not found for cluster {temp_id}"
                    )

                merge_directives.append({
                    "chapter_id": cid,
                    "source_temporary_cluster_id": temp_id,
                    "target_existing_unit_id": target,
                    "title": clean(c.get("title")),
                    "definition": clean(c.get("definition")),
                    "boundary": clean(c.get("boundary")),
                    "rationale": clean(c.get("rationale")),
                })
                crosswalk.append({
                    "chapter_id": cid,
                    "source_temporary_cluster_id": temp_id,
                    "disposition": disposition,
                    "final_unit_id": target,
                    "existing_unit_id": target,
                    "title": clean(c.get("title")),
                })

            elif disposition == "OUT_OF_SCOPE_THEME":
                out_of_scope_themes.append({
                    "chapter_id": cid,
                    "source_temporary_cluster_id": temp_id,
                    "title": clean(c.get("title")),
                    "definition": clean(c.get("definition")),
                    "boundary": clean(c.get("boundary")),
                    "rationale": clean(c.get("rationale")),
                })
                crosswalk.append({
                    "chapter_id": cid,
                    "source_temporary_cluster_id": temp_id,
                    "disposition": disposition,
                    "final_unit_id": "",
                    "existing_unit_id": "",
                    "title": clean(c.get("title")),
                })

    if original_unit_count != 115:
        raise RuntimeError(
            f"Expected 115 original units, found {original_unit_count}. "
            "Refusing to freeze a silently changed v1 ontology."
        )
    if accepted_new_count != 69:
        raise RuntimeError(
            f"Expected 69 accepted NEW_SUBUNIT themes, found {accepted_new_count}."
        )
    if len(merge_directives) != 24:
        raise RuntimeError(
            f"Expected 24 MERGE directives, found {len(merge_directives)}."
        )
    if len(out_of_scope_themes) != 22:
        raise RuntimeError(
            f"Expected 22 OUT_OF_SCOPE themes, found {len(out_of_scope_themes)}."
        )

    frozen = {
        "ontology_name": "ESMO_PDAC_2015_to_2023_Evidence_Ontology",
        "ontology_version": "v2_frozen",
        "status": "FROZEN",
        "major_chapters": CHAPTER_ORDER,
        "accepted_new_major_chapters": 0,
        "original_evidence_units": original_unit_count,
        "accepted_new_evidence_units": accepted_new_count,
        "final_evidence_units_total": original_unit_count + accepted_new_count,
        "chapters": chapters,
        "evidence_policy": {
            "ontology_inclusion_is_not_clinical_evidence_acceptance": True,
            "therapeutic_main_synthesis": (
                "Prioritize/include human RCTs, meta-analyses and systematic reviews "
                "with clinically meaningful patient-relevant endpoints."
            ),
            "final_paper_rejection_allowed": True,
            "appendix_for_non_translatable_or_context_evidence": True,
            "domain_specific_exception": (
                "Epidemiology, diagnosis, staging, prognosis, screening, risk "
                "stratification and follow-up use domain-appropriate clinical "
                "endpoint logic rather than an automatic therapeutic-RCT rule."
            ),
        },
    }

    audit = {
        "status": "FROZEN",
        "original_units": original_unit_count,
        "accepted_new_units": accepted_new_count,
        "final_units": original_unit_count + accepted_new_count,
        "merge_directives": merge_directives,
        "out_of_scope_themes": out_of_scope_themes,
        "accepted_new_major_chapters": 0,
        "new_major_chapter_candidates_resolved": 9,
        "id_policy": (
            "Original unit IDs are immutable. New v2 units receive chapter-local "
            "stable N-series IDs (e.g. 4.3.N01). Do not renumber after freeze."
        ),
    }

    OUT_ONTOLOGY.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_CROSSWALK.write_text(
        json.dumps(crosswalk, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return frozen, crosswalk, audit


def baseline_by_chapter(raw: Any) -> dict[str, Any]:
    """
    Preserve the ESMO baseline chapter object verbatim.
    Support common baseline JSON shapes without rewriting its content.
    """
    if isinstance(raw, dict):
        # Shape A: {"1": {...}, "2": {...}, ...}
        if all(cid in raw for cid in CHAPTER_ORDER):
            return {cid: raw[cid] for cid in CHAPTER_ORDER}

        # Shape B: {"chapters": {"1": {...}}}
        chapters = raw.get("chapters")
        if isinstance(chapters, dict) and all(cid in chapters for cid in CHAPTER_ORDER):
            return {cid: chapters[cid] for cid in CHAPTER_ORDER}

        # Shape C: {"chapters": [{"chapter_id":"1", ...}, ...]}
        if isinstance(chapters, list):
            mapped = {
                clean(x.get("chapter_id") or x.get("id")): x
                for x in chapters
                if isinstance(x, dict)
            }
            if all(cid in mapped for cid in CHAPTER_ORDER):
                return {cid: mapped[cid] for cid in CHAPTER_ORDER}

    raise RuntimeError(
        "Could not recognize esmo2015_baseline_by_chapter.json structure. "
        "No baseline content was modified; adjust only baseline_by_chapter() if needed."
    )


UNIT_FIELD_CANDIDATES = [
    "evidence_unit_id",
    "unit_id",
    "assigned_unit_id",
    "assigned_evidence_unit_id",
    "subunit_id",
    "existing_unit_id",
    "primary_unit_id",
]

ROLE_FIELD_CANDIDATES = [
    "assignment_role",
    "mapping_role",
    "role",
]

CONF_FIELD_CANDIDATES = [
    "assignment_confidence",
    "mapping_confidence",
    "submapping_confidence",
    "confidence",
]


def first_value(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = clean(row.get(field))
        if value:
            return value
    return ""


def scientific_record(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize paper metadata while preserving all upstream information separately.
    """
    return {
        "pmid": clean(row.get("pmid")),
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


def build_final_unit_maps(frozen: dict[str, Any], crosswalk: list[dict[str, Any]]):
    unit_meta = {}
    original_ids = set()

    for cid, chapter in frozen["chapters"].items():
        for unit in chapter["evidence_units"]:
            uid = clean(unit["final_unit_id"])
            unit_meta[uid] = {
                "chapter_id": cid,
                "chapter_title": clean(chapter.get("title")),
                "unit_id": uid,
                "unit_name": clean(unit.get("name")),
                "unit_definition": clean(unit.get("definition")),
                "unit_boundary": clean(unit.get("boundary")),
                "unit_origin": clean(unit.get("unit_origin")),
            }
            if unit["unit_origin"] == "ORIGINAL_ONTOLOGY_V1":
                original_ids.add(uid)

    temp_to_final = {}
    oos_temp_ids = set()
    for row in crosswalk:
        temp = clean(row.get("source_temporary_cluster_id"))
        disposition = clean(row.get("disposition"))
        final_id = clean(row.get("final_unit_id"))
        if disposition in {"NEW_SUBUNIT", "MERGE_INTO_EXISTING_UNIT"}:
            temp_to_final[temp] = final_id
        elif disposition == "OUT_OF_SCOPE_THEME":
            oos_temp_ids.add(temp)

    return unit_meta, original_ids, temp_to_final, oos_temp_ids


def normalize_direct_existing_stream(
    path: Path,
    provenance: str,
    unit_meta: dict[str, Any],
    excluded_pmids: set[str],
) -> list[dict[str, Any]]:
    rows = load_jsonl(path, optional=True)
    out = []

    for index, row in enumerate(rows, 1):
        pmid = clean(row.get("pmid"))
        if not pmid:
            raise RuntimeError(f"{path.name} row {index} missing PMID")
        if pmid in excluded_pmids:
            continue

        unit_id = first_value(row, UNIT_FIELD_CANDIDATES)
        if not unit_id:
            # Some upstream files may use selected_unit_ids or unit_ids arrays.
            array_value = (
                row.get("selected_unit_ids")
                or row.get("unit_ids")
                or row.get("evidence_unit_ids")
                or row.get("existing_unit_ids")
            )
            if isinstance(array_value, list) and array_value:
                for pos, uid in enumerate(array_value):
                    uid = clean(uid)
                    if uid not in unit_meta:
                        raise RuntimeError(
                            f"{path.name} row {index}: unknown unit ID {uid!r}"
                        )
                    out.append({
                        **scientific_record(row),
                        "final_unit_id": uid,
                        "chapter_id": unit_meta[uid]["chapter_id"],
                        "mapping_role": "primary" if pos == 0 else "secondary",
                        "mapping_confidence": first_value(row, CONF_FIELD_CANDIDATES),
                        "provenance": provenance,
                        "source_file": path.name,
                        "source_row": index,
                        "upstream_record": row,
                    })
                continue

            raise RuntimeError(
                f"{path.name} row {index}: could not find an evidence-unit ID. "
                f"Checked scalar fields={UNIT_FIELD_CANDIDATES} and array fields="
                f"['selected_unit_ids','unit_ids','evidence_unit_ids','existing_unit_ids']. "
                f"Available keys: {sorted(row.keys())}"
            )

        if unit_id not in unit_meta:
            raise RuntimeError(
                f"{path.name} row {index}: unknown final/original unit ID "
                f"{unit_id!r} for PMID {pmid}"
            )

        role = first_value(row, ROLE_FIELD_CANDIDATES).lower() or "primary"
        if role not in {"primary", "secondary"}:
            role = "primary"

        out.append({
            **scientific_record(row),
            "final_unit_id": unit_id,
            "chapter_id": unit_meta[unit_id]["chapter_id"],
            "mapping_role": role,
            "mapping_confidence": first_value(row, CONF_FIELD_CANDIDATES),
            "provenance": provenance,
            "source_file": path.name,
            "source_row": index,
            "upstream_record": row,
        })

    return out


def normalize_step2c(
    unit_meta: dict[str, Any],
    temp_to_final: dict[str, str],
    oos_temp_ids: set[str],
    excluded_pmids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(STEP2C_EXPANDED, optional=True)
    out = []
    oos = []

    for index, row in enumerate(rows, 1):
        pmid = clean(row.get("pmid"))
        if not pmid:
            raise RuntimeError(f"{STEP2C_EXPANDED.name} row {index} missing PMID")
        if pmid in excluded_pmids:
            continue

        disposition = clean(row.get("assigned_cluster_disposition"))
        cluster_id = clean(row.get("assigned_cluster_id"))
        existing_target = clean(row.get("assigned_existing_unit_id"))

        if disposition == "OUT_OF_SCOPE_THEME":
            oos.append(row)
            continue

        if disposition == "NEW_SUBUNIT":
            final_unit_id = temp_to_final.get(cluster_id, "")
        elif disposition == "MERGE_INTO_EXISTING_UNIT":
            final_unit_id = existing_target or temp_to_final.get(cluster_id, "")
        else:
            raise RuntimeError(
                f"{STEP2C_EXPANDED.name} row {index}: unexpected disposition "
                f"{disposition!r}"
            )

        if not final_unit_id or final_unit_id not in unit_meta:
            raise RuntimeError(
                f"{STEP2C_EXPANDED.name} row {index}: cannot resolve "
                f"cluster {cluster_id} disposition {disposition} to final unit."
            )

        role = clean(row.get("assignment_role")).lower() or "primary"
        if role not in {"primary", "secondary"}:
            role = "primary"

        out.append({
            **scientific_record(row),
            "final_unit_id": final_unit_id,
            "chapter_id": unit_meta[final_unit_id]["chapter_id"],
            "mapping_role": role,
            "mapping_confidence": clean(row.get("assignment_confidence")),
            "provenance": "STEP2C_NEW_TOPIC_ASSIGNMENT",
            "source_file": STEP2C_EXPANDED.name,
            "source_row": index,
            "source_cluster_id": cluster_id,
            "source_cluster_disposition": disposition,
            "upstream_record": row,
        })

    return out, oos


def normalize_major_resolved(
    resolved: list[dict[str, Any]],
    unit_meta: dict[str, Any],
    temp_to_final: dict[str, str],
    excluded_pmids: set[str],
) -> list[dict[str, Any]]:
    out = []

    for index, row in enumerate(resolved, 1):
        pmid = clean(row.get("pmid"))
        if pmid in excluded_pmids:
            continue

        assignments = [{
            "unit_key": clean(row.get("primary_unit_key")),
            "unit_type": clean(row.get("primary_unit_type")),
            "chapter_id": clean(row.get("primary_chapter_id")),
            "role": "primary",
        }]

        for s in row.get("secondary_assignments") or []:
            assignments.append({
                "unit_key": clean(s.get("unit_key")),
                "unit_type": clean(s.get("unit_type")),
                "chapter_id": clean(s.get("chapter_id")),
                "role": "secondary",
            })

        for a in assignments:
            source_key = a["unit_key"]

            if a["unit_type"] == "NEW_SUBUNIT":
                final_unit_id = temp_to_final.get(source_key, "")
            else:
                final_unit_id = source_key

            if final_unit_id not in unit_meta:
                raise RuntimeError(
                    f"{MAJOR_RESOLVED.name} row {index}: cannot resolve "
                    f"{source_key!r} ({a['unit_type']}) to final ontology."
                )

            out.append({
                **scientific_record(row),
                "final_unit_id": final_unit_id,
                "chapter_id": unit_meta[final_unit_id]["chapter_id"],
                "mapping_role": a["role"],
                "mapping_confidence": clean(row.get("confidence")),
                "provenance": "RESOLVED_NEW_MAJOR_CHAPTER_CANDIDATE",
                "source_file": MAJOR_RESOLVED.name,
                "source_row": index,
                "source_unit_key": source_key,
                "upstream_record": row,
            })

    return out


def role_rank(role: str) -> int:
    return {"primary": 2, "secondary": 1}.get(role, 0)


def confidence_rank(conf: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(conf.lower(), 0)


def deduplicate_assignments(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (clean(row["pmid"]), clean(row["final_unit_id"]))
        grouped[key].append(row)

    deduped = []

    for (pmid, unit_id), items in grouped.items():
        best = sorted(
            items,
            key=lambda r: (
                role_rank(clean(r.get("mapping_role")).lower()),
                confidence_rank(clean(r.get("mapping_confidence"))),
                len(clean(r.get("abstract"))),
            ),
            reverse=True,
        )[0]

        # Preserve richest scientific metadata.
        merged = deepcopy(best)
        for field in [
            "doi", "pmcid", "title", "abstract", "authors", "journal",
            "publication_date", "publication_year", "publication_types",
            "evidence_labels", "mesh_terms", "keywords",
        ]:
            richest = max(
                (clean(x.get(field)) for x in items),
                key=len,
                default="",
            )
            merged[field] = richest

        merged["mapping_role"] = (
            "primary"
            if any(clean(x.get("mapping_role")).lower() == "primary" for x in items)
            else "secondary"
        )

        confidences = [clean(x.get("mapping_confidence")).lower() for x in items]
        merged["mapping_confidence"] = max(
            confidences,
            key=confidence_rank,
            default="",
        )

        merged["provenance"] = sorted({
            clean(x.get("provenance")) for x in items if clean(x.get("provenance"))
        })
        merged["source_records"] = [
            {
                "source_file": clean(x.get("source_file")),
                "source_row": x.get("source_row"),
                "provenance": clean(x.get("provenance")),
            }
            for x in items
        ]

        # The later final GPT run fills these fields.
        merged["final_evidence_appraisal"] = {
            "status": None,
            "clinical_domain": None,
            "human_evidence": None,
            "study_design_eligible": None,
            "clinically_meaningful_endpoint": None,
            "surrogate_only": None,
            "clinical_translatability": None,
            "can_support_guideline_text": None,
            "can_support_recommendation_change": None,
            "rejection_reason": None,
            "appendix_eligible": None,
        }

        # Upstream full records are not needed in the canonical assignment file.
        merged.pop("upstream_record", None)
        deduped.append(merged)

    deduped.sort(
        key=lambda r: (
            chapter_sort_key(clean(r.get("chapter_id"))),
            clean(r.get("final_unit_id")),
            int(clean(r.get("pmid"))) if clean(r.get("pmid")).isdigit() else clean(r.get("pmid")),
        )
    )
    return deduped


def evidence_policy_for_chapter(cid: str) -> dict[str, Any]:
    if cid in {"4.1", "4.2", "4.3"}:
        return {
            "domain": "THERAPEUTIC_INTERVENTIONAL",
            "main_synthesis_rule": (
                "Use human RCTs, meta-analyses and systematic reviews with "
                "clinically meaningful patient-relevant endpoints as the main "
                "recommendation-driving evidence."
            ),
            "surrogate_only_policy": (
                "Do not use surrogate-only evidence to drive a clinical "
                "recommendation change unless a compelling validated link to "
                "patient-relevant benefit is explicitly established."
            ),
        }
    if cid == "1":
        return {
            "domain": "EPIDEMIOLOGY_RISK_SCREENING",
            "main_synthesis_rule": (
                "Use study-design and endpoint criteria appropriate to incidence, "
                "risk, screening and prevention questions; clinical relevance "
                "remains mandatory."
            ),
            "surrogate_only_policy": (
                "Mechanistic or non-human risk evidence may be context/appendix "
                "only unless clinically validated."
            ),
        }
    if cid == "2":
        return {
            "domain": "DIAGNOSIS_PATHOLOGY_BIOLOGY",
            "main_synthesis_rule": (
                "For diagnostic questions, prioritize clinically validated "
                "diagnostic accuracy/management-impact evidence and applicable "
                "systematic reviews/meta-analyses. Mechanistic/pathobiology evidence "
                "requires demonstrated clinical translation to influence guideline text."
            ),
            "surrogate_only_policy": (
                "Experimental molecular or model-system findings without clinical "
                "validation belong in context/appendix."
            ),
        }
    if cid == "3":
        return {
            "domain": "STAGING_RESECTABILITY_PROGNOSIS",
            "main_synthesis_rule": (
                "Use clinically validated staging, prognostic and risk-stratification "
                "evidence with patient-level clinical endpoints; treatment-change "
                "claims require stronger interventional evidence."
            ),
            "surrogate_only_policy": (
                "Unvalidated biomarker/surrogate associations are context/appendix only."
            ),
        }
    if cid == "5":
        return {
            "domain": "PERSONALISED_MEDICINE",
            "main_synthesis_rule": (
                "A biomarker or targeted strategy must demonstrate clinical "
                "validation and translatability. Predictive claims influencing "
                "therapy should preferentially be supported by human RCTs and/or "
                "high-quality systematic/meta-analytic clinical evidence."
            ),
            "surrogate_only_policy": (
                "Mechanistic, preclinical, model-system or surrogate-only precision "
                "medicine evidence must not drive a clinical recommendation."
            ),
        }
    if cid == "6":
        return {
            "domain": "FOLLOW_UP_SURVIVORSHIP",
            "main_synthesis_rule": (
                "Use clinically meaningful follow-up, recurrence, survivorship, "
                "quality-of-life and long-term outcome evidence with domain-appropriate "
                "study designs."
            ),
            "surrogate_only_policy": (
                "Biomarker-only surveillance evidence requires clinical validation "
                "before it can change follow-up recommendations."
            ),
        }
    raise RuntimeError(f"No evidence policy for chapter {cid}")


def main() -> None:
    resolved_major = validate_major_resolution()
    frozen, crosswalk, audit = freeze_ontology()

    baseline_raw = load_json(BASELINE)
    baseline_map = baseline_by_chapter(baseline_raw)

    excluded_pmids = manual_excluded_pmids()
    unit_meta, original_ids, temp_to_final, oos_temp_ids = build_final_unit_maps(
        frozen, crosswalk
    )

    all_assignments = []

    streams = [
        (
            INITIAL_EXISTING,
            "INITIAL_EXISTING_UNIT_SUBMAPPING",
        ),
        (
            RECOVERED_QUESTIONABLE_EXISTING,
            "RECOVERED_QUESTIONABLE_EXISTING_UNIT",
        ),
        (
            RECOVERY_NOVEL_EXISTING,
            "RECOVERED_NOVEL_EXISTING_UNIT",
        ),
    ]

    stream_counts = {}
    for path, provenance in streams:
        normalized = normalize_direct_existing_stream(
            path,
            provenance,
            unit_meta,
            excluded_pmids,
        )
        stream_counts[path.name] = len(normalized)
        all_assignments.extend(normalized)

    step2c, step2c_oos = normalize_step2c(
        unit_meta,
        temp_to_final,
        oos_temp_ids,
        excluded_pmids,
    )
    stream_counts[STEP2C_EXPANDED.name] = len(step2c)
    all_assignments.extend(step2c)

    major_rows = normalize_major_resolved(
        resolved_major,
        unit_meta,
        temp_to_final,
        excluded_pmids,
    )
    stream_counts[MAJOR_RESOLVED.name] = len(major_rows)
    all_assignments.extend(major_rows)

    deduped = deduplicate_assignments(all_assignments)
    write_jsonl(OUT_ASSIGNMENTS, deduped)

    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        by_unit[row["final_unit_id"]].append(row)

    master = []
    chapter_unit_counts = Counter()
    chapter_paper_assignment_counts = Counter()
    empty_units = []

    for cid in CHAPTER_ORDER:
        chapter = frozen["chapters"][cid]

        for unit in chapter["evidence_units"]:
            uid = clean(unit["final_unit_id"])
            evidence = by_unit.get(uid, [])

            # Sort evidence with stronger publication-type labels first only for
            # convenience; this does NOT constitute final clinical appraisal.
            evidence = sorted(
                evidence,
                key=lambda r: (
                    0 if "META" in clean(r.get("evidence_labels")).upper() else
                    1 if "SYSTEMATIC" in clean(r.get("evidence_labels")).upper() else
                    2 if "RCT" in clean(r.get("evidence_labels")).upper() else
                    3,
                    clean(r.get("publication_year")),
                    clean(r.get("pmid")),
                )
            )

            if not evidence:
                empty_units.append(uid)

            master_obj = {
                "chapter_id": cid,
                "chapter_title": clean(chapter.get("title")),
                "evidence_unit_id": uid,
                "evidence_unit_name": clean(unit.get("name")),
                "evidence_unit_definition": clean(unit.get("definition")),
                "evidence_unit_boundary": clean(unit.get("boundary")),
                "evidence_unit_origin": clean(unit.get("unit_origin")),
                "ontology_version": "v2_frozen",
                "original_esmo2015_chapter_context": baseline_map[cid],
                "original_unit_specific_text_status": (
                    "NOT_YET_SEPARATELY_EXTRACTED_FROM_CHAPTER_BASELINE"
                ),
                "final_evidence_policy": evidence_policy_for_chapter(cid),
                "mapped_evidence_count": len(evidence),
                "mapped_evidence": evidence,
                "final_synthesis_placeholders": {
                    "unit_evidence_appraisal_status": None,
                    "eligible_main_synthesis_pmids": [],
                    "context_only_pmids": [],
                    "rejected_pmids": [],
                    "appendix_pmids": [],
                    "change_decision": None,
                    "updated_guideline_text": None,
                    "recommendation_change_supported": None,
                    "synthesis_rationale": None,
                },
            }

            master.append(master_obj)
            chapter_unit_counts[cid] += 1
            chapter_paper_assignment_counts[cid] += len(evidence)

    write_jsonl(OUT_MASTER, master)

    unique_pmids = {r["pmid"] for r in deduped}
    provenance_counts = Counter()
    role_counts = Counter()
    for r in deduped:
        for p in r["provenance"]:
            provenance_counts[p] += 1
        role_counts[clean(r.get("mapping_role"))] += 1

    manifest = {
        "status": "READY_FOR_FINAL_EVIDENCE_APPRAISAL",
        "ontology_status": "FROZEN",
        "ontology_version": "v2_frozen",
        "accepted_new_major_chapters": 0,
        "final_evidence_units_total": frozen["final_evidence_units_total"],
        "original_evidence_units": frozen["original_evidence_units"],
        "accepted_new_evidence_units": frozen["accepted_new_evidence_units"],
        "combined_source_assignment_rows_before_deduplication": len(all_assignments),
        "final_unique_pmid_unit_assignments": len(deduped),
        "final_unique_pmids": len(unique_pmids),
        "manual_excluded_pmids_count": len(excluded_pmids),
        "step2c_out_of_scope_assignment_rows_not_integrated": len(step2c_oos),
        "empty_final_units": empty_units,
        "empty_final_units_count": len(empty_units),
        "chapter_unit_counts": dict(chapter_unit_counts),
        "chapter_paper_assignment_counts": dict(chapter_paper_assignment_counts),
        "input_stream_assignment_counts": stream_counts,
        "provenance_counts_after_deduplication": dict(provenance_counts),
        "mapping_role_counts": dict(role_counts),
        "baseline_policy": (
            "The exact/raw ESMO-2015 chapter baseline object is embedded in each "
            "unit record. Unit-specific old-text extraction has not yet been performed."
        ),
        "evidence_policy": frozen["evidence_policy"],
        "next_step": (
            "Final Stage A evidence appraisal. For each final evidence unit, review "
            "mapped abstracts and classify each paper as main-synthesis eligible, "
            "context-only, or rejected/appendix. For high-volume units, technical "
            "chunking is allowed without changing the medical evidence-unit boundary."
        ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("STEP 2E completed — ontology v2 frozen + Integration Master built.")
    print()
    print("ONTOLOGY")
    print(f"  original evidence units:           {frozen['original_evidence_units']:,}")
    print(f"  accepted new evidence units:       {frozen['accepted_new_evidence_units']:,}")
    print(f"  final evidence units:              {frozen['final_evidence_units_total']:,}")
    print("  accepted new major chapters:       0")
    print("  ontology status:                   FROZEN")
    print()
    print("EVIDENCE MASTER")
    print(f"  source assignment rows:            {len(all_assignments):,}")
    print(f"  unique PMID+unit assignments:      {len(deduped):,}")
    print(f"  unique PMIDs:                      {len(unique_pmids):,}")
    print(f"  final units with zero evidence:    {len(empty_units):,}")
    print(f"  STEP2C OOS rows excluded:          {len(step2c_oos):,}")
    print()
    print("Per chapter:")
    for cid in CHAPTER_ORDER:
        print(
            f"  chapter {cid:>3}: "
            f"{chapter_unit_counts[cid]:>3} units | "
            f"{chapter_paper_assignment_counts[cid]:>5} PMID-unit assignments"
        )
    print()
    print(f"  frozen ontology:     {OUT_ONTOLOGY}")
    print(f"  ID crosswalk:        {OUT_CROSSWALK}")
    print(f"  final assignments:   {OUT_ASSIGNMENTS}")
    print(f"  integration master:  {OUT_MASTER}")
    print(f"  manifest:            {OUT_MANIFEST}")
    print(f"  ontology audit:      {OUT_AUDIT}")
    print()
    print("No OpenAI/API call was made.")
    print()
    print(
        "NEXT: run final Stage A evidence appraisal. Do not start Stage B guideline "
        "rewriting until Stage A clinical eligibility/appraisal is complete."
    )


if __name__ == "__main__":
    main()
