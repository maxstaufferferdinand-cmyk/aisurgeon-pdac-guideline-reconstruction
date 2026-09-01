from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
SOURCE_SHA256 = "b65f49ba27e4640cb63976476818c79e97fc78cf111333f1ceed17e66e4b8482"
DOCX_NAME = "ESMO_HCC_2012_Living_Evidence_Update_2025-02-28_v1.docx"


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return f"ERROR: {exc.output.strip()}"


def docx_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            return "word/document.xml" in names and "[Content_Types].xml" in names
    except zipfile.BadZipFile:
        return False


def usage_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    phase_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for entry in ledger.get("requests", []):
        phase_counts[clean(entry.get("phase"))] += 1
        model_counts[clean(entry.get("model"))] += 1
        usage = entry.get("usage") or {}
        for key in [
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ]:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
        details = usage.get("input_tokens_details") or {}
        if isinstance(details.get("cached_tokens"), int):
            totals["cached_input_tokens"] += details["cached_tokens"]
        output_details = usage.get("output_tokens_details") or {}
        if isinstance(output_details.get("reasoning_tokens"), int):
            totals["reasoning_tokens"] += output_details["reasoning_tokens"]
    return {
        "request_count_by_phase": dict(phase_counts),
        "request_count_by_model": dict(model_counts),
        "token_usage_observed": dict(totals),
        "monetary_cost_accounting": "DISABLED_PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Final HCC reconstruction QC and report.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    args = parser.parse_args()
    root = Path(args.hcc_root)
    data = root / "data"
    source_pdf = root / "ESMOHCC2012.pdf"

    preflight = read_json(root / "audit" / "source_pdf_preflight.json", {})
    protocol = read_json(root / "config" / "protocol_lock.json", {})
    selection = read_json(data / "evidence_selection_summary_v2.json", {})
    retrieval = read_json(data / "pubmed_retrieval_summary_v2.json", {})
    mapping_qc = read_json(data / "gpt_mapping_appraisal_direct" / "direct_mapping_appraisal_merge_qc.json", {})
    integration = read_json(data / "guideline_integration_master_v2_manifest.json", {})
    stagea = read_json(data / "stageA_evidence_synthesis_manifest.json", {})
    stageb = read_json(data / "stageB_docx_manifest.json", {})
    ledger = read_json(root / "run_state" / "cost_ledger.json", {"requests": []})
    appraisals = read_csv(data / "gpt_mapping_appraisal_direct" / "pmid_unit_appraisals.csv")
    syntheses = read_jsonl(data / "stageA_unit_evidence_synthesis.jsonl")

    appraisal_counts = Counter(row.get("appraisal_status") for row in appraisals)
    docx_path = root / "output" / DOCX_NAME
    appendix_path = root / "output" / "ESMO_HCC_2012_Living_Evidence_Update_2025-02-28_v1_APPENDIX.docx"
    failures = []
    if sha256(source_pdf) != SOURCE_SHA256:
        failures.append("source_pdf_sha256_mismatch")
    if mapping_qc.get("missing_pmids") != 0:
        failures.append("mapping_appraisal_missing_pmids")
    if integration.get("status") != "READY_FOR_UNIT_SYNTHESIS":
        failures.append("integration_master_not_ready")
    if stagea.get("status") != "COMPLETE":
        failures.append("stageA_synthesis_incomplete")
    if not docx_ok(docx_path):
        failures.append("final_docx_invalid_or_missing")
    if not docx_ok(appendix_path):
        failures.append("appendix_docx_invalid_or_missing")
    if protocol.get("dates", {}).get("search_start") != "2012-07-01":
        failures.append("search_start_lock_mismatch")
    if protocol.get("dates", {}).get("search_end") != "2025-02-28":
        failures.append("search_end_lock_mismatch")
    if protocol.get("cost_control", {}).get("mode") != "PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION":
        failures.append("cost_control_override_not_recorded")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source_pdf_sha256": sha256(source_pdf),
        "source_update_date": "June 2012",
        "search_start": "2012-07-01",
        "search_end": "2025-02-28",
        "gemini_model": "models/gemini-3.5-flash",
        "openai_model": "gpt-5.6-sol",
        "paid_api_usage": usage_summary(ledger),
        "raw_pubmed_row_count": retrieval.get("unique_records_written") or selection.get("raw_pubmed_row_count"),
        "unique_pmid_count": selection.get("unique_pmid_count"),
        "rct_count": selection.get("selected_evidence_type_counts", {}).get("RANDOMIZED_CONTROLLED_TRIAL"),
        "meta_analysis_count": selection.get("selected_evidence_type_counts", {}).get("META_ANALYSIS"),
        "systematic_review_count": selection.get("selected_evidence_type_counts", {}).get("SYSTEMATIC_REVIEW"),
        "other_review_count": selection.get("selected_evidence_type_counts", {}).get("OTHER_REVIEW"),
        "excluded_guideline_consensus_count": selection.get("excluded_guideline_consensus_count"),
        "ambiguous_guidance_count": selection.get("ambiguous_guidance_count"),
        "mapped_pmid_count": mapping_qc.get("parsed_pmids"),
        "unmappable_pmid_count": mapping_qc.get("missing_pmids"),
        "final_evidence_unit_count": integration.get("final_evidence_unit_count"),
        "new_subunit_count": integration.get("new_subunit_count"),
        "evidence_appraisal_counts": dict(appraisal_counts),
        "stageA_unit_count": len(syntheses),
        "final_docx_path": str(docx_path),
        "appendix_path": str(appendix_path),
        "final_docx_sha256": sha256(docx_path),
        "appendix_docx_sha256": sha256(appendix_path),
        "source_pdf_preflight": preflight,
        "protocol_lock": protocol,
        "current_git_branch": git(["branch", "--show-current"]),
        "current_git_commit": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
        "remote_main_push_url": git(["remote", "get-url", "--push", "origin"]),
        "unresolved_scientific_limitations": [
            "Canonical Gemini source extraction produced an abbreviated chronological backbone for some prose; targeted repair was blocked by provider recitation filtering and is recorded in extraction QC.",
            "Evidence synthesis is abstract-level and requires expert clinical review before any clinical use.",
            "No figures or algorithms were generated in this first reconstruction run.",
        ],
    }
    write_json(root / "audit" / "final_hcc_reconstruction_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
