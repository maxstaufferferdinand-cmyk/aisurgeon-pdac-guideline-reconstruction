from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
DEFAULT_SOURCE_PDF = DEFAULT_HCC_ROOT / "ESMOHCC2012.pdf"
LOCKED_SEARCH_END = "2025-02-28"
REQUIRED_GEMINI_MODEL = "models/gemini-3.5-flash"
REQUIRED_OPENAI_MODEL = "gpt-5.6-sol"
LOCAL_PRICING_REQUIRED = "LOCAL_PRICING_REQUIRED"
PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION = "PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION"
USER_BUDGET_DEFAULTS = {
    "HCC_MAX_TOTAL_API_USD": 500.0,
    "HCC_MAX_OPENAI_API_USD": 300.0,
    "HCC_MAX_GEMINI_API_USD": 200.0,
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def run_command(args: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"missing executable: {exc.filename}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_pdfinfo(stdout: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hcc_preflight_") as temp_dir:
        text_path = Path(temp_dir) / "source.txt"
        result = run_command(["pdftotext", "-layout", str(pdf_path), str(text_path)])
        if not result["ok"]:
            return {
                "ok": False,
                "error": result.get("stderr") or result.get("error") or "pdftotext failed",
            }
        text = text_path.read_text(encoding="utf-8", errors="replace")
    return {
        "ok": True,
        "line_count": len(text.splitlines()),
        "char_count": len(text),
        "text": text,
    }


def render_page_preview(pdf_path: Path, hcc_root: Path) -> dict[str, Any]:
    out_prefix = hcc_root / "audit" / "source_page1_preview"
    result = run_command(
        ["pdftoppm", "-f", "1", "-l", "1", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
    )
    preview_path = out_prefix.with_suffix(".png")
    return {
        "ok": bool(result["ok"] and preview_path.exists()),
        "path": str(preview_path) if preview_path.exists() else None,
        "error": None if result["ok"] else (result.get("stderr") or result.get("error")),
    }


def month_after(month_name: str, year: int) -> str | None:
    try:
        month = dt.datetime.strptime(month_name, "%B").month
    except ValueError:
        return None
    first_this_month = dt.date(year, month, 1)
    if month == 12:
        first_next_month = dt.date(year + 1, 1, 1)
    else:
        first_next_month = dt.date(year, month + 1, 1)
    if first_this_month == dt.date(2012, 6, 1):
        return "2012-07-01"
    return first_next_month.isoformat()


def source_facts(metadata: dict[str, str], text: str) -> dict[str, Any]:
    last_update_match = re.search(r"last update\s+([A-Za-z]+)\s+(\d{4})", text, re.I)
    search_start = None
    last_update = None
    if last_update_match:
        last_update = f"{last_update_match.group(1)} {last_update_match.group(2)}"
        search_start = month_after(last_update_match.group(1), int(last_update_match.group(2)))

    publication_date = None
    subject = metadata.get("Subject", "")
    publication_match = re.search(r"\((\d{4})\)", subject)
    if publication_match:
        publication_date = publication_match.group(1)

    # The article is two-column; some reference-list numbers share a line with
    # the opposite column, so use a lookahead to capture overlapping list items.
    reference_numbers = sorted(
        {
            int(match.group(1))
            for match in re.finditer(r"(?=(?:^|\s)(\d{1,3})\.\s+[A-Z])", text, re.M)
        }
    )

    tables = []
    for match in re.finditer(r"(?m)^(Table\s+\d+\.\s+.+)$", text):
        tables.append(match.group(1).strip())
    figures = []
    for match in re.finditer(r"(?m)^(Figure\s+\d+\s+.+)$", text):
        figures.append(match.group(1).strip())

    chapter_order = [
        "incidence and epidemiology",
        "diagnosis and pathology",
        "staging",
        "management of local disease: radical therapies",
        "management of locally advanced/metastatic disease: palliative treatments",
        "response evaluation and follow-up",
    ]
    detected_chapters = [chapter for chapter in chapter_order if re.search(re.escape(chapter), text, re.I)]
    source_subsections = [
        subsection
        for subsection in [
            "transcatheter devices",
            "systemic therapy",
            "external beam radiotherapy",
            "note",
            "conflict of interest",
            "references",
        ]
        if re.search(r"(?m)^" + re.escape(subsection) + r"\s*$", text, re.I)
    ]

    grading_excerpt = None
    grading_match = re.search(
        r"Levels of\s+evidence\s+\[I\S?V\].+?standard clinical practice by the experts and the ESMO faculty\.",
        text,
        re.I | re.S,
    )
    if grading_match:
        grading_excerpt = " ".join(grading_match.group(0).split())

    return {
        "source_guideline_title": metadata.get("Title"),
        "source_guideline_publication_date": publication_date,
        "documented_last_update": last_update,
        "locked_search_start": search_start,
        "locked_search_end": LOCKED_SEARCH_END,
        "original_reference_number_range": {
            "min": min(reference_numbers) if reference_numbers else None,
            "max": max(reference_numbers) if reference_numbers else None,
            "count": len(reference_numbers),
        },
        "original_grading_system": {
            "levels": "I-V",
            "grades": "A-D",
            "source_description": grading_excerpt,
        },
        "original_chapter_order": detected_chapters,
        "source_native_subsection_order": source_subsections,
        "tables_detected": tables,
        "figures_detected": figures,
    }


def env_float(name: str) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return USER_BUDGET_DEFAULTS[name]
    return float(raw)


def list_gemini_models() -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY/GOOGLE_API_KEY missing"}
    url = "https://generativelanguage.googleapis.com/v1beta/models?key=" + urllib.parse.quote(key)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}
    models = [
        {
            "name": model.get("name"),
            "displayName": model.get("displayName"),
            "supportedGenerationMethods": model.get("supportedGenerationMethods", []),
            "inputTokenLimit": model.get("inputTokenLimit"),
            "outputTokenLimit": model.get("outputTokenLimit"),
        }
        for model in data.get("models", [])
        if "gemini" in (model.get("name", "") + model.get("displayName", "")).lower()
    ]
    return {
        "ok": True,
        "api_version": "v1beta",
        "required_model": REQUIRED_GEMINI_MODEL,
        "required_model_available": any(model["name"] == REQUIRED_GEMINI_MODEL for model in models),
        "models": models,
    }


def list_openai_models() -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"ok": False, "error": "OPENAI_API_KEY missing"}
    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:300]}
    model_ids = sorted(model.get("id", "") for model in data.get("data", []))
    return {
        "ok": True,
        "api_version": "v1",
        "required_model": REQUIRED_OPENAI_MODEL,
        "required_model_available": REQUIRED_OPENAI_MODEL in model_ids,
        "filtered_models": [
            model_id
            for model_id in model_ids
            if "gpt-5" in model_id.lower() or "sol" in model_id.lower()
        ],
    }


def load_pricing_config(path: Path, models: dict[str, str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "error": "provider pricing config missing",
            "required_schema": {
                "providers": {
                    "openai": {
                        models["openai"]: {
                            "input_usd_per_million_tokens": "number",
                            "output_usd_per_million_tokens": "number",
                            "cached_input_usd_per_million_tokens": "optional number",
                        }
                    },
                    "gemini": {
                        models["gemini"]: {
                            "input_usd_per_million_tokens": "number",
                            "output_usd_per_million_tokens": "number",
                            "pdf_input_usd_per_million_tokens": "optional number",
                        }
                    },
                }
            },
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "path": str(path), "error": f"invalid JSON: {exc}"}

    missing: list[str] = []
    for provider, model_id in models.items():
        model_prices = data.get("providers", {}).get(provider, {}).get(model_id)
        if not isinstance(model_prices, dict):
            missing.append(f"providers.{provider}.{model_id}")
            continue
        for field in ("input_usd_per_million_tokens", "output_usd_per_million_tokens"):
            if not isinstance(model_prices.get(field), (int, float)):
                missing.append(f"providers.{provider}.{model_id}.{field}")
    return {"ok": not missing, "path": str(path), "missing": missing, "loaded": not missing}


def build_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hcc_root = Path(args.hcc_root)
    pdf_path = Path(args.pdf)
    budgets = {name: env_float(name) for name in USER_BUDGET_DEFAULTS}
    cost_mode = args.cost_mode

    pdf_exists = pdf_path.exists()
    pdfinfo_result = run_command(["pdfinfo", str(pdf_path)]) if pdf_exists else {"ok": False, "error": "PDF missing"}
    metadata = parse_pdfinfo(pdfinfo_result.get("stdout", "")) if pdfinfo_result["ok"] else {}
    text_result = extract_pdf_text(pdf_path) if pdf_exists else {"ok": False, "error": "PDF missing"}
    text = text_result.get("text", "") if text_result["ok"] else ""
    preview = render_page_preview(pdf_path, hcc_root) if pdf_exists else {"ok": False, "error": "PDF missing"}

    facts = source_facts(metadata, text)
    source_preflight = {
        "created_at": utc_now(),
        "pdf_path": str(pdf_path),
        "pdf_exists": pdf_exists,
        "sha256": sha256_file(pdf_path) if pdf_exists else None,
        "pdfinfo_ok": pdfinfo_result["ok"],
        "pdf_metadata": metadata,
        "page_count": int(metadata["Pages"]) if metadata.get("Pages", "").isdigit() else None,
        "native_text_readability": {
            "pdftotext_ok": text_result["ok"],
            "line_count": text_result.get("line_count"),
            "char_count": text_result.get("char_count"),
        },
        "visual_readability": preview,
        "source_facts": facts,
        "blind_benchmark_policy": {
            "locked_benchmark_path": str(hcc_root / "ESMOHCC2025.pdf"),
            "status": "not_inspected",
        },
    }

    gemini_models = list_gemini_models()
    openai_models = list_openai_models()
    model_inventory = {
        "created_at": utc_now(),
        "gemini": gemini_models,
        "openai": openai_models,
    }

    pricing = load_pricing_config(
        hcc_root / "config" / "provider_pricing.json",
        {"openai": REQUIRED_OPENAI_MODEL, "gemini": REQUIRED_GEMINI_MODEL},
    )
    if cost_mode == PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION and not pricing["ok"]:
        pricing = {
            **pricing,
            "waived": True,
            "waiver_reason": "Explicit user authorization for provider-managed execution with no local monetary cost estimation.",
        }

    mandatory_missing = []
    if not pdf_exists:
        mandatory_missing.append("source PDF missing")
    if source_preflight["page_count"] != 8:
        mandatory_missing.append("unexpected or unknown source page count")
    if not text_result["ok"]:
        mandatory_missing.append("native text unreadable by deterministic preflight")
    if not preview["ok"]:
        mandatory_missing.append("page preview rendering failed")
    for key in (
        "source_guideline_title",
        "source_guideline_publication_date",
        "documented_last_update",
        "locked_search_start",
    ):
        if not facts.get(key):
            mandatory_missing.append(f"missing source fact: {key}")
    if facts["original_reference_number_range"]["min"] != 1 or facts["original_reference_number_range"]["max"] != 38:
        mandatory_missing.append("original reference range not confirmed as 1-38")

    paid_phase_blockers = []
    if not gemini_models.get("required_model_available"):
        paid_phase_blockers.append(f"required Gemini model unavailable: {REQUIRED_GEMINI_MODEL}")
    if not openai_models.get("required_model_available"):
        paid_phase_blockers.append(f"required OpenAI model unavailable: {REQUIRED_OPENAI_MODEL}")
    if not pricing["ok"] and cost_mode != PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION:
        paid_phase_blockers.append("provider pricing config missing or incomplete")

    protocol_lock = {
        "created_at": utc_now(),
        "protocol": "ESMO HCC 2012 living evidence update through 2025-02-28",
        "branch": "codex/hcc-2012-to-2025",
        "source_pdf": {
            "path": str(pdf_path),
            "sha256": source_preflight["sha256"],
            "page_count": source_preflight["page_count"],
            "title": facts["source_guideline_title"],
            "publication_date": facts["source_guideline_publication_date"],
            "documented_last_update": facts["documented_last_update"],
        },
        "dates": {
            "search_start": facts["locked_search_start"],
            "search_end": LOCKED_SEARCH_END,
            "search_start_rationale": "First day after the source PDF documented last-update month.",
        },
        "models": {
            "gemini_native_pdf_extractor": REQUIRED_GEMINI_MODEL,
            "openai_mapping_appraisal_synthesis": REQUIRED_OPENAI_MODEL,
        },
        "cost_control": {
            "mode": cost_mode,
            "paid_api_execution_authorized": cost_mode == PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION,
            "local_monetary_cost_accounting": "disabled_by_explicit_user_instruction"
            if cost_mode == PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION
            else "enabled_when_provider_pricing_config_is_available",
            "provider_side_billing_controls": "not_evaluated_by_pipeline",
            "token_and_request_usage_audited": True,
            "provider_pricing_required": cost_mode != PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION,
            "provider_pricing_gate_waived": cost_mode == PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION,
        },
        "budgets_usd": budgets,
        "pricing": pricing,
        "evidence_policies": {
            "retain": [
                "human randomized controlled trials",
                "meta-analyses",
                "systematic reviews",
                "other reviews",
            ],
            "exclude_overrides": [
                "clinical practice guidelines",
                "treatment guidelines",
                "consensus statements",
                "expert consensus publications",
                "society recommendations",
                "position statements",
            ],
            "appraisal_statuses": ["MAIN_SYNTHESIS", "CONTEXT_ONLY", "APPENDIX", "REJECT"],
            "change_signals": ["CONFIRM", "MODIFY", "ADD", "REMOVE", "INSUFFICIENT_EVIDENCE"],
            "final_reference_number_start": 39,
            "blind_benchmark": "Do not inspect ESMOHCC2025.pdf before final reconstruction lock and hash.",
        },
        "mandatory_source_preflight_complete": not mandatory_missing,
        "mandatory_source_preflight_missing": mandatory_missing,
        "paid_phase_blocked": bool(paid_phase_blockers),
        "paid_phase_blockers": paid_phase_blockers,
    }

    previous_ledger = read_json_if_exists(hcc_root / "run_state" / "cost_ledger.json") or {}
    previous_requests = previous_ledger.get("requests", [])
    previous_phases = previous_ledger.get("phases", [])
    cost_ledger = {
        "created_at": utc_now(),
        "schema_version": 1,
        "budgets_usd": budgets,
        "cost_control": protocol_lock["cost_control"],
        "pricing_config": pricing,
        "cumulative_estimated_usd": {"openai": 0.0, "gemini": 0.0, "total": 0.0},
        "cumulative_actual_usd": {"openai": 0.0, "gemini": 0.0, "total": 0.0},
        "requests": previous_requests if isinstance(previous_requests, list) else [],
        "phases": previous_phases if isinstance(previous_phases, list) else [],
        "hard_stop_active": bool(paid_phase_blockers),
        "hard_stop_reasons": paid_phase_blockers,
        "usage_measurement_policy": {
            "record_request_counts": True,
            "record_model_identifiers": True,
            "record_batch_identifiers": True,
            "record_tokens_when_returned": True,
            "calculate_usd": cost_mode != PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION and pricing.get("ok", False),
        },
    }

    return source_preflight, model_inventory, protocol_lock | {"cost_ledger_initial": cost_ledger}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create HCC preflight and protocol-lock artifacts.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--pdf", default=os.environ.get("HCC_PDF", str(DEFAULT_SOURCE_PDF)))
    parser.add_argument(
        "--cost-mode",
        choices=[LOCAL_PRICING_REQUIRED, PROVIDER_MANAGED_NO_LOCAL_COST_ESTIMATION],
        default=os.environ.get("HCC_COST_MODE", LOCAL_PRICING_REQUIRED),
        help="Cost-control mode. The provider-managed mode waives local pricing gates.",
    )
    args = parser.parse_args()

    hcc_root = Path(args.hcc_root)
    source_preflight, model_inventory, protocol_and_cost = build_preflight(args)
    cost_ledger = protocol_and_cost.pop("cost_ledger_initial")

    atomic_write_json(hcc_root / "audit" / "source_pdf_preflight.json", source_preflight)
    atomic_write_json(hcc_root / "audit" / "provider_model_inventory.json", model_inventory)
    atomic_write_json(hcc_root / "config" / "protocol_lock.json", protocol_and_cost)
    atomic_write_json(hcc_root / "run_state" / "cost_ledger.json", cost_ledger)

    summary = {
        "source_preflight_complete": protocol_and_cost["mandatory_source_preflight_complete"],
        "paid_phase_blocked": protocol_and_cost["paid_phase_blocked"],
        "paid_phase_blockers": protocol_and_cost["paid_phase_blockers"],
        "protocol_lock": str(hcc_root / "config" / "protocol_lock.json"),
        "cost_ledger": str(hcc_root / "run_state" / "cost_ledger.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if protocol_and_cost["paid_phase_blocked"] or not protocol_and_cost["mandatory_source_preflight_complete"] else 0


if __name__ == "__main__":
    sys.exit(main())
