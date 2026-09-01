from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def hcc_core() -> str:
    return (
        "("
        "hepatocellular carcinoma[Title/Abstract] OR HCC[Title/Abstract] OR "
        "\"liver cancer\"[Title/Abstract] OR \"liver neoplasms\"[MeSH Terms]"
        ")"
    )


def evidence_filter() -> str:
    return (
        "("
        "randomized controlled trial[Publication Type] OR controlled clinical trial[Publication Type] OR "
        "meta-analysis[Publication Type] OR systematic review[Publication Type] OR review[Publication Type] OR "
        "randomized[Title/Abstract] OR randomised[Title/Abstract] OR meta-analysis[Title/Abstract] OR "
        "\"systematic review\"[Title/Abstract]"
        ")"
    )


def make_query(domain_terms: list[str], start: str, end: str) -> str:
    domain = " OR ".join(domain_terms)
    return (
        f"({hcc_core()}) AND humans[MeSH Terms] AND ({evidence_filter()}) "
        f"AND ({domain}) AND {start}:{end}[Date - Publication]"
    )


def build_ontology(source_dir: Path) -> dict[str, Any]:
    document_map = read_json(source_dir / "document_map.json")
    formal_items = read_jsonl(source_dir / "formal_items.jsonl")
    chapters = [
        {
            "chapter_id": "HCC2012_CH01",
            "order_index": 1,
            "title": "incidence and epidemiology",
            "source_heading": "incidence and epidemiology",
            "evidence_units": [
                {"unit_id": "HCC2012_CH01_U01", "title": "Risk factors, epidemiology and prevention"},
                {"unit_id": "HCC2012_CH01_U02", "title": "Antiviral prevention and post-SVR risk"},
                {"unit_id": "HCC2012_CH01_U03", "title": "Surveillance eligibility and interval"},
                {"unit_id": "HCC2012_CH01_U04", "title": "Surveillance modalities including ultrasound and AFP"},
            ],
        },
        {
            "chapter_id": "HCC2012_CH02",
            "order_index": 2,
            "title": "diagnosis and pathology",
            "source_heading": "diagnosis and pathology",
            "evidence_units": [
                {"unit_id": "HCC2012_CH02_U01", "title": "Diagnostic work-up and laboratory assessment"},
                {"unit_id": "HCC2012_CH02_U02", "title": "CT and MRI non-invasive diagnosis"},
                {"unit_id": "HCC2012_CH02_U03", "title": "Pathology, biopsy and tumor seeding"},
                {"unit_id": "HCC2012_CH02_U04", "title": "Molecular pathology and biomarkers"},
            ],
        },
        {
            "chapter_id": "HCC2012_CH03",
            "order_index": 3,
            "title": "staging",
            "source_heading": "staging",
            "evidence_units": [
                {"unit_id": "HCC2012_CH03_U01", "title": "BCLC and alternative staging systems"},
                {"unit_id": "HCC2012_CH03_U02", "title": "Liver function, portal hypertension and performance status"},
                {"unit_id": "HCC2012_CH03_U03", "title": "Tumor extent, vascular invasion and extrahepatic spread"},
            ],
        },
        {
            "chapter_id": "HCC2012_CH04",
            "order_index": 4,
            "title": "management of local disease: radical therapies",
            "source_heading": "management of local disease: radical therapies",
            "evidence_units": [
                {"unit_id": "HCC2012_CH04_U01", "title": "Surgical resection"},
                {"unit_id": "HCC2012_CH04_U02", "title": "Local ablation including RFA and PEI"},
                {"unit_id": "HCC2012_CH04_U03", "title": "Liver transplantation and Milan criteria"},
                {"unit_id": "HCC2012_CH04_U04", "title": "Bridging, downstaging and transplant wait-list therapy"},
                {"unit_id": "HCC2012_CH04_U05", "title": "Neoadjuvant and adjuvant therapy after curative treatment"},
            ],
        },
        {
            "chapter_id": "HCC2012_CH05",
            "order_index": 5,
            "title": "management of locally advanced/metastatic disease: palliative treatments",
            "source_heading": "management of locally advanced/metastatic disease: palliative treatments",
            "evidence_units": [
                {"unit_id": "HCC2012_CH05_U01", "title": "Transarterial chemoembolization and drug-eluting beads"},
                {"unit_id": "HCC2012_CH05_U02", "title": "TACE combined or sequenced with systemic therapy"},
                {"unit_id": "HCC2012_CH05_U03", "title": "Radioembolization and internal radiation"},
                {"unit_id": "HCC2012_CH05_U04", "title": "Systemic targeted therapy"},
                {"unit_id": "HCC2012_CH05_U05", "title": "Immunotherapy and immunotherapy combinations"},
                {"unit_id": "HCC2012_CH05_U06", "title": "Cytotoxic, hormonal and other unsupported systemic therapies"},
                {"unit_id": "HCC2012_CH05_U07", "title": "External beam radiotherapy and symptom control"},
                {"unit_id": "HCC2012_CH05_U08", "title": "Best supportive care and end-stage disease"},
            ],
        },
        {
            "chapter_id": "HCC2012_CH06",
            "order_index": 6,
            "title": "response evaluation and follow-up",
            "source_heading": "response evaluation and follow-up",
            "evidence_units": [
                {"unit_id": "HCC2012_CH06_U01", "title": "mRECIST, RECIST and imaging response assessment"},
                {"unit_id": "HCC2012_CH06_U02", "title": "AFP and serum markers during treatment"},
                {"unit_id": "HCC2012_CH06_U03", "title": "Post-treatment follow-up after radical therapy"},
                {"unit_id": "HCC2012_CH06_U04", "title": "Monitoring during TACE and systemic therapy"},
            ],
        },
    ]
    formal_by_heading: dict[str, list[dict[str, Any]]] = {}
    for item in formal_items:
        formal_by_heading.setdefault(item.get("heading_path", "unknown"), []).append(item)
    return {
        "created_at": utc_now(),
        "source": {
            "document_map_chapters": document_map.get("chapters", []),
            "formal_item_count": len(formal_items),
        },
        "ontology_version": "hcc_source_ontology_v1",
        "chapters": chapters,
        "formal_items_by_source_heading": formal_by_heading,
        "policies": {
            "allow_multi_unit_mapping": True,
            "allow_new_subunits_after_recovery": True,
            "do_not_split_to_reduce_paper_counts": True,
            "major_chapter_candidate_requires_unrepresentable_evidence": True,
        },
    }


DOMAIN_TERMS = {
    "HCC2012_CH01_U01": ["epidemiology[Title/Abstract]", "incidence[Title/Abstract]", "risk factor*[Title/Abstract]", "prevention[Title/Abstract]", "vaccination[Title/Abstract]", "aflatoxin[Title/Abstract]", "alcohol[Title/Abstract]", "obesity[Title/Abstract]"],
    "HCC2012_CH01_U02": ["hepatitis B[Title/Abstract]", "HBV[Title/Abstract]", "hepatitis C[Title/Abstract]", "HCV[Title/Abstract]", "antiviral[Title/Abstract]", "sustained virological response[Title/Abstract]", "SVR[Title/Abstract]"],
    "HCC2012_CH01_U03": ["surveillance[Title/Abstract]", "screening[Title/Abstract]", "early detection[Title/Abstract]", "surveillance interval[Title/Abstract]"],
    "HCC2012_CH01_U04": ["ultrasound[Title/Abstract]", "ultrasonography[Title/Abstract]", "alpha-fetoprotein[Title/Abstract]", "AFP[Title/Abstract]", "biomarker[Title/Abstract]"],
    "HCC2012_CH02_U01": ["diagnostic work-up[Title/Abstract]", "diagnosis[Title/Abstract]", "laboratory[Title/Abstract]", "workup[Title/Abstract]"],
    "HCC2012_CH02_U02": ["CT[Title/Abstract]", "computed tomography[Title/Abstract]", "MRI[Title/Abstract]", "magnetic resonance[Title/Abstract]", "imaging[Title/Abstract]", "LI-RADS[Title/Abstract]", "washout[Title/Abstract]"],
    "HCC2012_CH02_U03": ["pathology[Title/Abstract]", "biopsy[Title/Abstract]", "histolog*[Title/Abstract]", "tumor seeding[Title/Abstract]", "stromal invasion[Title/Abstract]"],
    "HCC2012_CH02_U04": ["molecular[Title/Abstract]", "genomic*[Title/Abstract]", "biomarker*[Title/Abstract]", "glypican[Title/Abstract]", "cytokeratin 19[Title/Abstract]"],
    "HCC2012_CH03_U01": ["BCLC[Title/Abstract]", "Barcelona Clinic Liver Cancer[Title/Abstract]", "staging[Title/Abstract]", "prognostic score[Title/Abstract]", "TNM[Title/Abstract]"],
    "HCC2012_CH03_U02": ["Child-Pugh[Title/Abstract]", "ALBI[Title/Abstract]", "liver function[Title/Abstract]", "portal hypertension[Title/Abstract]", "performance status[Title/Abstract]"],
    "HCC2012_CH03_U03": ["vascular invasion[Title/Abstract]", "portal vein thrombosis[Title/Abstract]", "extrahepatic spread[Title/Abstract]", "tumor burden[Title/Abstract]"],
    "HCC2012_CH04_U01": ["resection[Title/Abstract]", "hepatectomy[Title/Abstract]", "surgery[Title/Abstract]", "surgical[Title/Abstract]"],
    "HCC2012_CH04_U02": ["radiofrequency ablation[Title/Abstract]", "RFA[Title/Abstract]", "microwave ablation[Title/Abstract]", "percutaneous ethanol injection[Title/Abstract]", "PEI[Title/Abstract]", "local ablation[Title/Abstract]"],
    "HCC2012_CH04_U03": ["liver transplantation[Title/Abstract]", "transplant[Title/Abstract]", "Milan criteria[Title/Abstract]"],
    "HCC2012_CH04_U04": ["bridging[Title/Abstract]", "downstaging[Title/Abstract]", "waitlist[Title/Abstract]", "waiting list[Title/Abstract]"],
    "HCC2012_CH04_U05": ["adjuvant[Title/Abstract]", "neoadjuvant[Title/Abstract]", "recurrence prevention[Title/Abstract]", "STORM[Title/Abstract]"],
    "HCC2012_CH05_U01": ["TACE[Title/Abstract]", "chemoembolization[Title/Abstract]", "chemoembolisation[Title/Abstract]", "drug-eluting bead*[Title/Abstract]", "DEB-TACE[Title/Abstract]"],
    "HCC2012_CH05_U02": ["TACE[Title/Abstract] AND sorafenib[Title/Abstract]", "chemoembolization[Title/Abstract] AND systemic[Title/Abstract]", "combination therapy[Title/Abstract]"],
    "HCC2012_CH05_U03": ["radioembolization[Title/Abstract]", "radioembolisation[Title/Abstract]", "Yttrium-90[Title/Abstract]", "Y-90[Title/Abstract]", "selective internal radiation[Title/Abstract]", "SIRT[Title/Abstract]"],
    "HCC2012_CH05_U04": ["sorafenib[Title/Abstract]", "lenvatinib[Title/Abstract]", "regorafenib[Title/Abstract]", "cabozantinib[Title/Abstract]", "ramucirumab[Title/Abstract]", "targeted therapy[Title/Abstract]", "tyrosine kinase inhibitor[Title/Abstract]"],
    "HCC2012_CH05_U05": ["immunotherapy[Title/Abstract]", "immune checkpoint[Title/Abstract]", "nivolumab[Title/Abstract]", "pembrolizumab[Title/Abstract]", "atezolizumab[Title/Abstract]", "bevacizumab[Title/Abstract]", "durvalumab[Title/Abstract]", "tremelimumab[Title/Abstract]", "ipilimumab[Title/Abstract]"],
    "HCC2012_CH05_U06": ["chemotherapy[Title/Abstract]", "doxorubicin[Title/Abstract]", "cisplatin[Title/Abstract]", "tamoxifen[Title/Abstract]", "somatostatin[Title/Abstract]", "anti-androgen[Title/Abstract]"],
    "HCC2012_CH05_U07": ["radiotherapy[Title/Abstract]", "radiation therapy[Title/Abstract]", "SBRT[Title/Abstract]", "stereotactic[Title/Abstract]", "bone metastases[Title/Abstract]", "pain[Title/Abstract]"],
    "HCC2012_CH05_U08": ["supportive care[Title/Abstract]", "palliative care[Title/Abstract]", "end-stage[Title/Abstract]", "quality of life[Title/Abstract]"],
    "HCC2012_CH06_U01": ["mRECIST[Title/Abstract]", "modified RECIST[Title/Abstract]", "RECIST[Title/Abstract]", "response assessment[Title/Abstract]", "tumor response[Title/Abstract]"],
    "HCC2012_CH06_U02": ["alpha-fetoprotein[Title/Abstract]", "AFP[Title/Abstract]", "tumor marker[Title/Abstract]", "serum marker[Title/Abstract]"],
    "HCC2012_CH06_U03": ["follow-up[Title/Abstract]", "recurrence surveillance[Title/Abstract]", "postoperative surveillance[Title/Abstract]", "after resection[Title/Abstract]", "after ablation[Title/Abstract]"],
    "HCC2012_CH06_U04": ["monitoring[Title/Abstract]", "progression[Title/Abstract]", "treatment decision[Title/Abstract]", "liver decompensation[Title/Abstract]"],
}


def build_queries(ontology: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    queries = []
    for chapter in ontology["chapters"]:
        for unit in chapter["evidence_units"]:
            unit_id = unit["unit_id"]
            queries.append(
                {
                    "query_id": f"Q_{unit_id}",
                    "chapter_id": chapter["chapter_id"],
                    "chapter_title": chapter["title"],
                    "unit_id": unit_id,
                    "unit_title": unit["title"],
                    "date_start": start,
                    "date_end": end,
                    "query": make_query(DOMAIN_TERMS[unit_id], start, end),
                    "evidence_filter_policy": "Search retains RCTs, reviews, systematic reviews, and meta-analyses; guideline/consensus records are explicitly excluded during deterministic selection.",
                }
            )
    return {
        "created_at": utc_now(),
        "search_start": start,
        "search_end": end,
        "query_count": len(queries),
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HCC source-derived ontology and PubMed query registry.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    protocol = read_json(hcc_root / "config" / "protocol_lock.json")
    source_dir = hcc_root / "data" / "source_extraction"
    ontology = build_ontology(source_dir)
    query_registry = build_queries(ontology, protocol["dates"]["search_start"], protocol["dates"]["search_end"])
    atomic_write_json(hcc_root / "data" / "ontology_v1.json", ontology)
    atomic_write_json(hcc_root / "data" / "pubmed_query_registry.json", query_registry)
    print(json.dumps({"ontology_chapters": len(ontology["chapters"]), "query_count": query_registry["query_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
