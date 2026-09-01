from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from hcc_gpt_map_appraise_batch import MODEL
from hcc_gpt_map_appraise_direct import OpenAIResponses, append_usage, atomic_write_json, response_text, utc_now


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
DOCX_NAME = "ESMO_HCC_2012_Living_Evidence_Update_2025-02-28_v1.docx"
APPENDIX_NAME = "ESMO_HCC_2012_Living_Evidence_Update_2025-02-28_v1_APPENDIX.docx"


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


def reconstruction_schema() -> dict[str, Any]:
    section = {
        "type": "object",
        "properties": {
            "heading": {"type": "string"},
            "original_2012_content": {"type": "string"},
            "evidence_update": {"type": "string"},
            "current_clinical_practice": {"type": "string"},
            "evidence_limitations": {"type": "string"},
            "change_signal": {
                "type": "string",
                "enum": ["CONFIRM", "MODIFY", "ADD", "REMOVE", "INSUFFICIENT_EVIDENCE"],
            },
            "cited_new_reference_numbers": {"type": "array", "items": {"type": "integer"}},
            "appendix_evidence": {"type": "string"},
        },
        "required": [
            "heading",
            "original_2012_content",
            "evidence_update",
            "current_clinical_practice",
            "evidence_limitations",
            "change_signal",
            "cited_new_reference_numbers",
            "appendix_evidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "hcc_guideline_reconstruction_chapter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "chapter_id": {"type": "string"},
                "chapter_title": {"type": "string"},
                "sections": {"type": "array", "items": section},
                "chapter_notes": {"type": "string"},
            },
            "required": ["chapter_id", "chapter_title", "sections", "chapter_notes"],
            "additionalProperties": False,
        },
    }


def instructions() -> str:
    return (
        "You are writing a complete English text-first living evidence update for a blinded "
        "scientific reconstruction of the ESMO 2012 hepatocellular carcinoma guideline through "
        "2025-02-28. Do not use any later ESMO HCC guideline, the 2025 human benchmark, or web "
        "knowledge. Use only the supplied 2012 source extraction, unit evidence memos, and reference "
        "registry.\n\n"
        "For every source section preserve the original 2012 citation numbers in the original-content "
        "field. New evidence citations must use only the supplied new reference numbers, beginning at "
        "[39]. Do not let APPENDIX or REJECT evidence drive current clinical practice. OTHER_REVIEW "
        "evidence may contextualize but must not independently justify recommendation changes. Do not "
        "generate figures or algorithms."
    )


def selected_metadata(hcc_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(hcc_root / "data" / "selected_evidence_v2.csv")
    return {clean(row.get("pmid")): row for row in rows if clean(row.get("pmid"))}


def pmid_order_from_synthesis(syntheses: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in syntheses:
        candidates: list[str] = []
        candidates.extend(str(p) for p in row.get("key_pmids", []))
        candidates.extend(str(p) for p in row.get("main_synthesis_pmids", [])[:40])
        candidates.extend(str(p) for p in row.get("context_only_pmids", [])[:12])
        candidates.extend(str(p) for p in row.get("appendix_pmids", [])[:6])
        for pmid in candidates:
            pmid = clean(pmid)
            if pmid and pmid not in seen:
                seen.add(pmid)
                ordered.append(pmid)
    return ordered


def format_reference(number: int, row: dict[str, str]) -> dict[str, Any]:
    authors = clean(row.get("authors"))
    if len(authors) > 180:
        authors = authors[:180].rsplit(" ", 1)[0] + " et al."
    title = clean(row.get("title"))
    journal = clean(row.get("journal"))
    year = clean(row.get("pub_year"))
    doi = clean(row.get("doi"))
    pmid = clean(row.get("pmid"))
    pieces = []
    if authors:
        pieces.append(authors + ".")
    if title:
        pieces.append(title)
    if journal or year:
        pieces.append(clean(f"{journal} {year}."))
    if doi:
        pieces.append(f"doi:{doi}.")
    pieces.append(f"PMID:{pmid}.")
    return {"number": number, "pmid": pmid, "citation": " ".join(pieces)}


def build_reference_registry(hcc_root: Path, syntheses: list[dict[str, Any]]) -> dict[str, Any]:
    source_refs = read_json(hcc_root / "data" / "source_extraction" / "original_references.json")
    meta = selected_metadata(hcc_root)
    ordered_pmids = [pmid for pmid in pmid_order_from_synthesis(syntheses) if pmid in meta]
    pmid_to_number = {pmid: 39 + index for index, pmid in enumerate(ordered_pmids)}
    new_refs = [format_reference(pmid_to_number[pmid], meta[pmid]) for pmid in ordered_pmids]
    registry = {
        "created_at": utc_now(),
        "original_reference_numbers": [1, 38],
        "original_references": source_refs,
        "new_reference_start": 39,
        "new_references": new_refs,
        "pmid_to_new_reference_number": pmid_to_number,
        "policy": "New references are numbered deterministically by first appearance in source-chronology synthesis order.",
    }
    atomic_write_json(hcc_root / "data" / "stageB_reference_registry.json", registry)
    return registry


def ref_numbers_for_unit(unit: dict[str, Any], registry: dict[str, Any]) -> dict[str, int]:
    mapping = registry["pmid_to_new_reference_number"]
    pmids: list[str] = []
    pmids.extend(str(p) for p in unit.get("key_pmids", []))
    pmids.extend(str(p) for p in unit.get("main_synthesis_pmids", [])[:40])
    pmids.extend(str(p) for p in unit.get("context_only_pmids", [])[:12])
    pmids.extend(str(p) for p in unit.get("appendix_pmids", [])[:6])
    return {pmid: mapping[pmid] for pmid in pmids if pmid in mapping}


def source_chronology_by_chapter(hcc_root: Path) -> dict[str, list[dict[str, Any]]]:
    ontology = read_json(hcc_root / "data" / "ontology_v1.json")
    heading_to_chapter = {clean(ch["source_heading"]).lower(): clean(ch["chapter_id"]) for ch in ontology["chapters"]}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in read_jsonl(hcc_root / "data" / "source_extraction" / "source_chronology.jsonl"):
        heading = clean(item.get("heading_path")).split("/")[0].strip().lower()
        cid = heading_to_chapter.get(heading)
        if cid:
            out[cid].append(item)
    return out


def chapter_prompt(
    chapter_id: str,
    chapter_title: str,
    source_items: list[dict[str, Any]],
    units: list[dict[str, Any]],
    registry: dict[str, Any],
) -> str:
    compact_units = []
    for unit in units:
        compact_units.append(
            {
                "unit_id": unit["unit_id"],
                "title": unit.get("evidence_unit_title"),
                "evidence_update": unit.get("evidence_update"),
                "current_clinical_practice": unit.get("current_clinical_practice"),
                "evidence_limitations": unit.get("evidence_limitations"),
                "change_signal": unit.get("change_signal"),
                "recommendation_change_supported": unit.get("recommendation_change_supported"),
                "synthesis_rationale": unit.get("synthesis_rationale"),
                "appendix_summary": unit.get("appendix_summary"),
                "new_reference_numbers_by_pmid": ref_numbers_for_unit(unit, registry),
            }
        )
    compact_source = [
        {
            "id": item.get("id"),
            "item_type": item.get("item_type"),
            "heading_path": item.get("heading_path"),
            "page": item.get("page"),
            "text": item.get("text"),
            "citation_numbers": item.get("citation_numbers"),
            "grades_or_levels": item.get("grades_or_levels"),
        }
        for item in source_items
    ]
    return (
        f"Chapter: {chapter_id} - {chapter_title}\n\n"
        "2012 source chronology items for this chapter:\n"
        f"{json.dumps(compact_source, ensure_ascii=False)}\n\n"
        "Final evidence-unit synthesis memos and allowed new reference numbers:\n"
        f"{json.dumps(compact_units, ensure_ascii=False)}\n\n"
        "Write one or more sections in the original chronology. The original_2012_content field should "
        "summarize/preserve the supplied source text with original citation numbers unchanged. Evidence "
        "updates and current practice must cite new evidence only with the supplied [number] references."
    )


def reconstruct_chapters(hcc_root: Path, model: str, retry_wait: int, max_output_tokens: int) -> list[dict[str, Any]]:
    syntheses = read_jsonl(hcc_root / "data" / "stageA_unit_evidence_synthesis.jsonl")
    registry = build_reference_registry(hcc_root, syntheses)
    by_chapter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chapter_titles: dict[str, str] = {}
    for row in syntheses:
        by_chapter[clean(row["chapter_id"])].append(row)
        chapter_titles[clean(row["chapter_id"])] = clean(row["chapter_title"])
    source_by_chapter = source_chronology_by_chapter(hcc_root)
    out_dir = hcc_root / "data" / "hcc_guideline_reconstruction"
    client = OpenAIResponses(os.environ.get("OPENAI_API_KEY", "").strip(), retry_wait)
    results: list[dict[str, Any]] = []
    for chapter_id in sorted(by_chapter):
        parsed_path = out_dir / f"{chapter_id}_parsed.json"
        if parsed_path.exists():
            results.append(read_json(parsed_path))
            continue
        prompt = chapter_prompt(
            chapter_id,
            chapter_titles[chapter_id],
            source_by_chapter.get(chapter_id, []),
            sorted(by_chapter[chapter_id], key=lambda r: clean(r.get("unit_id"))),
            registry,
        )
        body = {
            "model": model,
            "instructions": instructions(),
            "input": prompt,
            "text": {"format": reconstruction_schema()},
            "reasoning": {"effort": "high"},
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "project": "ESMO_HCC_2012_to_2025",
                "phase": "hcc_guideline_chapter_reconstruction",
                "chapter_id": chapter_id,
            },
        }
        response = client.create(body)
        append_usage(
            hcc_root,
            {
                "provider": "openai",
                "phase": "hcc_guideline_chapter_reconstruction",
                "model": response.get("model", model),
                "request_timestamp": utc_now(),
                "started_at": response.get("_request_started_at"),
                "completed_at": response.get("_request_completed_at"),
                "attempts": response.get("_retry_attempts", 0) + 1,
                "chapter_id": chapter_id,
                "usage": response.get("usage", {}),
            },
        )
        atomic_write_json(out_dir / f"{chapter_id}_raw_response.json", response)
        parsed = json.loads(response_text(response))
        parsed["chapter_id"] = chapter_id
        parsed["chapter_title"] = chapter_titles[chapter_id]
        atomic_write_json(parsed_path, parsed)
        results.append(parsed)
    results.sort(key=lambda r: clean(r["chapter_id"]))
    write_jsonl(hcc_root / "data" / "stageB_chapter_reconstruction.jsonl", results)
    manifest = {
        "created_at": utc_now(),
        "status": "COMPLETE",
        "chapter_count": len(results),
        "reference_registry": str(hcc_root / "data" / "stageB_reference_registry.json"),
        "chapter_reconstruction": str(hcc_root / "data" / "stageB_chapter_reconstruction.jsonl"),
    }
    atomic_write_json(hcc_root / "data" / "stageB_reconstruction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return results


def para(text: str, style: str | None = None) -> str:
    escaped = html.escape(text or "")
    ppr = ""
    if style == "title":
        ppr = '<w:pPr><w:pStyle w:val="Title"/></w:pPr>'
    elif style == "heading1":
        ppr = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
    elif style == "heading2":
        ppr = '<w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
    elif style == "small":
        ppr = '<w:pPr><w:spacing w:after="80"/></w:pPr>'
    return f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def document_xml(paragraphs: list[str]) -> str:
    body = "".join(paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>'
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        '<w:rPr><w:sz w:val="21"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>'
        "</w:styles>"
    )


def write_docx(path: Path, paragraphs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>",
        )
        docx.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>",
        )
        docx.writestr("word/document.xml", document_xml(paragraphs))
        docx.writestr("word/styles.xml", styles_xml())
        created = html.escape(dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat())
        docx.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:title>ESMO HCC 2012 Living Evidence Update 2025-02-28 v1</dc:title>"
            "<dc:creator>AISurgeon HCC reconstruction pipeline</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
            "</cp:coreProperties>",
        )
        docx.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>AISurgeon reconstruction pipeline</Application></Properties>",
        )


def make_docx(hcc_root: Path) -> dict[str, Any]:
    chapters = read_jsonl(hcc_root / "data" / "stageB_chapter_reconstruction.jsonl")
    registry = read_json(hcc_root / "data" / "stageB_reference_registry.json")
    output = hcc_root / "output"
    paragraphs: list[str] = []
    paragraphs.append(para("Hepatocellular carcinoma: ESMO 2012 living evidence update through 28 February 2025", "title"))
    paragraphs.append(para("Research proof-of-concept. Not intended for direct clinical use without expert review."))
    paragraphs.append(para("Source guideline: ESMO-ESDO Clinical Practice Guidelines for diagnosis, treatment and follow-up, 2012. Original citations [1]-[38] are preserved; new references begin at [39]."))
    appendix_paragraphs: list[str] = [para("Appendix Evidence", "title")]
    for chapter in chapters:
        paragraphs.append(para(f"{chapter['chapter_id']}. {chapter['chapter_title']}", "heading1"))
        for section in chapter.get("sections", []):
            heading = clean(section.get("heading")) or chapter["chapter_title"]
            paragraphs.append(para(heading, "heading2"))
            paragraphs.append(para("Original ESMO 2012 content", "small"))
            paragraphs.append(para(clean(section.get("original_2012_content"))))
            paragraphs.append(para("Evidence update through 28 February 2025", "small"))
            paragraphs.append(para(clean(section.get("evidence_update"))))
            paragraphs.append(para("Current clinical practice", "small"))
            paragraphs.append(para(clean(section.get("current_clinical_practice"))))
            paragraphs.append(para("Evidence limitations", "small"))
            paragraphs.append(para(clean(section.get("evidence_limitations"))))
            paragraphs.append(para(f"Internal change signal: {clean(section.get('change_signal'))}", "small"))
            appendix = clean(section.get("appendix_evidence"))
            if appendix:
                appendix_paragraphs.append(para(f"{chapter['chapter_id']} - {heading}", "heading1"))
                appendix_paragraphs.append(para(appendix))
    paragraphs.append(para("References", "heading1"))
    paragraphs.append(para("Original ESMO 2012 references", "heading2"))
    for ref in registry["original_references"]:
        paragraphs.append(para(f"[{ref['number']}] {clean(ref.get('full_text'))}"))
    paragraphs.append(para("New evidence references", "heading2"))
    for ref in registry["new_references"]:
        paragraphs.append(para(f"[{ref['number']}] {clean(ref.get('citation'))}"))
    appendix_paragraphs.append(para("Appendix-only PMIDs and weakly translatable evidence were not used to drive current clinical practice statements."))
    docx_path = output / DOCX_NAME
    appendix_path = output / APPENDIX_NAME
    write_docx(docx_path, paragraphs)
    write_docx(appendix_path, appendix_paragraphs)
    manifest = {
        "created_at": utc_now(),
        "status": "COMPLETE",
        "docx_path": str(docx_path),
        "appendix_docx_path": str(appendix_path),
        "chapter_count": len(chapters),
        "new_reference_count": len(registry["new_references"]),
    }
    atomic_write_json(hcc_root / "data" / "stageB_docx_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="HCC Stage-B reconstruction and DOCX generation.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", MODEL))
    parser.add_argument("--mode", choices=["run", "docx", "all"], default="all")
    parser.add_argument("--retry-wait", type=int, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)
    if args.mode in {"run", "all"}:
        reconstruct_chapters(hcc_root, args.model, args.retry_wait, args.max_output_tokens)
    if args.mode in {"docx", "all"}:
        make_docx(hcc_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
