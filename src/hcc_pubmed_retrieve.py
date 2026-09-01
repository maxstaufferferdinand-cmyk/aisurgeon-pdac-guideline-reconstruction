from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_HCC_ROOT = Path("/mnt/c/living_guideline_platform/PilotPOC/PilotHCC")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TRANSIENT_STATUS = {500, 502, 503, 504}
FIELDNAMES = [
    "pmid",
    "query_id",
    "chapter_id",
    "unit_id",
    "slice_start",
    "slice_end",
    "title",
    "abstract",
    "journal",
    "pub_year",
    "pub_date",
    "publication_types",
    "mesh_terms",
    "authors",
    "doi",
    "raw_xml_file",
]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def date_range_half_years(start: str, end: str) -> list[tuple[str, str]]:
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    ranges: list[tuple[str, str]] = []
    current = start_date
    while current <= end_date:
        half_end_month = 6 if current.month <= 6 else 12
        next_end = dt.date(current.year, half_end_month, 30 if half_end_month == 6 else 31)
        if next_end > end_date:
            next_end = end_date
        ranges.append((current.isoformat(), next_end.isoformat()))
        current = next_end + dt.timedelta(days=1)
    return ranges


def split_range(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    if s >= e:
        return None
    mid = s + (e - s) // 2
    return (s.isoformat(), mid.isoformat()), ((mid + dt.timedelta(days=1)).isoformat(), e.isoformat())


def request_with_retries(url: str, params: dict[str, str], *, method: str = "POST") -> bytes:
    encoded = urllib.parse.urlencode(params).encode("utf-8")
    retry_count = 0
    while True:
        try:
            if method == "POST":
                req = urllib.request.Request(url, data=encoded, method="POST")
            else:
                req = urllib.request.Request(url + "?" + encoded.decode("utf-8"), method="GET")
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT_STATUS:
                retry_count += 1
                time.sleep(120)
                continue
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403} or "api key" in body.lower():
                raise RuntimeError(f"NCBI credential or permission error: HTTP {exc.code}") from exc
            if exc.code == 429:
                retry_count += 1
                time.sleep(120)
                continue
            raise
        except (TimeoutError, urllib.error.URLError):
            retry_count += 1
            time.sleep(120)


def ncbi_params(email: str, tool: str, api_key: str | None) -> dict[str, str]:
    params = {"email": email, "tool": tool}
    if api_key:
        params["api_key"] = api_key
    return params


def esearch(term: str, start: str, end: str, creds: dict[str, str]) -> tuple[int, list[str]]:
    dated_term = re.sub(
        r"\d{4}[-/]\d{2}[-/]\d{2}:\d{4}[-/]\d{2}[-/]\d{2}\[Date - Publication\]",
        f"{start}:{end}[Date - Publication]",
        term,
    )
    params = {
        **creds,
        "db": "pubmed",
        "term": dated_term,
        "retmode": "json",
        "retmax": "10000",
        "sort": "pub date",
    }
    data = json.loads(request_with_retries(f"{EUTILS}/esearch.fcgi", params).decode("utf-8"))
    result = data.get("esearchresult", {})
    return int(result.get("count", "0")), result.get("idlist", [])


def efetch(pmids: list[str], creds: dict[str, str]) -> bytes:
    params = {
        **creds,
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    return request_with_retries(f"{EUTILS}/efetch.fcgi", params)


def text_of(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def article_year(article: ET.Element) -> str:
    for path in [
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
        "./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year",
        "./MedlineCitation/DateCompleted/Year",
    ]:
        value = article.findtext(path)
        if value:
            return value
    medline_date = article.findtext("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"\d{4}", medline_date)
    return match.group(0) if match else ""


def article_pub_date(article: ET.Element) -> str:
    pubdate = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pubdate is None:
        return article_year(article)
    parts = [pubdate.findtext(name) or "" for name in ("Year", "Month", "Day")]
    medline = pubdate.findtext("MedlineDate") or ""
    return " ".join(part for part in parts if part) or medline


def parse_articles(xml_bytes: bytes, raw_xml_file: str, provenance: dict[str, str]) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    rows: list[dict[str, str]] = []
    for article in root.findall("./PubmedArticle"):
        pmid = article.findtext("./MedlineCitation/PMID") or ""
        title = text_of(article.find("./MedlineCitation/Article/ArticleTitle"))
        abstract_parts = [
            text_of(elem)
            for elem in article.findall("./MedlineCitation/Article/Abstract/AbstractText")
        ]
        publication_types = [
            text_of(elem)
            for elem in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
        ]
        mesh_terms = [
            text_of(elem.find("DescriptorName"))
            for elem in article.findall("./MedlineCitation/MeshHeadingList/MeshHeading")
        ]
        authors = []
        for author in article.findall("./MedlineCitation/Article/AuthorList/Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            collective = author.findtext("CollectiveName") or ""
            name = " ".join(part for part in [fore, last] if part) or collective
            if name:
                authors.append(name)
        dois = [
            elem.text.strip()
            for elem in article.findall("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
            if elem.text and elem.text.strip()
        ]
        rows.append(
            {
                **provenance,
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(abstract_parts),
                "journal": article.findtext("./MedlineCitation/Article/Journal/Title") or "",
                "pub_year": article_year(article),
                "pub_date": article_pub_date(article),
                "publication_types": "|".join(publication_types),
                "mesh_terms": "|".join(mesh_terms),
                "authors": "|".join(authors),
                "doi": dois[0] if dois else "",
                "raw_xml_file": raw_xml_file,
            }
        )
    return rows


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def retrieve_slice(
    query: dict[str, Any],
    start: str,
    end: str,
    creds: dict[str, str],
    hcc_root: Path,
    state: dict[str, Any],
    max_per_slice: int,
) -> int:
    slice_key = f"{query['query_id']}|{start}|{end}"
    if slice_key in state.get("completed_slices", {}):
        return int(state["completed_slices"][slice_key]["count"])
    count, pmids = esearch(query["query"], start, end, creds)
    append_jsonl(
        hcc_root / "logs" / "pubmed_search_log.jsonl",
        {
            "timestamp": utc_now(),
            "query_id": query["query_id"],
            "unit_id": query["unit_id"],
            "chapter_id": query["chapter_id"],
            "slice_start": start,
            "slice_end": end,
            "count": count,
        },
    )
    if count > max_per_slice:
        split = split_range(start, end)
        if split is None:
            raise RuntimeError(f"PubMed slice too large and cannot split further: {slice_key} count={count}")
        return sum(
            retrieve_slice(query, part_start, part_end, creds, hcc_root, state, max_per_slice)
            for part_start, part_end in split
        )
    if not pmids:
        state.setdefault("completed_slices", {})[slice_key] = {"count": 0, "completed_at": utc_now()}
        atomic_write_json(hcc_root / "run_state" / "pubmed_retrieval_state.json", state)
        return 0

    raw_dir = hcc_root / "data" / "raw_xml" / query["query_id"]
    csv_path = hcc_root / "data" / "pubmed_results.csv"
    fetched = 0
    for batch_start in range(0, len(pmids), 200):
        batch_pmids = pmids[batch_start : batch_start + 200]
        batch_key = f"{slice_key}|{batch_start}"
        if batch_key in state.get("completed_batches", {}):
            fetched += len(batch_pmids)
            continue
        xml = efetch(batch_pmids, creds)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{start}_to_{end}_batch_{batch_start:05d}.xml"
        temp = raw_path.with_suffix(".xml.tmp")
        temp.write_bytes(xml)
        temp.replace(raw_path)
        rows = parse_articles(
            xml,
            str(raw_path),
            {
                "query_id": query["query_id"],
                "chapter_id": query["chapter_id"],
                "unit_id": query["unit_id"],
                "slice_start": start,
                "slice_end": end,
            },
        )
        write_rows(csv_path, rows)
        fetched += len(rows)
        state.setdefault("completed_batches", {})[batch_key] = {
            "count": len(rows),
            "completed_at": utc_now(),
            "raw_xml_file": str(raw_path),
        }
        atomic_write_json(hcc_root / "run_state" / "pubmed_retrieval_state.json", state)
        time.sleep(0.11)
    state.setdefault("completed_slices", {})[slice_key] = {
        "count": count,
        "fetched": fetched,
        "completed_at": utc_now(),
    }
    atomic_write_json(hcc_root / "run_state" / "pubmed_retrieval_state.json", state)
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve HCC PubMed evidence with resumable half-year slices.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
    parser.add_argument("--query-id", default="all")
    parser.add_argument("--max-per-slice", type=int, default=9500)
    args = parser.parse_args()
    hcc_root = Path(args.hcc_root)

    email = os.environ.get("NCBI_EMAIL", "").strip()
    tool = os.environ.get("NCBI_TOOL", "aisurgeon_hcc_reconstruction").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if not email:
        raise RuntimeError("NCBI_EMAIL is not set.")
    creds = ncbi_params(email, tool, api_key or None)
    registry = read_json(hcc_root / "data" / "pubmed_query_registry.json")
    queries = registry["queries"]
    if args.query_id != "all":
        queries = [query for query in queries if query["query_id"] == args.query_id]
    if not queries:
        raise RuntimeError(f"No queries matched {args.query_id}")

    state_path = hcc_root / "run_state" / "pubmed_retrieval_state.json"
    state = read_json(state_path) if state_path.exists() else {"completed_slices": {}, "completed_batches": {}}
    slices = date_range_half_years(registry["search_start"], registry["search_end"])
    total = 0
    for query in queries:
        for start, end in slices:
            total += retrieve_slice(query, start, end, creds, hcc_root, state, args.max_per_slice)
    print(json.dumps({"status": "complete", "query_count": len(queries), "slice_count": len(slices), "rows_retrieved_or_confirmed": total}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
