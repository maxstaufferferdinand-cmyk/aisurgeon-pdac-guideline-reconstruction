"""Run conservative, resumable PubMed E-utilities searches by guideline chapter.

Design goals
------------
* Fixed half-year publication-date slices from 2015-01-01 to 2023-08-31.
* Automatic recursive date splitting whenever a slice has >9,000 results.
* HTTP POST for the long PubMed queries and for EFetch batches.
* Deliberate low request rate (default: one request every 0.5 seconds).
* Small EFetch batches (default: 100 PMIDs).
* Incremental append to a UTF-8 CSV after every successful EFetch batch.
* Safe restart: existing (chapter_id, PMID) pairs are not downloaded twice.
* Full query registry, search log, and optional raw PubMed XML audit files.

Run from the project root, for example:
    uv run python src/run_pubmed_search.py --chapter 1 --count-only \
        --start-date 2015-01-01 --end-date 2015-06-30
    uv run python src/run_pubmed_search.py --chapter 1 \
        --start-date 2015-01-01 --end-date 2015-06-30
    uv run python src/run_pubmed_search.py --chapter all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import requests

from queries import CHAPTERS, ChapterQuery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "pubmed_results.csv"
DEFAULT_SEARCH_LOG = PROJECT_ROOT / "logs" / "search_log.csv"
DEFAULT_ERROR_LOG = PROJECT_ROOT / "logs" / "errors.jsonl"
DEFAULT_QUERY_REGISTRY = PROJECT_ROOT / "data" / "query_registry.json"
DEFAULT_RAW_XML_ROOT = PROJECT_ROOT / "data" / "raw_xml"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_START_DATE = date(2015, 1, 1)
DEFAULT_END_DATE = date(2023, 8, 31)
DEFAULT_FETCH_BATCH_SIZE = 100
DEFAULT_MAX_RECORDS_PER_SLICE = 9_000
DEFAULT_REQUEST_DELAY_SECONDS = 0.5
DEFAULT_TRANSIENT_RETRY_SECONDS = 120.0
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

OUTPUT_FIELDS = [
    "retrieved_at_utc",
    "chapter_id",
    "chapter_title",
    "slice_start",
    "slice_end",
    "query_sha256",
    "pmid",
    "doi",
    "pmcid",
    "pii",
    "title",
    "abstract",
    "authors",
    "affiliations",
    "journal",
    "journal_abbreviation",
    "issn",
    "publication_date",
    "electronic_date",
    "publication_year",
    "volume",
    "issue",
    "pages_or_elocation",
    "publication_types",
    "mesh_terms",
    "keywords",
    "languages",
    "country",
    "record_status",
]

SEARCH_LOG_FIELDS = [
    "run_id",
    "timestamp_utc",
    "mode",
    "chapter_id",
    "chapter_title",
    "slice_start",
    "slice_end",
    "esearch_count",
    "ids_returned",
    "ids_already_present",
    "new_rows_appended",
    "missing_pmids",
    "status",
    "query_sha256",
    "query_translation",
    "warnings_json",
    "message",
]


class NcbiError(RuntimeError):
    """Raised when an NCBI request or response is invalid after retries."""


@dataclass(slots=True)
class SearchMeta:
    count: int
    query_translation: str
    warnings: dict[str, Any]


@dataclass(slots=True)
class RunStats:
    leaf_slices: int = 0
    split_slices: int = 0
    esearch_count_total: int = 0
    ids_returned_total: int = 0
    existing_total: int = 0
    appended_total: int = 0
    missing_total: int = 0


class NcbiClient:
    """Small NCBI E-utilities client with POST, rate limiting, and retries."""

    def __init__(
        self,
        *,
        api_key: str,
        email: str,
        tool: str,
        request_delay_seconds: float,
        transient_retry_seconds: float = DEFAULT_TRANSIENT_RETRY_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.request_delay_seconds = request_delay_seconds
        self.transient_retry_seconds = transient_retry_seconds
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"{tool}/1.0 ({email})",
                "Accept": "application/json, application/xml, text/xml, */*",
            }
        )

    def close(self) -> None:
        self.session.close()

    def _common_params(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "email": self.email,
            "api_key": self.api_key,
        }

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _post(self, endpoint: str, data: dict[str, str]) -> requests.Response:
        """POST to NCBI with indefinite retries only for transient failures.

        Transient HTTP/server/network failures are retried every two minutes (or
        longer if NCBI explicitly sends a larger Retry-After value). Permanent
        client errors such as HTTP 400/401/403 fail immediately so a malformed
        query or invalid API key cannot loop forever unattended.
        """
        url = f"{EUTILS_BASE}/{endpoint}"
        payload = {**data, **self._common_params()}
        attempt = 0

        while True:
            attempt += 1
            self._wait_for_rate_limit()
            try:
                response = self.session.post(
                    url,
                    data=payload,
                    timeout=(30, 180),
                )
                self._last_request_at = time.monotonic()
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._last_request_at = time.monotonic()
                wait_seconds = self.transient_retry_seconds
                print(
                    f"  WARN: transient NCBI network error on {endpoint} "
                    f"({type(exc).__name__}: {exc}); attempt {attempt}; "
                    f"retrying in {wait_seconds:.0f}s",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            except requests.RequestException as exc:
                raise NcbiError(
                    f"Non-transient request error for {endpoint}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                wait_seconds = max(
                    self.transient_retry_seconds,
                    retry_after or 0.0,
                )
                snippet = _normalize_whitespace(response.text[:500])
                print(
                    f"  WARN: transient NCBI HTTP {response.status_code} on {endpoint}; "
                    f"attempt {attempt}; retrying in {wait_seconds:.0f}s"
                    + (f"; response: {snippet}" if snippet else ""),
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue

            if not response.ok:
                snippet = _normalize_whitespace(response.text[:1000])
                raise NcbiError(
                    f"NCBI HTTP {response.status_code} for {endpoint}; "
                    f"non-transient error, not retried. Response: {snippet!r}"
                )

            return response

    def esearch_count(self, query: str, start: date, end: date) -> SearchMeta:
        response = self._post(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "rettype": "count",
                "datetype": "pdat",
                "mindate": start.strftime("%Y/%m/%d"),
                "maxdate": end.strftime("%Y/%m/%d"),
            },
        )
        payload = _json_response(response, "ESearch count")
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            raise NcbiError(f"ESearch count response has no esearchresult: {payload}")
        if result.get("error"):
            raise NcbiError(f"ESearch count error: {result['error']}")

        try:
            count = int(result.get("count", "0"))
        except (TypeError, ValueError) as exc:
            raise NcbiError(f"Invalid ESearch count: {result.get('count')!r}") from exc

        return SearchMeta(
            count=count,
            query_translation=str(result.get("querytranslation", "")),
            warnings=_coerce_dict(result.get("warninglist")),
        )

    def esearch_ids(
        self,
        query: str,
        start: date,
        end: date,
        retmax: int,
    ) -> tuple[list[str], SearchMeta]:
        response = self._post(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "rettype": "uilist",
                "retstart": "0",
                "retmax": str(retmax),
                "sort": "pub_date",
                "datetype": "pdat",
                "mindate": start.strftime("%Y/%m/%d"),
                "maxdate": end.strftime("%Y/%m/%d"),
            },
        )
        payload = _json_response(response, "ESearch IDs")
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            raise NcbiError(f"ESearch ID response has no esearchresult: {payload}")
        if result.get("error"):
            raise NcbiError(f"ESearch ID error: {result['error']}")

        try:
            count = int(result.get("count", "0"))
        except (TypeError, ValueError) as exc:
            raise NcbiError(f"Invalid ESearch count in ID response: {result.get('count')!r}") from exc

        raw_ids = result.get("idlist", [])
        if not isinstance(raw_ids, list):
            raise NcbiError(f"Invalid ESearch idlist: {type(raw_ids).__name__}")
        ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        ids = list(dict.fromkeys(ids))

        return ids, SearchMeta(
            count=count,
            query_translation=str(result.get("querytranslation", "")),
            warnings=_coerce_dict(result.get("warninglist")),
        )

    def efetch_xml(self, pmids: Sequence[str]) -> bytes:
        if not pmids:
            raise ValueError("efetch_xml requires at least one PMID")
        response = self._post(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        content = response.content
        if not content.strip():
            raise NcbiError("EFetch returned an empty response")
        return content


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _json_response(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:500]
        raise NcbiError(f"{context}: invalid JSON response: {snippet!r}") from exc
    if not isinstance(payload, dict):
        raise NcbiError(f"{context}: JSON root is not an object")
    if payload.get("error"):
        raise NcbiError(f"{context}: {payload['error']}")
    return payload


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def _half_year_ranges(start: date, end: date) -> Iterator[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        if cursor.month <= 6:
            boundary = date(cursor.year, 6, 30)
        else:
            boundary = date(cursor.year, 12, 31)
        yield cursor, min(boundary, end)
        cursor = min(boundary, end) + timedelta(days=1)


def _split_date_range(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    if start >= end:
        raise NcbiError(
            f"A single publication day ({start.isoformat()}) still exceeds the safe "
            "ESearch threshold. Add an additional query subdivision; date splitting "
            "cannot reduce this slice further."
        )
    midpoint = start + timedelta(days=(end - start).days // 2)
    return (start, midpoint), (midpoint + timedelta(days=1), end)


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _normalize_whitespace("".join(element.itertext()))


def _find_text(root: ET.Element, path: str) -> str:
    return _element_text(root.find(path))


def _unique_join(values: Iterable[str], separator: str = "; ") -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = _normalize_whitespace(raw)
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return separator.join(ordered)


def _date_from_node(node: ET.Element | None) -> str:
    if node is None:
        return ""
    year = _find_text(node, "Year")
    month = _find_text(node, "Month")
    day = _find_text(node, "Day")
    medline_date = _find_text(node, "MedlineDate")
    season = _find_text(node, "Season")
    if year:
        parts = [year]
        if month:
            parts.append(month)
        if day:
            parts.append(day)
        if season:
            parts.append(season)
        return "-".join(parts)
    return medline_date or season


def _extract_year(*values: str) -> str:
    for value in values:
        match = re.search(r"\b(18|19|20|21)\d{2}\b", value)
        if match:
            return match.group(0)
    return ""


def _extract_article_ids(record: ET.Element) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for node in record.findall("./PubmedData/ArticleIdList/ArticleId"):
        id_type = (node.attrib.get("IdType") or "").strip().lower()
        value = _element_text(node)
        if id_type and value and id_type not in identifiers:
            identifiers[id_type] = value

    article = record.find("./MedlineCitation/Article")
    if article is not None:
        for node in article.findall("./ELocationID"):
            id_type = (node.attrib.get("EIdType") or "").strip().lower()
            value = _element_text(node)
            if id_type and value and id_type not in identifiers:
                identifiers[id_type] = value
    return identifiers


def _extract_authors(article: ET.Element | None) -> str:
    if article is None:
        return ""
    authors: list[str] = []
    for author in article.findall("./AuthorList/Author"):
        collective = _find_text(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        last = _find_text(author, "LastName")
        fore = _find_text(author, "ForeName")
        initials = _find_text(author, "Initials")
        suffix = _find_text(author, "Suffix")
        given = fore or initials
        name = " ".join(part for part in (given, last, suffix) if part)
        if name:
            authors.append(name)
    return _unique_join(authors)


def _extract_affiliations(article: ET.Element | None) -> str:
    if article is None:
        return ""
    values = (
        _element_text(node)
        for node in article.findall("./AuthorList/Author/AffiliationInfo/Affiliation")
    )
    return _unique_join(values, separator=" | ")


def _extract_abstract(article: ET.Element | None) -> str:
    if article is None:
        return ""
    sections: list[str] = []
    for node in article.findall("./Abstract/AbstractText"):
        text = _element_text(node)
        if not text:
            continue
        label = _normalize_whitespace(
            node.attrib.get("Label") or node.attrib.get("NlmCategory") or ""
        )
        if label and not text.casefold().startswith(label.casefold()):
            sections.append(f"{label}: {text}")
        else:
            sections.append(text)
    return _unique_join(sections, separator=" ")


def _extract_mesh(record: ET.Element) -> str:
    values: list[str] = []
    for heading in record.findall("./MedlineCitation/MeshHeadingList/MeshHeading"):
        descriptor_node = heading.find("DescriptorName")
        descriptor = _element_text(descriptor_node)
        if not descriptor:
            continue
        descriptor_major = (
            descriptor_node is not None
            and descriptor_node.attrib.get("MajorTopicYN") == "Y"
        )
        qualifiers: list[str] = []
        for qualifier_node in heading.findall("QualifierName"):
            qualifier = _element_text(qualifier_node)
            if not qualifier:
                continue
            if qualifier_node.attrib.get("MajorTopicYN") == "Y":
                qualifier = f"{qualifier}*"
            qualifiers.append(qualifier)
        label = f"{descriptor}{'*' if descriptor_major else ''}"
        if qualifiers:
            label = f"{label} / {', '.join(qualifiers)}"
        values.append(label)
    return _unique_join(values)


def _extract_keywords(record: ET.Element) -> str:
    return _unique_join(
        _element_text(node)
        for node in record.findall("./MedlineCitation/KeywordList/Keyword")
    )


def _parse_pubmed_article(record: ET.Element) -> dict[str, str]:
    citation = record.find("./MedlineCitation")
    article = record.find("./MedlineCitation/Article")
    journal = record.find("./MedlineCitation/Article/Journal")
    journal_issue = record.find("./MedlineCitation/Article/Journal/JournalIssue")

    pmid = _find_text(record, "./MedlineCitation/PMID")
    identifiers = _extract_article_ids(record)
    if not pmid:
        pmid = identifiers.get("pubmed", "")

    publication_date = _date_from_node(
        record.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    )
    electronic_date = _date_from_node(record.find("./MedlineCitation/Article/ArticleDate"))

    pagination = _find_text(record, "./MedlineCitation/Article/Pagination/MedlinePgn")
    if not pagination and article is not None:
        elocations = [
            _element_text(node)
            for node in article.findall("./ELocationID")
            if (node.attrib.get("EIdType") or "").lower() not in {"doi", "pii"}
        ]
        pagination = _unique_join(elocations)

    publication_types = _unique_join(
        _element_text(node)
        for node in record.findall(
            "./MedlineCitation/Article/PublicationTypeList/PublicationType"
        )
    )
    languages = _unique_join(
        _element_text(node)
        for node in record.findall("./MedlineCitation/Article/Language")
    )

    return {
        "pmid": pmid,
        "doi": identifiers.get("doi", ""),
        "pmcid": identifiers.get("pmc", ""),
        "pii": identifiers.get("pii", ""),
        "title": _find_text(record, "./MedlineCitation/Article/ArticleTitle"),
        "abstract": _extract_abstract(article),
        "authors": _extract_authors(article),
        "affiliations": _extract_affiliations(article),
        "journal": _find_text(record, "./MedlineCitation/Article/Journal/Title"),
        "journal_abbreviation": (
            _find_text(record, "./MedlineCitation/Article/Journal/ISOAbbreviation")
            or _find_text(record, "./MedlineCitation/MedlineJournalInfo/MedlineTA")
        ),
        "issn": _find_text(record, "./MedlineCitation/Article/Journal/ISSN"),
        "publication_date": publication_date,
        "electronic_date": electronic_date,
        "publication_year": _extract_year(publication_date, electronic_date),
        "volume": _find_text(journal_issue, "Volume") if journal_issue is not None else "",
        "issue": _find_text(journal_issue, "Issue") if journal_issue is not None else "",
        "pages_or_elocation": pagination,
        "publication_types": publication_types,
        "mesh_terms": _extract_mesh(record),
        "keywords": _extract_keywords(record),
        "languages": languages,
        "country": _find_text(record, "./MedlineCitation/MedlineJournalInfo/Country"),
        "record_status": "ok",
    }


def _parse_pubmed_book_article(record: ET.Element) -> dict[str, str]:
    book_document = record.find("./BookDocument")
    if book_document is None:
        return {}
    pmid = _find_text(record, "./BookDocument/PMID")
    title = _find_text(record, "./BookDocument/ArticleTitle")
    abstract_sections = [
        _element_text(node) for node in book_document.findall("./Abstract/AbstractText")
    ]
    pub_date = _date_from_node(book_document.find("./Book/PubDate"))
    return {
        "pmid": pmid,
        "doi": "",
        "pmcid": "",
        "pii": "",
        "title": title,
        "abstract": _unique_join(abstract_sections, separator=" "),
        "authors": _extract_authors(book_document),
        "affiliations": _extract_affiliations(book_document),
        "journal": _find_text(book_document, "./Book/BookTitle"),
        "journal_abbreviation": "",
        "issn": "",
        "publication_date": pub_date,
        "electronic_date": "",
        "publication_year": _extract_year(pub_date),
        "volume": "",
        "issue": "",
        "pages_or_elocation": "",
        "publication_types": "PubMed Book Article",
        "mesh_terms": "",
        "keywords": _unique_join(
            _element_text(node) for node in book_document.findall("./KeywordList/Keyword")
        ),
        "languages": _unique_join(
            _element_text(node) for node in book_document.findall("./Language")
        ),
        "country": "",
        "record_status": "pubmed_book_article",
    }


def parse_pubmed_xml(content: bytes) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        snippet = content[:500].decode("utf-8", errors="replace")
        raise NcbiError(f"Invalid PubMed XML: {snippet!r}") from exc

    errors = [_element_text(node) for node in root.findall(".//ERROR")]
    if errors:
        raise NcbiError(f"EFetch XML error: {' | '.join(errors)}")

    rows: list[dict[str, str]] = []
    for record in root.findall("./PubmedArticle"):
        row = _parse_pubmed_article(record)
        if row.get("pmid"):
            rows.append(row)
    for record in root.findall("./PubmedBookArticle"):
        row = _parse_pubmed_book_article(record)
        if row.get("pmid"):
            rows.append(row)

    deduplicated: dict[str, dict[str, str]] = {}
    for row in rows:
        deduplicated.setdefault(row["pmid"], row)
    return list(deduplicated.values())


def _validate_existing_csv(path: Path, expected_fields: Sequence[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return
    if list(header) != list(expected_fields):
        raise RuntimeError(
            f"CSV schema mismatch in {path}.\n"
            f"Expected: {list(expected_fields)}\n"
            f"Found:    {header}\n"
            "Rename or remove the old CSV before running this version."
        )


def _append_csv_rows(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    encoding = "utf-8-sig" if is_new else "utf-8"
    with path.open("a", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _load_seen_pairs(path: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not path.exists() or path.stat().st_size == 0:
        return seen
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "chapter_id" not in (reader.fieldnames or []) or "pmid" not in (reader.fieldnames or []):
            raise RuntimeError(f"Cannot resume from {path}: chapter_id/pmid columns missing")
        for row in reader:
            chapter_id = (row.get("chapter_id") or "").strip()
            pmid = (row.get("pmid") or "").strip()
            if chapter_id and pmid:
                seen.add((chapter_id, pmid))
    return seen


def _append_search_log(path: Path, row: dict[str, Any]) -> None:
    _append_csv_rows(path, SEARCH_LOG_FIELDS, [row])


def _append_error_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_query_registry(
    path: Path,
    chapters: Sequence[ChapterQuery],
    start: date,
    end: date,
    max_records_per_slice: int,
    fetch_batch_size: int,
    request_delay_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "database": "PubMed",
        "date_type": "pdat",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "base_batching": "calendar half-years",
        "automatic_split_threshold": max_records_per_slice,
        "efetch_batch_size": fetch_batch_size,
        "request_delay_seconds": request_delay_seconds,
        "chapters": [
            {
                "chapter_id": chapter.chapter_id,
                "title": chapter.title,
                "query_sha256": _query_sha256(chapter.query),
                "query": chapter.query,
            }
            for chapter in chapters
        ],
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _save_raw_xml(
    root: Path,
    chapter_id: str,
    start: date,
    end: date,
    batch_index: int,
    pmids: Sequence[str],
    content: bytes,
) -> Path:
    chapter_dir = root / f"chapter_{chapter_id.replace('.', '_')}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{start.isoformat()}_{end.isoformat()}_"
        f"batch_{batch_index:05d}_{pmids[0]}_{pmids[-1]}.xml"
    )
    target = chapter_dir / filename
    temp = target.with_suffix(".xml.tmp")
    temp.write_bytes(content)
    temp.replace(target)
    return target


def _base_log_row(
    *,
    run_id: str,
    mode: str,
    chapter: ChapterQuery,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now_iso(),
        "mode": mode,
        "chapter_id": chapter.chapter_id,
        "chapter_title": chapter.title,
        "slice_start": start.isoformat(),
        "slice_end": end.isoformat(),
        "esearch_count": "",
        "ids_returned": "",
        "ids_already_present": "",
        "new_rows_appended": "",
        "missing_pmids": "",
        "status": "",
        "query_sha256": _query_sha256(chapter.query),
        "query_translation": "",
        "warnings_json": "",
        "message": "",
    }


def process_date_range(
    *,
    client: NcbiClient,
    chapter: ChapterQuery,
    start: date,
    end: date,
    run_id: str,
    count_only: bool,
    max_records_per_slice: int,
    fetch_batch_size: int,
    output_path: Path,
    search_log_path: Path,
    error_log_path: Path,
    raw_xml_root: Path,
    save_raw_xml: bool,
    seen_pairs: set[tuple[str, str]],
    stats: RunStats,
    depth: int = 0,
) -> None:
    indent = "  " * depth
    query_hash = _query_sha256(chapter.query)
    mode = "count_only" if count_only else "fetch"

    print(
        f"{indent}[{chapter.chapter_id}] {start.isoformat()} .. {end.isoformat()} — counting",
        flush=True,
    )
    try:
        meta = client.esearch_count(chapter.query, start, end)
    except Exception as exc:
        _append_error_log(
            error_log_path,
            {
                "timestamp_utc": _utc_now_iso(),
                "run_id": run_id,
                "stage": "esearch_count",
                "chapter_id": chapter.chapter_id,
                "slice_start": start.isoformat(),
                "slice_end": end.isoformat(),
                "query_sha256": query_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    print(f"{indent}  count = {meta.count:,}", flush=True)

    if meta.count > max_records_per_slice:
        stats.split_slices += 1
        log_row = _base_log_row(
            run_id=run_id,
            mode=mode,
            chapter=chapter,
            start=start,
            end=end,
        )
        log_row.update(
            {
                "esearch_count": meta.count,
                "status": "split",
                "query_translation": meta.query_translation,
                "warnings_json": json.dumps(meta.warnings, ensure_ascii=False, sort_keys=True),
                "message": (
                    f"Count exceeds safe threshold {max_records_per_slice}; "
                    "date range split recursively"
                ),
            }
        )
        _append_search_log(search_log_path, log_row)

        left, right = _split_date_range(start, end)
        print(
            f"{indent}  SPLIT -> {left[0]}..{left[1]} and {right[0]}..{right[1]}",
            flush=True,
        )
        process_date_range(
            client=client,
            chapter=chapter,
            start=left[0],
            end=left[1],
            run_id=run_id,
            count_only=count_only,
            max_records_per_slice=max_records_per_slice,
            fetch_batch_size=fetch_batch_size,
            output_path=output_path,
            search_log_path=search_log_path,
            error_log_path=error_log_path,
            raw_xml_root=raw_xml_root,
            save_raw_xml=save_raw_xml,
            seen_pairs=seen_pairs,
            stats=stats,
            depth=depth + 1,
        )
        process_date_range(
            client=client,
            chapter=chapter,
            start=right[0],
            end=right[1],
            run_id=run_id,
            count_only=count_only,
            max_records_per_slice=max_records_per_slice,
            fetch_batch_size=fetch_batch_size,
            output_path=output_path,
            search_log_path=search_log_path,
            error_log_path=error_log_path,
            raw_xml_root=raw_xml_root,
            save_raw_xml=save_raw_xml,
            seen_pairs=seen_pairs,
            stats=stats,
            depth=depth + 1,
        )
        return

    stats.leaf_slices += 1
    stats.esearch_count_total += meta.count

    if count_only:
        log_row = _base_log_row(
            run_id=run_id,
            mode=mode,
            chapter=chapter,
            start=start,
            end=end,
        )
        log_row.update(
            {
                "esearch_count": meta.count,
                "status": "counted",
                "query_translation": meta.query_translation,
                "warnings_json": json.dumps(meta.warnings, ensure_ascii=False, sort_keys=True),
                "message": "Safe leaf slice; no records fetched",
            }
        )
        _append_search_log(search_log_path, log_row)
        return

    if meta.count == 0:
        log_row = _base_log_row(
            run_id=run_id,
            mode=mode,
            chapter=chapter,
            start=start,
            end=end,
        )
        log_row.update(
            {
                "esearch_count": 0,
                "ids_returned": 0,
                "ids_already_present": 0,
                "new_rows_appended": 0,
                "missing_pmids": 0,
                "status": "completed_empty",
                "query_translation": meta.query_translation,
                "warnings_json": json.dumps(meta.warnings, ensure_ascii=False, sort_keys=True),
                "message": "No PubMed records in this date slice",
            }
        )
        _append_search_log(search_log_path, log_row)
        return

    try:
        pmids, id_meta = client.esearch_ids(
            chapter.query,
            start,
            end,
            retmax=max_records_per_slice,
        )
    except Exception as exc:
        _append_error_log(
            error_log_path,
            {
                "timestamp_utc": _utc_now_iso(),
                "run_id": run_id,
                "stage": "esearch_ids",
                "chapter_id": chapter.chapter_id,
                "slice_start": start.isoformat(),
                "slice_end": end.isoformat(),
                "query_sha256": query_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    if id_meta.count > max_records_per_slice:
        # PubMed may have changed between the count and ID calls. Do not risk truncation.
        print(
            f"{indent}  Count increased to {id_meta.count:,}; splitting instead of fetching",
            flush=True,
        )
        left, right = _split_date_range(start, end)
        process_date_range(
            client=client,
            chapter=chapter,
            start=left[0],
            end=left[1],
            run_id=run_id,
            count_only=False,
            max_records_per_slice=max_records_per_slice,
            fetch_batch_size=fetch_batch_size,
            output_path=output_path,
            search_log_path=search_log_path,
            error_log_path=error_log_path,
            raw_xml_root=raw_xml_root,
            save_raw_xml=save_raw_xml,
            seen_pairs=seen_pairs,
            stats=stats,
            depth=depth + 1,
        )
        process_date_range(
            client=client,
            chapter=chapter,
            start=right[0],
            end=right[1],
            run_id=run_id,
            count_only=False,
            max_records_per_slice=max_records_per_slice,
            fetch_batch_size=fetch_batch_size,
            output_path=output_path,
            search_log_path=search_log_path,
            error_log_path=error_log_path,
            raw_xml_root=raw_xml_root,
            save_raw_xml=save_raw_xml,
            seen_pairs=seen_pairs,
            stats=stats,
            depth=depth + 1,
        )
        return

    if len(pmids) != id_meta.count:
        raise NcbiError(
            f"ESearch returned {len(pmids)} unique PMIDs, but reported "
            f"count={id_meta.count} for {chapter.chapter_id}, "
            f"{start.isoformat()}..{end.isoformat()}. Stopping to avoid silent loss."
        )

    stats.ids_returned_total += len(pmids)
    new_pmids = [
        pmid for pmid in pmids if (chapter.chapter_id, pmid) not in seen_pairs
    ]
    already_present = len(pmids) - len(new_pmids)
    stats.existing_total += already_present

    print(
        f"{indent}  IDs = {len(pmids):,}; already in CSV = {already_present:,}; "
        f"to fetch = {len(new_pmids):,}",
        flush=True,
    )

    appended = 0
    missing_pmids: list[str] = []
    total_batches = (len(new_pmids) + fetch_batch_size - 1) // fetch_batch_size

    for batch_index, batch in enumerate(_chunks(new_pmids, fetch_batch_size), start=1):
        print(
            f"{indent}  EFetch batch {batch_index}/{total_batches} "
            f"({len(batch)} PMIDs)",
            flush=True,
        )
        try:
            xml_content = client.efetch_xml(batch)
            if save_raw_xml:
                raw_path = _save_raw_xml(
                    raw_xml_root,
                    chapter.chapter_id,
                    start,
                    end,
                    batch_index,
                    batch,
                    xml_content,
                )
                print(f"{indent}    raw XML: {raw_path}", flush=True)
            parsed_rows = parse_pubmed_xml(xml_content)
        except Exception as exc:
            _append_error_log(
                error_log_path,
                {
                    "timestamp_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "stage": "efetch_or_parse",
                    "chapter_id": chapter.chapter_id,
                    "slice_start": start.isoformat(),
                    "slice_end": end.isoformat(),
                    "batch_index": batch_index,
                    "pmids": batch,
                    "query_sha256": query_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        parsed_by_pmid = {row["pmid"]: row for row in parsed_rows if row.get("pmid")}
        missing_in_batch = [pmid for pmid in batch if pmid not in parsed_by_pmid]
        missing_pmids.extend(missing_in_batch)

        enriched_rows: list[dict[str, Any]] = []
        retrieved_at = _utc_now_iso()
        for pmid in batch:
            article_row = parsed_by_pmid.get(pmid)
            if article_row is None:
                continue
            pair = (chapter.chapter_id, pmid)
            if pair in seen_pairs:
                continue
            enriched_rows.append(
                {
                    "retrieved_at_utc": retrieved_at,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "slice_start": start.isoformat(),
                    "slice_end": end.isoformat(),
                    "query_sha256": query_hash,
                    **article_row,
                }
            )

        _append_csv_rows(output_path, OUTPUT_FIELDS, enriched_rows)
        for row in enriched_rows:
            seen_pairs.add((chapter.chapter_id, row["pmid"]))
        appended += len(enriched_rows)
        stats.appended_total += len(enriched_rows)

        if missing_in_batch:
            print(
                f"{indent}    WARN: {len(missing_in_batch)} PMID(s) absent from EFetch: "
                f"{','.join(missing_in_batch)}",
                flush=True,
            )

    stats.missing_total += len(missing_pmids)
    log_row = _base_log_row(
        run_id=run_id,
        mode=mode,
        chapter=chapter,
        start=start,
        end=end,
    )
    combined_warnings = {**meta.warnings, **id_meta.warnings}
    log_row.update(
        {
            "esearch_count": id_meta.count,
            "ids_returned": len(pmids),
            "ids_already_present": already_present,
            "new_rows_appended": appended,
            "missing_pmids": ";".join(missing_pmids),
            "status": "completed" if not missing_pmids else "completed_with_missing_pmids",
            "query_translation": id_meta.query_translation or meta.query_translation,
            "warnings_json": json.dumps(combined_warnings, ensure_ascii=False, sort_keys=True),
            "message": "Incremental CSV append completed",
        }
    )
    _append_search_log(search_log_path, log_row)


def _select_chapters(value: str) -> list[ChapterQuery]:
    if value == "all":
        return list(CHAPTERS.values())
    return [CHAPTERS[value]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search PubMed for the ESMO PDAC PoC in half-year date slices, "
            "automatically split slices above the safe ESearch threshold, and "
            "append PubMed records incrementally to CSV."
        )
    )
    parser.add_argument(
        "--chapter",
        choices=["all", *CHAPTERS.keys()],
        default="all",
        help="Guideline chapter to run; default: all",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=DEFAULT_START_DATE,
        help=f"Publication-date start (YYYY-MM-DD); default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=DEFAULT_END_DATE,
        help=f"Publication-date end (YYYY-MM-DD); default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count and split ranges; do not retrieve PMIDs or article records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path; default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=DEFAULT_FETCH_BATCH_SIZE,
        help=f"PMIDs per EFetch POST; default: {DEFAULT_FETCH_BATCH_SIZE}",
    )
    parser.add_argument(
        "--max-records-per-slice",
        type=int,
        default=DEFAULT_MAX_RECORDS_PER_SLICE,
        help=(
            "Date slices above this count are split recursively; "
            f"default: {DEFAULT_MAX_RECORDS_PER_SLICE}"
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help=(
            "Minimum seconds between NCBI requests; must be >=0.34; "
            f"default: {DEFAULT_REQUEST_DELAY_SECONDS}"
        ),
    )
    parser.add_argument(
        "--transient-retry-wait",
        type=float,
        default=DEFAULT_TRANSIENT_RETRY_SECONDS,
        help=(
            "Seconds to wait before retrying transient HTTP/network failures; "
            f"default: {DEFAULT_TRANSIENT_RETRY_SECONDS:.0f}"
        ),
    )
    parser.add_argument(
        "--no-raw-xml",
        action="store_true",
        help="Do not save the raw PubMed EFetch XML audit files",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if not 1 <= args.fetch_batch_size <= 200:
        parser.error("--fetch-batch-size must be between 1 and 200")
    if not 1 <= args.max_records_per_slice <= 10_000:
        parser.error("--max-records-per-slice must be between 1 and 10000")
    if args.request_delay < 0.34:
        parser.error("--request-delay must be >=0.34 seconds")
    if args.transient_retry_wait < 1:
        parser.error("--transient-retry-wait must be >=1 second")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "Set it in PowerShell before starting the search."
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        api_key = _required_env("NCBI_API_KEY")
        email = _required_env("NCBI_EMAIL")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    tool = os.environ.get("NCBI_TOOL", "AISurgeon_PDCA_PDAC_PoC").strip()
    if not tool or any(character.isspace() for character in tool):
        print(
            "ERROR: NCBI_TOOL must be non-empty and contain no whitespace.",
            file=sys.stderr,
        )
        return 2

    chapters = _select_chapters(args.chapter)
    output_path = args.output.resolve()
    search_log_path = DEFAULT_SEARCH_LOG.resolve()
    error_log_path = DEFAULT_ERROR_LOG.resolve()
    query_registry_path = DEFAULT_QUERY_REGISTRY.resolve()
    raw_xml_root = DEFAULT_RAW_XML_ROOT.resolve()

    _validate_existing_csv(output_path, OUTPUT_FIELDS)
    _validate_existing_csv(search_log_path, SEARCH_LOG_FIELDS)
    seen_pairs = _load_seen_pairs(output_path)

    _write_query_registry(
        query_registry_path,
        chapters,
        args.start_date,
        args.end_date,
        args.max_records_per_slice,
        args.fetch_batch_size,
        args.request_delay,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stats = RunStats()
    client = NcbiClient(
        api_key=api_key,
        email=email,
        tool=tool,
        request_delay_seconds=args.request_delay,
        transient_retry_seconds=args.transient_retry_wait,
    )

    print("\nESMO PDAC PubMed PoC", flush=True)
    print(f"  run_id:             {run_id}", flush=True)
    print(f"  chapters:           {', '.join(ch.chapter_id for ch in chapters)}", flush=True)
    print(f"  publication dates:  {args.start_date} .. {args.end_date}", flush=True)
    print("  base date slices:   calendar half-years", flush=True)
    print(f"  split threshold:    {args.max_records_per_slice:,}", flush=True)
    print(f"  EFetch batch size:  {args.fetch_batch_size}", flush=True)
    print(f"  request delay:      {args.request_delay:.2f}s", flush=True)
    print(f"  transient retry:    {args.transient_retry_wait:.0f}s, unlimited", flush=True)
    print(f"  API key loaded:     yes (not printed)", flush=True)
    print(f"  NCBI email:         {email}", flush=True)
    print(f"  NCBI tool:          {tool}", flush=True)
    print(f"  output CSV:         {output_path}", flush=True)
    print(f"  existing rows seen: {len(seen_pairs):,} chapter/PMID pairs", flush=True)
    print(f"  mode:               {'count only' if args.count_only else 'fetch and append'}\n", flush=True)

    try:
        for chapter in chapters:
            print(f"\n=== Chapter {chapter.chapter_id}: {chapter.title} ===", flush=True)
            for slice_start, slice_end in _half_year_ranges(args.start_date, args.end_date):
                process_date_range(
                    client=client,
                    chapter=chapter,
                    start=slice_start,
                    end=slice_end,
                    run_id=run_id,
                    count_only=args.count_only,
                    max_records_per_slice=args.max_records_per_slice,
                    fetch_batch_size=args.fetch_batch_size,
                    output_path=output_path,
                    search_log_path=search_log_path,
                    error_log_path=error_log_path,
                    raw_xml_root=raw_xml_root,
                    save_raw_xml=not args.no_raw_xml,
                    seen_pairs=seen_pairs,
                    stats=stats,
                )
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. Completed EFetch batches are already in the CSV; "
            "rerun the same command to resume safely.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}\n"
            "The run stopped deliberately to avoid silent data loss. Completed batches "
            "remain in the CSV; fix the error and rerun the same command to resume.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        client.close()

    print("\nRun completed.", flush=True)
    print(f"  safe leaf slices:      {stats.leaf_slices:,}", flush=True)
    print(f"  ranges auto-split:     {stats.split_slices:,}", flush=True)
    print(f"  ESearch count total:   {stats.esearch_count_total:,}", flush=True)
    print(f"  PMIDs returned total:  {stats.ids_returned_total:,}", flush=True)
    print(f"  already present:       {stats.existing_total:,}", flush=True)
    print(f"  rows newly appended:   {stats.appended_total:,}", flush=True)
    print(f"  PMIDs missing EFetch:  {stats.missing_total:,}", flush=True)
    print(f"  CSV:                   {output_path}", flush=True)
    print(f"  search log:            {search_log_path}", flush=True)
    print(f"  query registry:        {query_registry_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
