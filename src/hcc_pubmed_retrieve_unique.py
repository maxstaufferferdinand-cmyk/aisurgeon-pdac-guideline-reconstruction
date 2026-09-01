from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from hcc_pubmed_retrieve import (
    DEFAULT_HCC_ROOT,
    FIELDNAMES,
    date_range_half_years,
    efetch,
    esearch,
    ncbi_params,
    parse_articles,
    split_range,
)


PROVENANCE_FIELDS = ["pmid", "query_id", "chapter_id", "unit_id", "slice_start", "slice_end"]
STATE_PATH = Path("run_state/pubmed_unique_retrieval_state_v2.json")
PROVENANCE_PATH = Path("data/pubmed_query_provenance_v2.csv")
UNIQUE_RECORDS_PATH = Path("data/pubmed_unique_records_v2.csv")
RAW_XML_DIR = Path("data/raw_xml_unique_v2")
SUMMARY_PATH = Path("data/pubmed_retrieval_summary_v2.json")


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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists or not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def collect_slice_pmids(
    query: dict[str, Any],
    start: str,
    end: str,
    creds: dict[str, str],
    hcc_root: Path,
    state: dict[str, Any],
    max_per_slice: int,
) -> set[str]:
    key = f"{query['query_id']}|{start}|{end}"
    if key in state.get("completed_esearch_slices", {}):
        return set(state["completed_esearch_slices"][key].get("pmids", []))
    count, pmids = esearch(query["query"], start, end, creds)
    append_jsonl(
        hcc_root / "logs" / "pubmed_esearch_provenance_log.jsonl",
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
            raise RuntimeError(f"PubMed ESearch slice too large and cannot split further: {key} count={count}")
        found: set[str] = set()
        for part_start, part_end in split:
            found.update(collect_slice_pmids(query, part_start, part_end, creds, hcc_root, state, max_per_slice))
        return found
    rows = [
        {
            "pmid": pmid,
            "query_id": query["query_id"],
            "chapter_id": query["chapter_id"],
            "unit_id": query["unit_id"],
            "slice_start": start,
            "slice_end": end,
        }
        for pmid in pmids
    ]
    if rows:
        write_csv(hcc_root / PROVENANCE_PATH, PROVENANCE_FIELDS, rows)
    state.setdefault("completed_esearch_slices", {})[key] = {
        "count": count,
        "pmids": pmids,
        "completed_at": utc_now(),
    }
    atomic_write_json(hcc_root / STATE_PATH, state)
    time.sleep(0.11)
    return set(pmids)


def collect_all_pmids(hcc_root: Path, queries: list[dict[str, Any]], creds: dict[str, str], max_per_slice: int) -> set[str]:
    state_path = hcc_root / STATE_PATH
    state = read_json(state_path) if state_path.exists() else {"completed_esearch_slices": {}, "completed_efetch_batches": {}}
    registry = read_json(hcc_root / "data" / "pubmed_query_registry.json")
    slices = date_range_half_years(registry["search_start"], registry["search_end"])
    all_pmids: set[str] = set()
    for query in queries:
        for start, end in slices:
            all_pmids.update(collect_slice_pmids(query, start, end, creds, hcc_root, state, max_per_slice))
    return all_pmids


def fetch_unique_pmids(hcc_root: Path, pmids: list[str], creds: dict[str, str]) -> int:
    state_path = hcc_root / STATE_PATH
    state = read_json(state_path) if state_path.exists() else {"completed_efetch_batches": {}}
    raw_dir = hcc_root / RAW_XML_DIR
    csv_path = hcc_root / UNIQUE_RECORDS_PATH
    fetched = 0
    for batch_start in range(0, len(pmids), 200):
        batch_pmids = pmids[batch_start : batch_start + 200]
        batch_key = f"unique|{batch_start}"
        if batch_key in state.get("completed_efetch_batches", {}):
            fetched += int(state["completed_efetch_batches"][batch_key].get("count", 0))
            continue
        xml = efetch(batch_pmids, creds)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"unique_batch_{batch_start:06d}.xml"
        temp = raw_path.with_suffix(".xml.tmp")
        temp.write_bytes(xml)
        temp.replace(raw_path)
        rows = parse_articles(
            xml,
            str(raw_path),
            {
                "query_id": "UNIQUE_PMID_EFETCH",
                "chapter_id": "",
                "unit_id": "",
                "slice_start": "",
                "slice_end": "",
            },
        )
        write_csv(csv_path, FIELDNAMES, rows)
        fetched += len(rows)
        state.setdefault("completed_efetch_batches", {})[batch_key] = {
            "count": len(rows),
            "completed_at": utc_now(),
            "raw_xml_file": str(raw_path),
        }
        atomic_write_json(state_path, state)
        time.sleep(0.11)
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve PubMed query provenance and fetch each unique PMID once.")
    parser.add_argument("--hcc-root", default=os.environ.get("HCC_ROOT", str(DEFAULT_HCC_ROOT)))
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
    pmids = sorted(collect_all_pmids(hcc_root, registry["queries"], creds, args.max_per_slice), key=int)
    fetched = fetch_unique_pmids(hcc_root, pmids, creds)
    summary = {
        "status": "complete",
        "query_count": len(registry["queries"]),
        "unique_pmids": len(pmids),
        "unique_records_fetched_or_confirmed": fetched,
    }
    atomic_write_json(hcc_root / SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
