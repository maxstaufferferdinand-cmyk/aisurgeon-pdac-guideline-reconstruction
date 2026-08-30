#!/usr/bin/env python3
"""
Recover unresolved ESMO-PDAC evidence mappings.

Inputs
------
data/questionable_chapter_assignments.jsonl
data/novel_topic_records.jsonl
data/evidence_unit_ontology.json

Questionable records:
- remap from scratch to one or more of the 8 major chapters, or OUT_OF_SCOPE.

Novel-topic records:
- re-check for an existing unit in the current chapter;
- otherwise mark NEW_SUBUNIT_CANDIDATE,
  NEW_MAJOR_CHAPTER_CANDIDATE, or OUT_OF_SCOPE.

No evidence appraisal or guideline rewriting occurs here.

v2 fix:
- Batch custom_id is unique per paper-chapter assignment.
- Questionable records are preserved by (PMID, original chapter), not PMID alone.
- Preparation hard-fails locally if any duplicate custom_id remains.
"""

from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path
from collections import Counter
from typing import Any
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"

Q_IN = DATA / "questionable_chapter_assignments.jsonl"
N_IN = DATA / "novel_topic_records.jsonl"
ONTOLOGY = DATA / "evidence_unit_ontology.json"

BATCH_IN = DATA / "gpt_recovery_batch_input.jsonl"
BATCH_OUT = DATA / "gpt_recovery_batch_output.jsonl"
BATCH_ERR = DATA / "gpt_recovery_batch_errors.jsonl"
STATE = DATA / "gpt_recovery_state.json"

OUT_Q = DATA / "recovery_questionable_major_chapters.jsonl"
OUT_N = DATA / "recovery_novel_topic_triage.jsonl"
OUT_EXISTING = DATA / "recovery_existing_unit_assignments.jsonl"
OUT_NEW_SUB = DATA / "recovery_new_subunit_candidates.jsonl"
OUT_NEW_MAJOR = DATA / "recovery_new_major_chapter_candidates.jsonl"
OUT_OOS = DATA / "recovery_out_of_scope.jsonl"
OUT_FAIL = DATA / "recovery_parse_failures.jsonl"
MANIFEST = DATA / "recovery_manifest.json"

API = "https://api.openai.com/v1"
TERMINAL = {"completed", "failed", "expired", "cancelled"}

CHAPTERS = {
    "1": ("Incidence and epidemiology",
          "Incidence, mortality, population survival, demographics/geography, inherited/familial risk, high-risk screening, and etiologic/risk-factor epidemiology."),
    "2": ("Diagnosis and pathology/molecular biology",
          "Clinical presentation/recognition, histopathology, precursor lesions, tumour biology, genomics and molecular pathogenesis when not primarily treatment-selection questions."),
    "3": ("Staging and risk assessment",
          "TNM, CA19-9 for burden/prognosis, staging imaging, EUS/biopsy, metastasis/nodal detection, vascular involvement, resectability and pre-treatment risk assessment."),
    "4.1": ("Treatment of localised disease",
            "Resectable/localised disease: curative surgery, margins, lymphadenectomy, perioperative risk, preoperative biliary drainage, adjuvant and perioperative/neoadjuvant therapy."),
    "4.2": ("Treatment of non-resectable disease – borderline resectable / locally advanced",
            "Borderline-resectable or locally advanced non-metastatic disease: induction/neoadjuvant systemic therapy, radiotherapy, conversion surgery and local therapies."),
    "4.3": ("Treatment of advanced/metastatic disease",
            "Advanced/metastatic disease: palliative/supportive care, systemic first/later-line therapy, sequencing, response assessment and advanced rare exocrine cancers."),
    "5": ("Personalised medicine",
          "Treatment-directed biomarkers, germline/somatic testing, precision oncology, actionable alterations, biomarker-selected therapy and treatment-directed liquid biopsy."),
    "6": ("Follow-up and long-term implications",
          "Post-curative surveillance, recurrence detection, follow-up imaging/biomarkers, surveillance intervals, survivorship and long-term consequences."),
}

Q_SYS = """You are correcting a major-chapter assignment in a medical clinical-practice-guideline evidence-update workflow.

Re-evaluate the publication from scratch across the eight predefined ESMO pancreatic-cancer chapters.

Rules:
- Map substantive scientific content, not background mentions.
- Assign the smallest set of major chapters genuinely needed.
- Multiple chapters are allowed only for genuinely multidomain work.
- If no chapter is appropriate, choose out_of_scope.
- Do not assess evidence quality, efficacy, risk of bias, or guideline change.
- Do not create a new major chapter in this task.
- Use only title/abstract and supplied metadata.
Return only schema-valid JSON.
"""

N_SYS = """You are recovering novel-topic evidence for a medical clinical-practice-guideline update.

The publication already has a major chapter but did not fit a predefined evidence unit. Decide whether it:
1) actually fits one or more existing evidence units in that chapter;
2) is a coherent new_subunit_candidate within that chapter;
3) is an exceptional new_major_chapter_candidate;
4) is out_of_scope.

Rules:
- Prefer an existing unit when it genuinely fits.
- New terminology alone does not justify a new unit.
- new_subunit_candidate requires a medically coherent topic not adequately represented by existing units.
- new_major_chapter_candidate is exceptional and should be used only for a structurally distinct guideline domain that could plausibly organize multiple evidence questions. One niche paper is not enough to establish a chapter; this only flags a candidate for later cross-paper clustering.
- Do not assess evidence quality, treatment efficacy, recommendation strength, or guideline change.
- Use only title/abstract and supplied metadata.
Return only schema-valid JSON.
"""

def clean(v: Any) -> str:
    return " ".join(str(v or "").replace("\ufeff", "").split())

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"{path.name} line {n}: {e}") from e
    return rows

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def article(rec):
    return (
        f"PMID: {clean(rec.get('pmid'))}\n"
        f"Title: {clean(rec.get('title')) or '[missing]'}\n"
        f"Abstract: {clean(rec.get('abstract')) or '[no abstract available]'}\n"
        f"Publication types: {clean(rec.get('publication_types')) or '[not available]'}\n"
        f"Evidence labels: {clean(rec.get('evidence_labels')) or '[not available]'}\n"
        f"MeSH terms: {clean(rec.get('mesh_terms')) or '[not available]'}\n"
        f"Keywords: {clean(rec.get('keywords')) or '[not available]'}\n"
        f"Publication year: {clean(rec.get('publication_year')) or '[not available]'}"
    )

def major_prompt():
    parts = ["ALLOWED MAJOR CHAPTERS:"]
    for cid, (title, definition) in CHAPTERS.items():
        parts += ["", f"{cid} — {title}", f"Definition: {definition}"]
    return "\n".join(parts)

def unit_prompt(ontology, cid):
    ch = ontology["chapters"][cid]
    parts = [f"CURRENT MAJOR CHAPTER: {cid} — {ch['title']}", "",
             "EXISTING EVIDENCE UNITS:"]
    for u in ch["evidence_units"]:
        parts += ["", f"{u['id']} — {u['name']}",
                  f"Definition: {u['definition']}",
                  f"Boundary: {u['boundary']}"]
    return "\n".join(parts)

def q_schema():
    ids = list(CHAPTERS)
    return {
        "name": "questionable_major_chapter_recovery",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string",
                             "enum": ["mapped_to_existing_chapter", "out_of_scope"]},
                "chapter_ids": {"type": "array",
                                "items": {"type": "string", "enum": ids},
                                "minItems": 0, "maxItems": 3},
                "confidence": {"type": "string",
                               "enum": ["high", "medium", "low"]},
                "rationale": {"type": "string"},
            },
            "required": ["decision", "chapter_ids", "confidence", "rationale"],
            "additionalProperties": False,
        },
    }

def n_schema(unit_ids):
    return {
        "name": "novel_topic_recovery",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string",
                             "enum": ["existing_unit", "new_subunit_candidate",
                                      "new_major_chapter_candidate", "out_of_scope"]},
                "existing_unit_ids": {"type": "array",
                                      "items": {"type": "string", "enum": unit_ids},
                                      "minItems": 0, "maxItems": 6},
                "candidate_title": {"type": "string"},
                "candidate_description": {"type": "string"},
                "confidence": {"type": "string",
                               "enum": ["high", "medium", "low"]},
                "rationale": {"type": "string"},
            },
            "required": ["decision", "existing_unit_ids", "candidate_title",
                         "candidate_description", "confidence", "rationale"],
            "additionalProperties": False,
        },
    }

def prepare(model, effort):
    ontology = load_json(ONTOLOGY)
    qs = load_jsonl(Q_IN)
    ns = load_jsonl(N_IN)
    unit_ids = {cid: [u["id"] for u in ch["evidence_units"]]
                for cid, ch in ontology["chapters"].items()}

    with BATCH_IN.open("w", encoding="utf-8") as f:
        for rec in qs:
            pmid = clean(rec.get("pmid"))
            user = major_prompt() + "\n\nPUBLICATION:\n" + article(rec)
            body = {
                "model": model,
                "messages": [{"role": "system", "content": Q_SYS},
                             {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": q_schema()},
                "max_completion_tokens": 1600,
                "reasoning_effort": effort,
            }
            f.write(json.dumps({
                "custom_id": f"q-{pmid}-{clean(rec.get('chapter_id')).replace('.', '_')}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }, ensure_ascii=False) + "\n")

        for rec in ns:
            pmid = clean(rec.get("pmid"))
            cid = clean(rec.get("chapter_id"))
            if cid not in ontology["chapters"]:
                raise RuntimeError(f"Bad novel chapter {cid} for PMID {pmid}")
            user = (
                unit_prompt(ontology, cid) + "\n\nPUBLICATION:\n" + article(rec) +
                f"\nPrior novel label: {clean(rec.get('novel_topic_label')) or '[none]'}" +
                f"\nPrior novel description: {clean(rec.get('novel_topic_description')) or '[none]'}"
            )
            body = {
                "model": model,
                "messages": [{"role": "system", "content": N_SYS},
                             {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": n_schema(unit_ids[cid])},
                "max_completion_tokens": 1800,
                "reasoning_effort": effort,
            }
            f.write(json.dumps({
                "custom_id": f"n-{pmid}-{cid.replace('.', '_')}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }, ensure_ascii=False) + "\n")

    # Mandatory Batch API QC: every custom_id must be unique.
    custom_ids = []
    with BATCH_IN.open("r", encoding="utf-8") as qc_f:
        for line_no, line in enumerate(qc_f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_ids.append(obj["custom_id"])
    duplicate_ids = sorted({x for x in custom_ids if custom_ids.count(x) > 1})
    if duplicate_ids:
        raise RuntimeError(
            f"HARD FAIL: {len(duplicate_ids)} duplicate Batch custom_id values remain. "
            f"Examples: {duplicate_ids[:10]}"
        )

    size = BATCH_IN.stat().st_size / 1024 / 1024
    total = len(qs) + len(ns)
    if total > 50000 or size > 190:
        raise RuntimeError(f"Unsafe batch size: {total} requests, {size:.1f} MB")
    print("\nRecovery batch prepared.")
    print(f"  questionable:     {len(qs):,}")
    print(f"  novel topics:     {len(ns):,}")
    print(f"  total requests:   {total:,}")
    print(f"  JSONL MB:         {size:.2f}")
    print(f"  model:            {model}")
    print(f"  reasoning:        {effort}")
    print(f"  unique custom IDs:{len(set(custom_ids)):,}/{len(custom_ids):,}")
    print(f"  file:             {BATCH_IN}")

class Client:
    def __init__(self, key, wait):
        if not key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        self.s = requests.Session()
        self.h = {"Authorization": f"Bearer {key}"}
        self.wait = wait

    def request(self, method, path, timeout=300, **kwargs):
        url = path if path.startswith("http") else API + path
        while True:
            headers = dict(self.h)
            headers.update(kwargs.pop("headers", {}))
            try:
                r = self.s.request(method, url, headers=headers,
                                   timeout=timeout, **kwargs)
            except (requests.Timeout, requests.ConnectionError,
                    requests.ChunkedEncodingError,
                    requests.ContentDecodingError) as e:
                print(f"WARN {type(e).__name__}; retry in {self.wait}s")
                time.sleep(self.wait); continue
            if r.status_code in {408,409,429,500,502,503,504}:
                print(f"WARN HTTP {r.status_code}; retry in {self.wait}s")
                time.sleep(self.wait); continue
            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:5000]}")
            return r

def state_load():
    return load_json(STATE) if STATE.exists() else {}

def state_save(x):
    STATE.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def submit(client, model, effort):
    st = state_load()
    if st.get("batch_id"):
        print("Existing batch:", st["batch_id"])
        return st
    with BATCH_IN.open("rb") as f:
        up = client.request("POST", "/files",
                            files={"file": (BATCH_IN.name, f, "application/jsonl")},
                            data={"purpose": "batch"}).json()
    payload = {
        "input_file_id": up["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"project":"ESMO_PDAC_PoC",
                     "task":"recover_questionable_and_novel",
                     "model":model, "reasoning_effort":effort},
    }
    b = client.request("POST", "/batches",
                       headers={"Content-Type":"application/json"},
                       data=json.dumps(payload)).json()
    st = {"input_file_id": up["id"], "batch_id": b["id"],
          "status": b.get("status"), "model": model,
          "reasoning_effort": effort}
    state_save(st)
    print("Batch id:", b["id"])
    print("Status:", b.get("status"))
    return st

def download(client, fid, dest):
    dest.write_bytes(client.request("GET", f"/files/{fid}/content",
                                    timeout=600).content)

def watch(client, poll):
    st = state_load()
    bid = st.get("batch_id")
    if not bid: raise RuntimeError("No batch state")
    while True:
        b = client.request("GET", f"/batches/{bid}").json()
        c = b.get("request_counts") or {}
        print(f"status={b.get('status')}; total={c.get('total')}; "
              f"completed={c.get('completed')}; failed={c.get('failed')}")
        st.update({"status":b.get("status"),
                   "output_file_id":b.get("output_file_id"),
                   "error_file_id":b.get("error_file_id"),
                   "request_counts":c})
        state_save(st)
        if b.get("status") in TERMINAL:
            if b.get("output_file_id"): download(client,b["output_file_id"],BATCH_OUT)
            if b.get("error_file_id"): download(client,b["error_file_id"],BATCH_ERR)
            return b
        time.sleep(poll)

def parse_id(s):
    # custom_id must be unique for every request in an OpenAI Batch.
    # Questionable records are paper-CHAPTER assignments, so PMID alone is
    # insufficient because the same PMID can occur in more than one chapter.
    m = re.fullmatch(r"q-(\d+)-([0-9_]+)", s)
    if m:
        return ("q", m.group(1), m.group(2).replace("_", "."))
    m = re.fullmatch(r"n-(\d+)-([0-9_]+)", s)
    if m:
        return ("n", m.group(1), m.group(2).replace("_", "."))
    raise ValueError(s)

def validate_q(p):
    d = p["decision"]; ids = [c for c in CHAPTERS if c in set(p["chapter_ids"])]
    if d == "mapped_to_existing_chapter" and not ids: raise ValueError("no chapters")
    if d == "out_of_scope" and ids: raise ValueError("OOS with chapters")
    return {"decision":d, "chapter_ids":ids, "confidence":p["confidence"],
            "rationale":clean(p["rationale"])}

def validate_n(p, allowed):
    """
    Deterministically normalize schema-valid GPT output according to the
    explicit `decision` field.

    Structured Outputs requires all schema fields to exist, so GPT may populate
    fields that are irrelevant for the selected decision. Those irrelevant
    fields are cleared rather than treating the whole mapping as failed.
    """
    d = p["decision"]

    units = list(dict.fromkeys(p.get("existing_unit_ids", [])))

    if any(u not in allowed for u in units):
        raise ValueError("invalid unit")

    title = clean(p.get("candidate_title", ""))
    desc = clean(p.get("candidate_description", ""))

    normalization = []

    if d == "existing_unit":
        if not units:
            raise ValueError("existing_unit without existing_unit_ids")

        if title or desc:
            normalization.append("cleared_irrelevant_candidate_fields")

        title = ""
        desc = ""

    elif d in {"new_subunit_candidate", "new_major_chapter_candidate"}:
        if not title or not desc:
            raise ValueError("candidate missing title or description")

        if units:
            normalization.append("cleared_irrelevant_existing_unit_ids")

        units = []

    elif d == "out_of_scope":
        if units or title or desc:
            normalization.append("cleared_irrelevant_assignment_fields")

        units = []
        title = ""
        desc = ""

    else:
        raise ValueError("bad decision")

    return {
        "decision": d,
        "existing_unit_ids": units,
        "candidate_title": title,
        "candidate_description": desc,
        "confidence": p["confidence"],
        "rationale": clean(p["rationale"]),
        "deterministic_normalization": "; ".join(normalization),
    }


def merge():
    ontology=load_json(ONTOLOGY)
    qsrc={(clean(r["pmid"]), clean(r["chapter_id"])): r for r in load_jsonl(Q_IN)}
    nsrc={(clean(r["pmid"]),clean(r["chapter_id"])):r for r in load_jsonl(N_IN)}
    allowed={cid:{u["id"] for u in ch["evidence_units"]}
             for cid,ch in ontology["chapters"].items()}

    qr=[]; nr=[]; failures=[]
    with BATCH_OUT.open("r",encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            obj=json.loads(line); cid=obj.get("custom_id","")
            try:
                kind,pmid,chapter=parse_id(cid)
                resp=obj.get("response")
                if obj.get("error") or not resp or resp.get("status_code")!=200:
                    raise ValueError(obj.get("error") or f"HTTP {resp.get('status_code') if resp else 'missing'}")
                p=json.loads(resp["body"]["choices"][0]["message"]["content"])
                if kind=="q":
                    qr.append({
                        **qsrc[(pmid, chapter)],
                        **validate_q(p),
                        "recovery_type": "questionable",
                        "original_questionable_chapter_id": chapter,
                    })
                else:
                    nr.append({**nsrc[(pmid,chapter)],**validate_n(p,allowed[chapter]),
                               "recovery_type":"novel"})
            except Exception as e:
                failures.append({"line":ln,"custom_id":cid,"error":f"{type(e).__name__}: {e}"})

    existing=[]; newsub=[]; newmajor=[]; oos=[]
    qc=Counter(); nc=Counter()
    for r in qr:
        qc[r["decision"]]+=1
        if r["decision"]=="out_of_scope": oos.append(r)
    for r in nr:
        nc[r["decision"]]+=1
        if r["decision"]=="existing_unit": existing.append(r)
        elif r["decision"]=="new_subunit_candidate": newsub.append(r)
        elif r["decision"]=="new_major_chapter_candidate": newmajor.append(r)
        elif r["decision"]=="out_of_scope": oos.append(r)

    write_jsonl(OUT_Q,qr); write_jsonl(OUT_N,nr); write_jsonl(OUT_EXISTING,existing)
    write_jsonl(OUT_NEW_SUB,newsub); write_jsonl(OUT_NEW_MAJOR,newmajor)
    write_jsonl(OUT_OOS,oos); write_jsonl(OUT_FAIL,failures)

    manifest={
        "questionable_input":len(qsrc),"questionable_successful":len(qr),
        "novel_input":len(nsrc),"novel_successful":len(nr),
        "parse_failures":len(failures),
        "questionable_decision_counts":dict(qc),
        "novel_decision_counts":dict(nc),
        "recovered_existing_unit_records":len(existing),
        "new_subunit_candidates":len(newsub),
        "new_major_chapter_candidates":len(newmajor),
        "out_of_scope_records":len(oos),
        "next_steps":[
            "Map recovered questionable records to precise evidence units in their corrected chapter(s).",
            "Cluster new_subunit_candidates within each major chapter into coherent cross-paper candidate subunits.",
            "Cluster new_major_chapter_candidates globally and accept a new major chapter only if a coherent multi-paper guideline domain emerges.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nRecovery merge completed.")
    print(f"  questionable: {len(qr):,}/{len(qsrc):,}")
    print(f"  novel:        {len(nr):,}/{len(nsrc):,}")
    print(f"  failures:     {len(failures):,}")
    print("  questionable decisions:", dict(qc))
    print("  novel decisions:", dict(nc))
    print(f"  existing-unit recovered:      {len(existing):,}")
    print(f"  new subunit candidates:       {len(newsub):,}")
    print(f"  new major chapter candidates: {len(newmajor):,}")
    print(f"  out of scope:                 {len(oos):,}")
    print(f"  manifest: {MANIFEST}")

def test(client):
    with BATCH_IN.open("r",encoding="utf-8") as f:
        req=json.loads(f.readline())
    r=client.request("POST","/chat/completions",
                     headers={"Content-Type":"application/json"},
                     data=json.dumps(req["body"]))
    print("Testing:",req["custom_id"]); print("HTTP",r.status_code)
    print(json.dumps(r.json(),ensure_ascii=False,indent=2)[:7000])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["prepare","test","submit","watch","merge","all"],default="prepare")
    p.add_argument("--model",default=os.environ.get("OPENAI_MODEL","gpt-5.6-sol"))
    p.add_argument("--reasoning-effort",choices=["none","low","medium","high","xhigh","max"],default="high")
    p.add_argument("--poll-seconds",type=int,default=300)
    p.add_argument("--retry-wait",type=int,default=120)
    p.add_argument("--reset-state",action="store_true")
    a=p.parse_args()

    if a.reset_state and STATE.exists(): STATE.unlink()
    if a.mode in {"prepare","all"}:
        prepare(a.model,a.reasoning_effort)
        if a.mode=="prepare": return 0

    key=os.environ.get("OPENAI_API_KEY","").strip()
    if a.mode in {"test","submit","watch","all"} and not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client=Client(key,a.retry_wait) if key else None

    if a.mode=="test": test(client); return 0
    if a.mode in {"submit","all"}:
        submit(client,a.model,a.reasoning_effort)
        if a.mode=="submit": return 0
    if a.mode in {"watch","all"}:
        b=watch(client,a.poll_seconds)
        print("Batch terminal status:",b.get("status"))
        if b.get("status")!="completed": return 2
        merge(); return 0
    if a.mode=="merge": merge(); return 0
    return 0

if __name__=="__main__":
    raise SystemExit(main())
