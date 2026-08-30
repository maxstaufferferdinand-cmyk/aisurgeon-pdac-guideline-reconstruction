# AISurgeon PDAC Guideline Reconstruction

Clean code and reproducibility documentation for reconstructing the AISurgeon
PDAC living-guideline proof-of-concept pipeline. The pipeline searches PubMed
for evidence published from 2015-01-01 through 2023-08-31, maps publications to
ESMO pancreatic cancer guideline chapters and evidence units, synthesizes
evidence with OpenAI Batch jobs, and assembles a final guideline-update document.

This repository intentionally excludes generated data and document artifacts.
The original full local project remains available at:

`/mnt/c/living_guideline_platform/PilotPOC`

## Repository Contents

- `src/`: current pipeline scripts.
- `legacy/`: preserved earlier PubMed retrieval scripts for reproducibility
  comparison.
- `docs/PIPELINE.md`: every script in execution order.
- `docs/DATA_POLICY.md`: excluded-file policy and rationale.
- `data/README.md`: local data staging instructions.
- `tests/`: non-network smoke tests for syntax and importability.

## Data Policy

Only code and reproducibility documentation should be committed. Do not commit
raw PubMed XML, generated CSV/JSON/JSONL files, OpenAI Batch inputs or outputs,
state files, logs, PDFs, DOCX files, spreadsheets, archives, `.env` files, or
credentials. See `docs/DATA_POLICY.md`.

## Install

Install `uv`, then run:

```bash
uv sync
```

If `uv` is unavailable, use Python 3.12 or newer and install the dependencies
declared in `pyproject.toml`.

## Environment Variables

Create a local `.env` or set shell environment variables outside version
control. `.env.example` lists the variable names used by the scripts and contains
no real values.

The PubMed retrieval step uses:

- `NCBI_API_KEY`
- `NCBI_EMAIL`
- `NCBI_TOOL`

OpenAI Batch steps use:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Scripts must not print API key or credential values.

## Smoke Checks

Run non-network validation before preparing a first commit:

```bash
python3 -m unittest discover -s tests
```

The smoke tests compile and import scripts without calling PubMed, OpenAI, or
other external APIs.

## PubMed Validation Example

Count chapter 1 for the first half of 2015 without downloading records:

```bash
uv run python src/run_pubmed_search.py \
  --chapter 1 \
  --start-date 2015-01-01 \
  --end-date 2015-06-30 \
  --count-only
```

Then fetch that same slice:

```bash
uv run python src/run_pubmed_search.py \
  --chapter 1 \
  --start-date 2015-01-01 \
  --end-date 2015-06-30
```

Run a full chapter or all chapters:

```bash
uv run python src/run_pubmed_search.py --chapter 1
uv run python src/run_pubmed_search.py --chapter all
```

Interrupting with `Ctrl+C` is safe for the PubMed retrieval script. Completed
EFetch batches remain in the local CSV; rerun the same command to resume.

## Pipeline

See `docs/PIPELINE.md` for the full execution order, including deterministic
steps, OpenAI Batch stages, repair scripts, and legacy PubMed scripts.

## Chapter Identifiers

- `1`: Incidence and epidemiology
- `2`: Diagnosis, pathology and molecular biology
- `3`: Staging and risk assessment
- `4.1`: Treatment of localised disease
- `4.2`: Borderline resectable and locally advanced disease
- `4.3`: Advanced/metastatic disease
- `5`: Personalised medicine
- `6`: Follow-up and long-term implications
