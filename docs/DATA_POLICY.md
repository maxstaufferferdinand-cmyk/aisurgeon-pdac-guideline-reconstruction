# Data Policy

This repository is a clean code and reproducibility documentation copy. It must
not contain patient data, raw literature exports, generated model inputs or
outputs, logs, documents, archives, or credentials.

## Excluded Content

The `.gitignore` policy excludes all files by default and only allows source
code plus reproducibility documentation. The following categories are
intentionally excluded:

- `data/` contents other than `data/README.md`: local inputs, generated CSV,
  JSON, JSONL, XML, manifests, audits, parse failures, and pipeline outputs.
- `logs/`, `output/`, `outputs/`, `results/`, `artifacts/`, `batch/`, and
  `batches/`: runtime logs, Word outputs, generated reports, and batch
  artifacts.
- OpenAI Batch files and state files: request JSONL, response JSONL, error JSONL,
  batch state JSON, and repair state JSON.
- Raw PubMed XML and PubMed-derived CSV/JSONL outputs: large reproducible
  artifacts that must be regenerated or supplied locally.
- PDFs, DOCX, XLSX/XLS, ODT, and ZIP/TAR archives: source guideline documents,
  final document exports, spreadsheets, and compressed project snapshots.
- `.env`, credential, key, token, certificate, and secret-like filenames:
  anything that may contain API keys or account credentials.
- Python caches, virtual environments, editor settings, and OS-local files:
  machine-specific artifacts unrelated to reproducibility.

## Local Data Use

Scripts read and write local files under `data/`, `logs/`, and `output/`.
Create those directories when running the pipeline locally. The expected file
names are documented in `docs/PIPELINE.md`; the files themselves are not tracked.

When copying inputs from the original full project, copy only the files required
for the step being run and keep them uncommitted.
