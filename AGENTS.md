# AGENTS.md

Project-specific instructions for future Codex work in this repository.

## Scope

This is a clean code and reproducibility documentation copy for
`aisurgeon-pdac-guideline-reconstruction`. The original full working project,
including local data and generated artifacts, remains outside this repository at:

`/mnt/c/living_guideline_platform/PilotPOC`

Do not assume that data artifacts are present in this clean copy.

## Non-Negotiable Rules

- Do not commit, push, or create a GitHub repository unless the user explicitly asks.
- Do not delete, move, or rename existing Python scripts.
- Do not change scientific logic, prompts, evidence rules, model names, or output schemas without explicit user approval.
- Do not print environment variables, API keys, tokens, credential values, or contents of `.env` files.
- Do not add raw PubMed XML, generated CSV/JSONL data, OpenAI Batch inputs or outputs, logs, PDFs, DOCX files, ZIP archives, or credentials to the repository.
- Treat `data/` as a local working area. Only `data/README.md` belongs in version control.

## Dependency Policy

- Keep dependency changes minimal.
- Review `pyproject.toml` and `uv.lock` together when project metadata changes.
- Do not add test dependencies for smoke tests unless needed; prefer the Python standard library for import and syntax checks.

## Testing Policy

- Smoke tests must not call PubMed, OpenAI, or any external API.
- Prefer import-by-path tests with bytecode disabled for script importability checks.
- Run syntax checks across `src/`, `legacy/`, and `tests/` before proposing a first commit.

## Data Handling

- Document required local input filenames instead of committing them.
- If a command needs data from the original full project, copy it locally outside version control or regenerate it.
- Report possible secrets by filename and category only, never by value.
