# Pipeline

Run commands from the repository root. Supply required local inputs under
`data/`; generated data, batch files, state, logs, and document outputs remain
untracked.

The scripts below are listed in intended execution order. Modes shown as
`prepare`, `submit`, `watch`, `merge`, or `all` are implemented by the
corresponding script; use `--help` on a script for exact CLI options.

## 1. PubMed Retrieval

`src/run_pubmed_search.py`

Runs chapter-specific PubMed E-utilities searches for the 2015-01-01 to
2023-08-31 publication-date window. It writes `data/pubmed_results.csv`,
`data/query_registry.json`, optional raw XML under `data/raw_xml/`, and logs
under `logs/`.

`src/queries.py`

Defines the eight chapter-specific PubMed query strings used by
`run_pubmed_search.py`.

## 2. Evidence Selection

`src/deduplicate_and_select_evidence.py`

Deduplicates PubMed rows globally by PMID, classifies evidence types, excludes
guidelines/consensus records, and writes selected and excluded CSV outputs.

## 3. Major Chapter Mapping

`src/gpt_map_chapters_batch.py`

Prepares, submits, watches, downloads, parses, and merges the OpenAI Batch job
that maps selected publications to one or more of the eight ESMO-PDAC chapters.

## 4. Chapter Integration Inputs

`src/build_mapped_evidence_by_chapter.py`

Converts the chapter-mapped CSV into one JSONL record per paper-chapter
assignment and writes chapter mapping QC outputs.

`src/build_guideline_integration_master.py`

Builds the initial chapter-wise integration master by joining the ESMO 2015
baseline with mapped evidence by chapter. This script makes no API call.

## 5. Existing Evidence-Unit Submapping

`src/gpt_submap_evidence_units_batch.py`

Maps each paper-chapter assignment to precise existing evidence units. It also
separates novel-topic records and questionable chapter assignments for later
resolution.

## 6. Recovery and Questionable Records

`src/gpt_recover_questionable_and_novel_batch.py`

Rechecks questionable chapter assignments and novel-topic records. Outputs are
used to recover existing evidence-unit assignments and collect new-subunit or
new-major-chapter candidates.

`src/gpt_recover_questionable_and_novel_batch_before_normalization.py`

Legacy pre-normalization version of the recovery script. Keep it for
reproducibility audits; do not use it for new runs unless intentionally
reproducing that earlier stage.

`src/gpt_submap_recovered_questionables_batch.py`

Submaps recovered questionable records into existing evidence units or
new-subunit candidate streams.

## 7. New Subunit Taxonomy

`src/combine_new_subunit_candidates.py`

Combines new-subunit candidate streams and applies manual evidence exclusions.
This script makes no API call.

`src/gpt_design_new_subunit_taxonomy_batch.py`

Clusters consolidated new-subunit candidates into a broad, medically coherent
proposed taxonomy within existing major chapters.

`src/repair_taxonomy_chapters_2_3.py`

Conditional repair script for taxonomy chapters 2 and 3 when the original
responses exhausted the earlier token budget. It reuses the original request
bodies and changes only the repair parameters described in the script.

`src/gpt_assign_new_subunit_candidates_batch.py`

Assigns each new-subunit candidate record to the proposed taxonomy clusters and
writes expanded assignment and QC outputs.

`src/build_new_subunit_support_report.py`

Builds deterministic support and QC reports for the proposed taxonomy. This
script makes no API call.

`src/resolve_new_major_chapter_candidates.py`

Resolves new-major-chapter candidate records back into accepted existing major
chapters and evidence units or accepted new subunits.

## 8. Ontology Freeze and Final Integration Master

`src/freeze_ontology_v2_and_build_integration_master.py`

Freezes ontology v2, writes crosswalk/audit files, deduplicates final evidence
assignments, and builds the final guideline integration master. This script
makes no API call.

## 9. Stage A Evidence Synthesis

`src/final_stageA_evidence_synthesis_gpt56.py`

Prepares chunk-level and reducer-level OpenAI Batch jobs for final evidence
appraisal and synthesis by frozen evidence unit.

`src/repair_stageA_reducer_partitions.py`

Conditional deterministic repair for known Stage-A reducer outputs whose PMID
status partitions were incomplete or duplicated. It uses completed chunk
decisions as canonical and makes no API call.

## 10. Stage B Rewrite and DOCX Assembly

`src/stageB_rewrite_and_docx.py`

Prepares, submits, watches, parses, and merges chapter rewrite Batch outputs,
then can assemble the final DOCX output locally when the optional `python-docx`
dependency is available in the runtime environment.

## Legacy Scripts

`legacy/PilotPOC_NCBI_v3/src/run_pubmed_search.py` and
`legacy/PilotPOC_NCBI_v3/src/queries.py` preserve the earlier PubMed retrieval
implementation for reproducibility comparison. They are not the default current
pipeline entry point.
