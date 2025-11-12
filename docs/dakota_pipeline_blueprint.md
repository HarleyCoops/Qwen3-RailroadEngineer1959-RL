# Dakota Dictionary Extraction Blueprint

This blueprint codifies the remediation plan for the Dakota 1890 dictionary
pipeline. It aligns the extraction, validation, and dataset curation stages so
that supervised and RL training never reward English↔English copies.

## 1. Source Audit
- Catalogue every scanned folio before OCR with page range, column layout, and
typography (handwritten vs. type).
- Store metadata in `data/source_manifest.json` (generated outside of this
repository) so conversion tools can reason about batch boundaries.

## 2. High-Fidelity OCR + VLM Pass
- Run Gemini 2.5 Flash or an equivalent VLM with two prompts per page:
  1. Dakota headwords with diacritics, part-of-speech, morphology, and examples.
  2. English definitions, context sentences, and notes.
- Force structured JSON responses to prevent column swaps.

## 3. Structural Tagging
- Require the VLM to emit fields: `dakota`, `english`, `pos`, `morphology`,
  `examples`, `notes`, `page`, and `confidence`.
- `dakota_extraction/core/page_processor.py` already requests this shape; the
  builder now enforces it downstream.

## 4. Orthography Validator
- `DakotaOrthographyValidator` checks every headword for distinctive Dakota
  characters or combining marks.
- Any batch with >0.5% failures aborts and surfaces sample strings.

## 5. Context Stitching
- VLM output concatenates multi-sentence definitions into the `english` field
  and, when relevant, appends `examples` and `notes` arrays.

## 6. Backward-Question Generator
- `build_rl_tasks` emits Stoney Nakoda–style forward (Dakota→English) and
  backward (English→Dakota) QA pairs with metadata tags for direction.

## 7. Difficulty Labelling
- `estimate_difficulty` scores entries using definition length, morphology, and
  usage notes to support curriculum scheduling.

## 8. Deduplication & Canonicalization
- `TrainingDatasetBuilder` collapses duplicate headwords using an NFC-based key
  while preserving aliases and taking the highest-confidence record.

## 9. Provenance Logging
- Every dataset artifact references `entry_id`, `page_number`, and
  `source_image`; `dataset_report.json` captures duplicate clusters and
  guardrail stats.

## 10. Manual Spot Checks
- `manual_spot_check.json` samples ≥1% of canonical entries for review, keeping
  Dakota and English columns side-by-side.

## 11. Dataset Partitioning
- `sft_train.jsonl` and `sft_valid.jsonl` serve supervised fine-tuning.
- `rl_tasks_all.jsonl` consolidates RL prompts with consistent schema.

## 12. Continuous QA Guardrails
- CI uses `tests/test_training_dataset_builder.py` to block merges when
  orthography guardrails or identical Dakota/English strings appear.
- Extend with custom schema tests as additional guardrails are devised.
