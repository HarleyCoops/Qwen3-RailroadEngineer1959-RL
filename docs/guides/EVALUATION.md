# Evaluation Guide

This repository ships a small, reproducible evaluation kit focused on Dakota dictionary extraction quality.

## Files
- `eval/fixtures/sample_ground_truth.jsonl`: ground-truth subset of Dakota ↔ English pairs.
- `eval/fixtures/sample_predictions.jsonl`: example predictions to demonstrate metrics.
- `eval/score_extraction.py`: scoring helpers (character distance, token accuracy, diacritic preservation).
- `eval/run_eval.py`: CLI wrapper that outputs a Markdown report, including commit hash and timestamp.

## Metrics
| Metric | Description |
| --- | --- |
| **Token accuracy** | Fraction of exact `{dakota, english}` pair matches. |
| **Char distance** | Average normalized Levenshtein distance (lower is better) across Dakota and English text. |
| **Diacritic preservation** | Proportion of combining-mark code points preserved in predictions. |

## Usage
```bash
python -m pip install python-Levenshtein
python eval/run_eval.py --pred eval/fixtures/sample_predictions.jsonl \
                        --truth eval/fixtures/sample_ground_truth.jsonl \
                        --out eval/report.md
```

The generated report is a Markdown file summarizing metrics and referencing the repository commit hash for reproducibility. For larger-scale evaluations, replace the fixtures with your model outputs while keeping the same JSONL schema (`{"dakota": ..., "english": ...}`).

## Interpretation
- Aim for **token accuracy** close to 1.0 when evaluated on gold data.
- **Char distance** approaching 0 indicates near-perfect character-level matches.
- **Diacritic preservation** near 1 signals that accent marks survived extraction.

When reporting results, accompany metrics with qualitative review from Dakota language experts to ensure respectful, accurate usage.
