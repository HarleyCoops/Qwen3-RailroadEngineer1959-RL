# Evaluation (Small, Reproducible Subset)

This folder ships a **versioned subset** and scoring tools to reproduce extraction metrics without paid APIs.

## Files
- `fixtures/sample_ground_truth.jsonl` — canonical pairs `{dakota, english}`.
- `fixtures/sample_predictions.jsonl` — example predictions for smoke.
- `score_extraction.py` — metrics.
- `run_eval.py` — CLI wrapper to generate a Markdown report.

## Run
```bash
python -m pip install python-Levenshtein
python eval/run_eval.py --pred eval/fixtures/sample_predictions.jsonl \
                        --truth eval/fixtures/sample_ground_truth.jsonl \
                        --out eval/report.md
```

Outputs `eval/report.md` with character-level (normalized), token accuracy, and diacritic-preservation.
