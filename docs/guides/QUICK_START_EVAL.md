# Quick Start Command for Dakota1890 Eval

## For UI (Recommended - Free Hosted Evals)

Use the Prime Intellect UI at:
https://app.primeintellect.ai/dashboard/environments/harleycooper/dakota1890

**Configuration:**
- Model: Select `qwen/qwen3-235b-a22` from the model dropdown
- Number of Examples: `5`
- Rollouts per example: `1`
- Environment Arguments: Add `dataset_path` entry (check environment docs for exact path format)

## For CLI

If you prefer CLI, get the exact model name first:

```powershell
# Check available models
prime inference models | Select-String "qwen.*235"

# Then run (replace MODEL_NAME with exact ID from list)
$datasetPath = (Resolve-Path dakota_rl_training/datasets/grammar_tasks_complete.jsonl).Path -replace '\\', '/'
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 10}'

prime env eval dakota1890 `
  -m <MODEL_NAME_FROM_LIST> `
  -n 5 `
  -r 1 `
  -t 256 `
  -T 0.7 `
  --env-args $envArgs
```

**Note**: The model name format might need to match exactly what's shown in `prime inference models`. The UI model selector will show the correct name.

