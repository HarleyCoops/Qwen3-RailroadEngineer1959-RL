# Hosted Eval Configuration Guide - Short Test Run

## Recommended Configuration for Testing Pipes

### Number of Examples
**Recommendation: `5`**

- Minimal enough to test quickly
- Large enough to catch basic issues
- Can increase to `10` if you want slightly more confidence
- Avoid "Use all examples" checkbox for test runs

### Rollouts per Example
**Recommendation: `1`**

- For testing pipes, you don't need statistical aggregation
- Multiple rollouts are for robust benchmarking, not pipe testing
- `1` rollout = fastest completion = quickest feedback
- Save `3` rollouts for actual evaluation runs after pipes work

### Custom Environment Arguments

**YES, you need custom environment arguments!**

#### Required Arguments:

```json
{
  "dataset_path": "/path/to/dakota_rl_training/datasets/grammar_tasks_complete.jsonl",
  "max_examples": 10
}
```

**Important**: In the UI, you'll need to provide the dataset path that's accessible from Prime Intellect's servers. This might be:
- A public URL to your dataset
- A path relative to your environment's data directory
- Or the dataset needs to be packaged with your environment

#### For Weights & Biases & Hugging Face:

**Good News**: The UI says "Environment secrets will be auto exposed for the evaluation"

This means:
- If you've configured `WANDB_API_KEY`, `WANDB_PROJECT`, `HF_TOKEN` as **secrets** in your Prime Intellect environment settings, they'll be automatically available
- You DON'T need to pass them as environment arguments
- The environment will pick them up automatically

**However**, if you want to specify W&B project/entity explicitly, you can add:

```json
{
  "dataset_path": "/path/to/dataset.jsonl",
  "max_examples": 10
}
```

And ensure these are set as environment secrets in Prime Intellect:
- `WANDB_API_KEY` - Your W&B API key
- `WANDB_PROJECT` - Project name (defaults to "dakota-rl-grammar" if not set)
- `WANDB_ENTITY` - Your W&B username/entity (optional)
- `HF_TOKEN` - Your Hugging Face token (for accessing models/datasets)

### Model Selection

For Qwen3 235B-a22, use model ID: `qwen/qwen3-235b-a22`

Available variants:
- `qwen/qwen3-235b-a22` ($0.22/$0.88 per 1M tokens) - Cheapest
- `qwen/qwen3-235b-a22` ($0.65/$3 per 1M tokens) - Different version

## Complete UI Configuration

**Step 2 Configuration:**
- **Number of Examples:** `5` (or `10` for slightly more confidence)
- **Use all examples:**  Unchecked
- **Rollouts per example:** `1`
- **Environment Arguments:** Click "+ Add Entry" and add:
  - Key: `dataset_path`
  - Value: Your dataset path (as accessible from Prime servers)
  - Key: `max_examples` (optional but recommended)
  - Value: `10`

**Before Running:**
1. Ensure your dataset is accessible (packaged with environment or accessible via URL)
2. Set environment secrets in Prime Intellect dashboard:
   - `WANDB_API_KEY`
   - `WANDB_PROJECT` (optional, defaults to "dakota-rl-grammar")
   - `HF_TOKEN` (if needed for model access)

## Troubleshooting

**If dataset_path doesn't work:**
- Check if your environment package includes the dataset
- Consider uploading dataset to a public URL and using that path
- Or ensure the dataset is in the environment's expected location

**If W&B/HF auth fails:**
- Verify secrets are set in Prime Intellect environment settings
- Check secret names match exactly: `WANDB_API_KEY`, `HF_TOKEN`
- Secrets are case-sensitive

