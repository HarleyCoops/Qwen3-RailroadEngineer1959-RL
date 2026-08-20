# Post-Training Plan: Analytics & HF Model Upload

## Overview
After training completes, we'll run all existing analytics scripts and upload to HuggingFace with amazing graphics.

## Step 1: Get New Run IDs

Once training completes, get the run IDs from W&B:
- Trainer run ID (from `dakota-0.6b-ledger-test-400-trainer`)
- Orchestrator run ID (from `dakota-0.6b-ledger-test-400-orchestrator`)

You can find these in:
- W&B dashboard: https://wandb.ai/christian-cooper-us/dakota-rl-grammar
- Or from the training output logs

## Step 2: Generate All Visualizations

### 2.1 Main RL Visualizations
```bash
python scripts/create_rl_visualizations.py \
    --trainer-id <NEW_TRAINER_ID> \
    --orchestrator-id <NEW_ORCHESTRATOR_ID> \
    --project dakota-rl-grammar \
    --entity christian-cooper-us \
    --output-dir wandb_visualizations
```

**Outputs:**
- `wandb_visualizations/reward_progression.png`
- `wandb_visualizations/training_metrics.png`
- `wandb_visualizations/performance_metrics.png`
- `wandb_visualizations/comprehensive_dashboard.png`

### 2.2 Reward Ledger Plot (if CSV exists)
```bash
# If reward_ledger.csv was generated during training
python scripts/analysis/plot_reward_ledger.py \
    --csv wandb_analysis/reward_ledger.csv \
    --out wandb_analysis/reward_ledger.png
```

### 2.3 Generate Ledger Snippet Table
```bash
# Create markdown table snippet
python scripts/analysis/make_ledger_snippet.py
```

**Output:** `wandb_analysis/reward_ledger_head_tail.md`

### 2.4 Comprehensive W&B Analysis
```bash
# Export comprehensive analysis
python scripts/analysis/export_comprehensive_analysis.py \
    --trainer-run <NEW_TRAINER_ID> \
    --orchestrator-run <NEW_ORCHESTRATOR_ID>
```

### 2.5 Detailed W&B Report
```bash
# Create detailed report
python scripts/analysis/create_wandb_report.py \
    --trainer-run <NEW_TRAINER_ID> \
    --orchestrator-run <NEW_ORCHESTRATOR_ID>
```

## Step 3: Prepare Model Card

Update `MODEL_CARD.md` with:
- New run IDs and links
- Updated visualizations (point to new images)
- Training metrics from new run
- Reward ledger section (if available)

## Step 4: Initialize/Update HF Model Repo

### 4.1 Create Repo (if new version)
```bash
# The script will create repo if it doesn't exist
python scripts/conversion/upload_model_card.py \
    --repo-id "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL" \
    --model-card-path MODEL_CARD.md
```

### 4.2 Upload Model Weights (from training output)
```bash
# After training completes, upload model weights
# Check where weights are saved (likely in ~/dakota_rl_training/outputs/ledger_test_400/weights/)
# Then use huggingface-cli or the upload script
```

## Step 5: Upload Visualizations to HF

Copy all visualization images to the HF repo:
- `wandb_visualizations/*.png`
- `wandb_analysis/reward_ledger.png` (if exists)

These will be referenced in the MODEL_CARD.md

## Quick Command Summary

```bash
# Set run IDs (get from W&B after training completes)
export TRAINER_ID="<new_trainer_id>"
export ORCHESTRATOR_ID="<new_orchestrator_id>"

# Generate all visualizations
python scripts/create_rl_visualizations.py \
    --trainer-id $TRAINER_ID \
    --orchestrator-id $ORCHESTRATOR_ID

# Plot reward ledger (if CSV exists)
python scripts/analysis/plot_reward_ledger.py

# Generate ledger snippet
python scripts/analysis/make_ledger_snippet.py

# Export comprehensive analysis
python scripts/analysis/export_comprehensive_analysis.py \
    --trainer-run $TRAINER_ID \
    --orchestrator-run $ORCHESTRATOR_ID

# Upload model card to HF
python scripts/conversion/upload_model_card.py
```

## Notes

- All scripts already exist - no new code needed
- Keep same repo name but update version in MODEL_CARD.md
- All visualizations will be automatically generated from W&B data
- Make sure W&B API key is set in `.env` or environment

