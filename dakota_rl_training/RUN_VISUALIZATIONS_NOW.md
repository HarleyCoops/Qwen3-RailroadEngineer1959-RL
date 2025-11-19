# Run Visualizations for Qwen3-0.6B-Dakota-Grammar-RL-400

## Step 1: Get Run IDs

From the training output, I can see:
- **Trainer Run ID**: `yut26kcm` (from wandb path: `run-20251112_192353-yut26kcm`)
- **Orchestrator Run ID**: Need to check W&B dashboard or logs

**On Server (to find orchestrator run):**
```bash
# Check orchestrator logs for run ID
ls -la ~/dakota_rl_training/outputs/ledger_test_400/wandb/

# Or check W&B dashboard:
# https://wandb.ai/christian-cooper-us/dakota-rl-grammar
# Look for run named: dakota-0.6b-ledger-test-400-orchestrator
```

## Step 2: Run All Visualizations

**On Local Machine (PowerShell):**

```powershell
# Set run IDs (update ORCHESTRATOR_ID once you have it)
$env:TRAINER_ID = "yut26kcm"
$env:ORCHESTRATOR_ID = "<orchestrator_id_here>"

# Main RL visualizations
python scripts/create_rl_visualizations.py `
    --trainer-id $env:TRAINER_ID `
    --orchestrator-id $env:ORCHESTRATOR_ID `
    --project dakota-rl-grammar `
    --entity christian-cooper-us `
    --output-dir wandb_visualizations

# Reward ledger plot (if CSV exists from training)
python scripts/analysis/plot_reward_ledger.py `
    --csv wandb_analysis/reward_ledger.csv `
    --out wandb_analysis/reward_ledger.png

# Generate ledger snippet table
python scripts/analysis/make_ledger_snippet.py

# Export comprehensive analysis
python scripts/analysis/export_comprehensive_analysis.py `
    --trainer-run $env:TRAINER_ID `
    --orchestrator-run $env:ORCHESTRATOR_ID

# Create detailed W&B report
python scripts/analysis/create_wandb_report.py `
    --trainer-run $env:TRAINER_ID `
    --orchestrator-run $env:ORCHESTRATOR_ID
```

## Step 3: Check Outputs

After running, verify these files exist:
- `wandb_visualizations/reward_progression.png`
- `wandb_visualizations/training_metrics.png`
- `wandb_visualizations/performance_metrics.png`
- `wandb_visualizations/comprehensive_dashboard.png`
- `wandb_analysis/reward_ledger.png` (if CSV exists)
- `wandb_analysis/reward_ledger_head_tail.md` (if CSV exists)

## Next Steps

After visualizations are generated:
1. Update MODEL_CARD.md with new run info
2. Upload model card to HF
3. Upload model weights to HF

