# Run Visualizations for Qwen3-0.6B-Dakota-Grammar-RL-400

## Run IDs Identified
- **Trainer Run ID**: `yut26kcm`
- **Orchestrator Run ID**: `1y33h9zr` (from `run-20251112_192348-1y33h9zr`)

## Run All Visualizations

**On Local Machine (PowerShell):**

```powershell
# Set run IDs
$env:TRAINER_ID = "yut26kcm"
$env:ORCHESTRATOR_ID = "1y33h9zr"

# Main RL visualizations (creates 4 beautiful plots)
python scripts/create_rl_visualizations.py `
    --trainer-id $env:TRAINER_ID `
    --orchestrator-id $env:ORCHESTRATOR_ID `
    --project dakota-rl-grammar `
    --entity christian-cooper-us `
    --output-dir wandb_visualizations

# Reward ledger plot (if CSV exists)
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

## Expected Outputs

After running, you should have:
-  `wandb_visualizations/reward_progression.png`
-  `wandb_visualizations/training_metrics.png`
-  `wandb_visualizations/performance_metrics.png`
-  `wandb_visualizations/comprehensive_dashboard.png`
-  `wandb_analysis/reward_ledger.png` (if CSV exists)
-  `wandb_analysis/reward_ledger_head_tail.md` (if CSV exists)

## Next Steps

After visualizations are generated:
1. Update MODEL_CARD.md with new run info and links
2. Upload model card to HF repo: `HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL-400`
3. Upload model weights from training output

