# Corrected W&B Sync Commands

## Issue 1: Run sync from project root with absolute paths

`uv run wandb` must be executed from `/workspace/prime-rl` where the .venv lives. Use absolute paths.

## Issue 2: Create wandb_analysis in project root

The analysis directory should be `/workspace/prime-rl/wandb_analysis`, not `~/wandb_analysis`.

## Corrected Commands

**On Server:**

```bash
cd /workspace/prime-rl

# Sync runs from project root with absolute paths
uv run wandb sync \
  ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm \
  ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr

# Create wandb_analysis in project root (not home directory)
mkdir -p wandb_analysis

# Copy configs to project root analysis directory
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm/files/config.yaml wandb_analysis/trainer_config.yaml
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr/files/config.yaml wandb_analysis/orchestrator_config.yaml

# Check for reward ledger CSV (it may not exist)
ls -lh ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv 2>/dev/null || echo "No reward ledger CSV found - orchestrator may not have emitted it"
```

## Note on Reward Ledger

The reward ledger CSV doesn't exist because the orchestrator never emitted it. This is expected if the ledger logging wasn't fully integrated in this run. The main visualizations and W&B metrics are still available.

## Verify Sync

After syncing, check W&B dashboard:
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/yut26kcm
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/1y33h9zr

Both should show as "Finished" with all data synced.

