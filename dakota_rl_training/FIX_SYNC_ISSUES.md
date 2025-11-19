# Fix W&B Sync Issues

## Problem
- `wandb sync --clean` fails with IndexError on directory
- `wandb_analysis` directory doesn't exist

## Solution: Sync Individual Runs + Create Directory

**On Server:**

```bash
cd /workspace/prime-rl

# Create wandb_analysis directory first
mkdir -p ~/wandb_analysis

# Sync individual runs (not the whole directory)
cd ~/dakota_rl_training/outputs/ledger_test_400/wandb

# Sync trainer run
uv run wandb sync run-20251112_192353-yut26kcm

# Sync orchestrator run  
uv run wandb sync run-20251112_192348-1y33h9zr

# Go back to prime-rl
cd /workspace/prime-rl

# Copy configs (create directory first)
mkdir -p ~/wandb_analysis
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm/files/config.yaml ~/wandb_analysis/trainer_config.yaml
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr/files/config.yaml ~/wandb_analysis/orchestrator_config.yaml

# Check for reward ledger CSV
ls -lh ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv 2>/dev/null || echo "No reward ledger CSV found"

# If reward ledger exists, copy it
cp ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv ~/wandb_analysis/ 2>/dev/null || echo "No reward ledger CSV found"
```

## Alternative: Sync Without --clean Flag

If individual syncs don't work, try syncing the parent directory without --clean:

```bash
cd ~/dakota_rl_training/outputs/ledger_test_400
uv run wandb sync wandb/
```

## Verify Sync

After syncing, check W&B dashboard:
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/yut26kcm
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/1y33h9zr

Both should show as "Finished" status.

