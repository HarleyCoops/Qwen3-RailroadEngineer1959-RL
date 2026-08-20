# Fix W&B Sync - Use Python Module

## Problem
`uv run wandb sync` fails because wandb isn't available as a command. Use Python module instead.

## Solution: Use Python -m wandb

**On Server:**

```bash
cd /workspace/prime-rl

# Create wandb_analysis directory first
mkdir -p ~/wandb_analysis

# Sync individual runs using Python module
cd ~/dakota_rl_training/outputs/ledger_test_400/wandb

# Sync trainer run
uv run python -m wandb sync run-20251112_192353-yut26kcm

# Sync orchestrator run  
uv run python -m wandb sync run-20251112_192348-1y33h9zr

# Go back to prime-rl
cd /workspace/prime-rl

# Copy configs
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm/files/config.yaml ~/wandb_analysis/trainer_config.yaml
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr/files/config.yaml ~/wandb_analysis/orchestrator_config.yaml

# Check for reward ledger CSV
ls -lh ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv 2>/dev/null || echo "No reward ledger CSV found"

# If reward ledger exists, copy it
cp ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv ~/wandb_analysis/ 2>/dev/null || echo "No reward ledger CSV found"
```

## Alternative: Check if Runs Already Synced

The runs might already be synced automatically. Check W&B dashboard:
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/yut26kcm
- https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/1y33h9zr

If they show as "Finished", no sync needed!

