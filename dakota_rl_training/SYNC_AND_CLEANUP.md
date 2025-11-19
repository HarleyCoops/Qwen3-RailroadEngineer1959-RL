# Sync W&B Runs and Cleanup After Training

## Step 1: Sync W&B Runs to Cloud

**On Server:**

```bash
cd /workspace/prime-rl

# Sync all wandb runs to cloud (this ensures both trainer and orchestrator show as finished)
uv run wandb sync --clean ~/dakota_rl_training/outputs/ledger_test_400/wandb
```

This ensures both `yut26kcm` (trainer) and `1y33h9zr` (orchestrator) runs show up as finished on wandb.ai.

## Step 2: Copy Configs/Logs for Later Analysis

**On Server:**

```bash
# Copy trainer config
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm/files/config.yaml wandb_analysis/trainer_config.yaml

# Copy orchestrator config
cp ~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr/files/config.yaml wandb_analysis/orchestrator_config.yaml
```

## Step 3: Verify Checkpoints + Reward Ledger

**On Server:**

```bash
# Check checkpoints exist
ls -lh ~/dakota_rl_training/outputs/ledger_test_400/checkpoints

# Check for reward ledger CSV
ls -lh ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv

# If reward ledger exists, copy it to analysis directory
# (Adjust path if your analysis directory is elsewhere)
cp ~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv wandb_analysis/ 2>/dev/null || echo "No reward ledger CSV found"
```

## Step 4: Shut Everything Down

**On Server:**

```bash
# Check for running processes
ps aux | grep -E "prime_rl|torchrun|rl" | grep -v grep

# Kill any remaining processes (if needed)
pkill -f "prime_rl" 2>/dev/null
pkill -f "torchrun" 2>/dev/null

# Verify GPUs are idle
nvidia-smi

# Once idle, it's safe to stop the instance
```

## Step 5: Download Files to Local Machine (Optional)

**From PowerShell (if you want to download configs/logs locally):**

```powershell
# Download configs
scp -i C:\Users\chris\.ssh\prime_rl_key -P 1234 root@185.216.20.236:~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192353-yut26kcm/files/config.yaml wandb_analysis/trainer_config.yaml

scp -i C:\Users\chris\.ssh\prime_rl_key -P 1234 root@185.216.20.236:~/dakota_rl_training/outputs/ledger_test_400/wandb/run-20251112_192348-1y33h9zr/files/config.yaml wandb_analysis/orchestrator_config.yaml

# Download reward ledger CSV (if it exists)
scp -i C:\Users\chris\.ssh\prime_rl_key -P 1234 root@185.216.20.236:~/dakota_rl_training/outputs/ledger_test_400/logs/reward_ledger*.csv wandb_analysis/ 2>$null
```

## Verification Checklist

After running these commands, verify:

- [ ] W&B runs show as "Finished" on wandb.ai
- [ ] Config files copied to wandb_analysis/
- [ ] Checkpoints exist in outputs/ledger_test_400/checkpoints/
- [ ] Reward ledger CSV copied (if it exists)
- [ ] No processes running (nvidia-smi shows idle GPUs)
- [ ] Safe to stop instance

