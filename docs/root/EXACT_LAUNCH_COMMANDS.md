# Exact Commands to Launch Training on Prime Intellect Instance

## Prerequisites
- You have SSH access to your Prime Intellect instance
- Instance is provisioned with 8x A100 GPUs
- Config files are synced to the instance

## Step 1: SSH into Your Instance

```bash
ssh <your-instance-ip-or-hostname>
# Or however Prime Intellect provides SSH access
```

## Step 2: Navigate to Config Directory

```bash
cd ~/dakota-rl-training/configs
```

## Step 3: Verify Configs Are Correct

```bash
# Check micro_batch_size and environment section
cat orch_30b.toml | grep -E "micro_batch_size|environment"

# Check tensor parallelism
cat infer_30b.toml | grep "tp ="

# Check trust_remote_code
cat train_30b.toml | grep "trust_remote_code"
```

**Expected output:**
```
micro_batch_size = 2
[environment]
tp = 2
trust_remote_code = true
```

## Step 4: Clean Previous Outputs (if any)

```bash
rm -rf ~/dakota-rl-training/outputs/*
```

## Step 5: Launch Training

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
```

## Step 6: Monitor Logs (in separate terminal)

**Option A: Watch trainer logs**
```bash
watch -n 2 'tail -50 ~/dakota-rl-training/outputs/logs/trainer/rank_0.log'
```

**Option B: Tail orchestrator logs**
```bash
tail -f ~/dakota-rl-training/outputs/logs/orchestrator/orchestrator.log
```

**Option C: Check all logs**
```bash
# Trainer logs
tail -f ~/dakota-rl-training/outputs/logs/trainer/rank_*.log

# Orchestrator logs
tail -f ~/dakota-rl-training/outputs/logs/orchestrator/*.log

# Inference logs
tail -f ~/dakota-rl-training/outputs/logs/inference/*.log
```

## Complete One-Liner (if paths are different)

If your paths are different, adjust accordingly:

```bash
cd ~/prime-rl && uv run rl --trainer @ ~/dakota-rl-training/configs/train_30b.toml --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml --inference @ ~/dakota-rl-training/configs/infer_30b.toml --trainer-gpu-ids 4,5,6,7 --inference-gpu-ids 0,1,2,3 --output-dir ~/dakota-rl-training/outputs
```

## Troubleshooting Commands

**Check if configs exist:**
```bash
ls -la ~/dakota-rl-training/configs/*.toml
```

**Check GPU availability:**
```bash
nvidia-smi
```

**Check if prime-rl is installed:**
```bash
cd ~/prime-rl && uv run python -c "import prime_rl; print('OK')"
```

**Kill previous training (if stuck):**
```bash
pkill -f "uv run rl"
```

**Check running processes:**
```bash
ps aux | grep "uv run rl"
```

## What to Look For in Logs

**Good signs:**
- "Initializing model and tokenizer"
- "Starting RL trainer"
- "Model loading completes"
- GPU utilization > 0%

**Bad signs:**
- "Connection closed"
- "Rank 3 crashed"
- "Out of memory"
- "Config file not found"

## Quick Copy-Paste Version

```bash
# Verify configs
cd ~/dakota-rl-training/configs
cat orch_30b.toml | grep -E "micro_batch_size|environment"
cat infer_30b.toml | grep "tp ="
cat train_30b.toml | grep "trust_remote_code"

# Clean outputs
rm -rf ~/dakota-rl-training/outputs/*

# Launch training
cd ~/prime-rl
uv run rl --trainer @ ~/dakota-rl-training/configs/train_30b.toml --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml --inference @ ~/dakota-rl-training/configs/infer_30b.toml --trainer-gpu-ids 4,5,6,7 --inference-gpu-ids 0,1,2,3 --output-dir ~/dakota-rl-training/outputs
```

