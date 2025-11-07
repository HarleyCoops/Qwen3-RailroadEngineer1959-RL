# Launch Training on Lambda Labs / Hyperbolic Labs

## Overview

Both Lambda Labs and Hyperbolic Labs provide GPU instances that you can SSH into and run Prime-RL directly - much simpler than Prime Intellect's platform.

## Option 1: Lambda Labs

### Step 1: Get Lambda Labs Instance

1. Go to: https://lambdalabs.com
2. Sign up / Log in
3. Create/reserve instance:
   - **Instance Type**: 8x A100 (or equivalent)
   - **Get SSH command** from dashboard

### Step 2: SSH In

```bash
ssh ubuntu@<lambda-labs-ip>
```

### Step 3: Setup on Instance

```bash
# Clone your repo (or transfer files)
git clone https://github.com/HarleyCoops/Dakota1890.git
cd Dakota1890

# OR transfer files via SCP from Windows
# scp -r dakota_rl_training ubuntu@<lambda-labs-ip>:~/

# Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### Step 4: Upload Config Files

**From Windows PowerShell:**
```powershell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml ubuntu@<lambda-labs-ip>:~/dakota_rl_training/configs/
```

### Step 5: Launch Training

```bash
cd ~/prime-rl

uv run rl \
  --trainer.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --trainer.model.trust-remote-code \
  --trainer.model.ac.freq 1 \
  --trainer.optim.lr 1e-6 \
  --trainer.ckpt.interval 100 \
  --trainer.max-steps 500 \
  --trainer.wandb.project "dakota-rl-grammar" \
  --orchestrator.batch-size 256 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.seq-len 1536 \
  --orchestrator.mask-truncated-completions \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.env '[{"id": "harleycooper/dakota1890"}]' \
  --orchestrator.max-steps 500 \
  --orchestrator.wandb.project "dakota-rl-grammar" \
  --orchestrator.ckpt.interval 100 \
  --inference.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --inference.model.enforce-eager \
  --inference.model.trust-remote-code \
  --inference.parallel.tp 2 \
  --inference.parallel.dp 2 \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

---

## Option 2: Hyperbolic Labs

Same process as Lambda Labs:

1. Go to: https://hyperboliclabs.com (or whatever their URL is)
2. Reserve GPU instance
3. SSH in
4. Follow same setup steps

---

## Quick Setup Script

**On Lambda/Hyperbolic instance:**

```bash
#!/bin/bash
# Quick setup script

# Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Verify GPU access
nvidia-smi

# Create config directory
mkdir -p ~/dakota_rl_training/configs
```

---

## Upload Configs from Windows

```powershell
# Replace <instance-ip> with your Lambda/Hyperbolic Labs IP
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml ubuntu@<instance-ip>:~/dakota_rl_training/configs/
```

---

## Launch Command (Same for Both)

Once setup is complete, use the same launch command as above.

---

## Advantages

✅ **Simple**: Just SSH and run commands
✅ **No platform complexity**: Direct Linux access
✅ **Full control**: You control everything
✅ **Easy debugging**: Standard Linux environment
✅ **Works with Prime-RL**: No platform-specific issues

---

**Choose Lambda Labs or Hyperbolic Labs, reserve an instance, SSH in, and run the setup!**




