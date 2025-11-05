# How to Launch RL Training - Simple Guide

## TL;DR: You're on Windows, so use Prime Intellect Cloud Platform

**You CANNOT run training locally on Windows** - prime-rl requires Linux + NVIDIA GPUs.

## The One Way That Works: Prime Intellect Web UI

### Step 1: Go to the Platform
1. Open: **https://app.primeintellect.ai**
2. Log in with your account

### Step 2: Create Training Job
1. Click **"New Training Job"** or **"Create Job"** button
2. Select **"RL Training"** or **"GRPO"**

### Step 3: Configure Instance
- **Instance Type**: Select **8x A100 GPUs**
- **GPU Allocation**:
  - Inference: 4 GPUs (0-3)
  - Trainer: 4 GPUs (4-7)

### Step 4: Upload Config Files
Upload these 3 files from `dakota_rl_training/configs/`:

```
✓ orch_30b.toml       (orchestrator config)
✓ train_30b.toml      (trainer config)
✓ infer_30b.toml      (inference config)
```

**Where to find them:**
```
C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml
```

### Step 5: Set Model & Environment
- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **Environment**: `harleycooper/dakota1890`

### Step 6: Launch
- Click **"Launch Training"**
- Wait 10-30 minutes for instance provisioning
- Training runs ~1.5 hours

### Step 7: Monitor
- Watch logs in Prime Intellect dashboard
- Check Weights & Biases project: `dakota-rl-grammar`

---

## What NOT to Do

❌ **Don't try to run locally on Windows** - It won't work
❌ **Don't use `launch_grpo_30b.ps1`** - That's for Linux instances
❌ **Don't use `launch_primeintellect.py`** - API might not work, use web UI instead

---

## Quick Checklist

Before launching, verify:

- [ ] All 3 config files exist in `dakota_rl_training/configs/`
- [ ] You have Prime Intellect account + API key
- [ ] Environment `harleycooper/dakota1890` is published
- [ ] Model `qwen/qwen3-30b-a3b-instruct-2507` is available

---

## If Something Goes Wrong

### "Config file not found"
→ Make sure you uploaded all 3 files (.toml files)

### "Model not found"
→ Check model name: `qwen/qwen3-30b-a3b-instruct-2507`

### "Environment not found"
→ Verify environment ID: `harleycooper/dakota1890` is published

### "Out of memory"
→ Reduce batch_size in `orch_30b.toml` from 512 to 256

---

## Expected Timeline

- **Provisioning**: 10-30 minutes
- **Training**: ~90 minutes (1.5 hours)
- **Total**: ~2 hours

---

## That's It!

Just upload the 3 config files to the web UI and launch. No command line needed.

