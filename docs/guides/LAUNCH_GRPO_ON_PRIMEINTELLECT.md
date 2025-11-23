# Launch GRPO Training on Prime Intellect Cloud Platform

## Overview

You need to create a training job on Prime Intellect's cloud platform. The configs are ready - you just need to upload them and configure the instance.

## Step-by-Step Guide

### Step 1: Go to Prime Intellect Platform

1. Visit: https://app.primeintellect.ai
2. Log in with your account (PI_API_KEY)

### Step 2: Create New Training Job

1. Click **"New Training Job"** or **"Create Job"** (button location varies by UI)
2. Select **"RL Training"** or **"Custom Training"**
3. Choose **"GRPO"** as the algorithm

### Step 3: Configure Instance

**GPU Resources:**
- **Instance Type**: Select **8x A100** GPUs
- **GPU Allocation**:
  - **Inference**: 4 GPUs (GPUs 0-3)
  - **Trainer**: 4 GPUs (GPUs 4-7)
- **Region**: Choose closest to you for lower latency

### Step 4: Upload Configuration Files

Upload these files from `dakota_rl_training/configs/`:

1. **orchestrator config**: `orch_30b.toml`
   - Contains: batch_size, rollouts_per_example, environment ID
   - This is the main orchestrator config

2. **trainer config**: `train_30b.toml`
   - Contains: optimizer settings, learning rate, wandb project
   - Training-specific settings

3. **inference config**: `infer_30b.toml`
   - Contains: model name, parallel config (dp=4, tp=1)
   - Inference server settings

**Files to upload:**
```
dakota_rl_training/configs/orch_30b.toml
dakota_rl_training/configs/train_30b.toml
dakota_rl_training/configs/infer_30b.toml
```

### Step 5: Configure Model

- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **Environment**: `harleycooper/dakota1890`
- **Dataset**: The environment loads its own dataset (10,576 examples)

### Step 6: Set Environment Variables

If needed, configure:
- `WANDB_PROJECT=dakota-rl-grammar` (for Weights & Biases logging)
- `PI_API_KEY` (should already be set)

### Step 7: Review and Launch

1. **Review settings**:
   - Model: qwen3-30b-a3b-instruct-2507
   - GPUs: 8x A100
   - Algorithm: GRPO
   - Environment: harleycooper/dakota1890
   - Max steps: 500

2. **Estimated time**: ~1.5 hours (90 minutes)
3. **Estimated cost**: ~$12-60

4. **Click "Launch Training"**

### Step 8: Monitor Training

**Prime Intellect Dashboard:**
- View logs in real-time
- Monitor GPU utilization
- Check job status

**Weights & Biases** (if configured):
- Project: `dakota-rl-grammar`
- Metrics: reward/mean, char_accuracy, etc.

## Config Files Summary

### orch_30b.toml
- **batch_size**: 512
- **rollouts_per_example**: 16
- **max_steps**: 500
- **environment**: harleycooper/dakota1890
- **model**: qwen/qwen3-30b-a3b-instruct-2507

### train_30b.toml
- **learning_rate**: 1e-6
- **wandb_project**: dakota-rl-grammar
- **checkpoint_interval**: 100 steps

### infer_30b.toml
- **model**: qwen/qwen3-30b-a3b-instruct-2507
- **dp**: 4 (data parallelism - 4 GPUs)
- **tp**: 1 (tensor parallelism - single GPU per model)

## GPU Allocation (8 A100s)

```
GPUs 0-3: Inference Server (4 GPUs, data parallel)
GPUs 4-7: Trainer (4 GPUs, FSDP2)
```

## Expected Timeline

- **Job start**: 10-30 minutes (instance provisioning)
- **Training time**: ~1.5 hours (90 minutes)
- **Total**: ~2 hours

## Troubleshooting

### "Config file not found"
- Make sure you uploaded all 3 config files
- Check file paths are correct

### "Model not found"
- Verify model name: `qwen/qwen3-30b-a3b-instruct-2507`
- Check if model is available in Prime Intellect's model registry

### "Environment not found"
- Verify environment ID: `harleycooper/dakota1890`
- Make sure environment is published (version 0.1.1)

### "Out of memory"
- Reduce batch_size in `orch_30b.toml` (e.g., 256 instead of 512)
- Reduce rollouts_per_example (e.g., 8 instead of 16)

## After Training Completes

1. **Download checkpoints** from Prime Intellect dashboard
2. **Run evaluation** on trained model:
   ```powershell
   prime env eval harleycooper/dakota1890 `
     -m qwen/qwen3-30b-a3b-instruct-2507 `
     -n 100 `
     --weights-dir <checkpoint_path>
   ```
3. **Compare results** with baseline eval

## Summary

**What you need to do:**
1.  Configs are ready (already created)
2.  Go to https://app.primeintellect.ai
3.  Create new RL training job
4.  Select 8x A100 instance
5.  Upload 3 config files (orch_30b.toml, train_30b.toml, infer_30b.toml)
6.  Set model: qwen/qwen3-30b-a3b-instruct-2507
7.  Set environment: harleycooper/dakota1890
8.  Launch training

**You don't need to:**
-  Run commands locally (Windows can't run prime-rl)
-  Set up Linux VM
-  Install dependencies locally

Just upload the configs to Prime Intellect's platform!

