# GRPO Training Launch Checklist

## Pre-Launch Verification

### [COMPLETE] Configuration Files Ready
- [x] `orch_30b.toml` - Orchestrator config
- [x] `train_30b.toml` - Trainer config  
- [x] `infer_30b.toml` - Inference config

### [COMPLETE] Configuration Settings Verified
- [x] Model: `qwen/qwen3-30b-a3b-instruct-2507`
- [x] Environment: `harleycooper/dakota1890`
- [x] GPU Allocation: 8x A100 (4 inference + 4 trainer)
- [x] Batch size: 512
- [x] Learning rate: 1e-6
- [x] Max steps: 500
- [x] WandB project: `dakota-rl-grammar`

## Launch Steps

### Step 1: Access Prime Intellect Platform
- [ ] Go to https://app.primeintellect.ai
- [ ] Log in with your account

### Step 2: Create New Training Job
- [ ] Click "New Training Job" or "Create Job"
- [ ] Select "RL Training" or "GRPO"

### Step 3: Configure Instance
- [ ] Select **8x A100** GPU instance
- [ ] Choose preferred region (lowest latency)

### Step 4: Upload Config Files
Upload from `dakota_rl_training/configs/`:
- [ ] Upload `orch_30b.toml` (orchestrator)
- [ ] Upload `train_30b.toml` (trainer)
- [ ] Upload `infer_30b.toml` (inference)

### Step 5: Set Model & Environment
- [ ] Model: `qwen/qwen3-30b-a3b-instruct-2507`
- [ ] Environment: `harleycooper/dakota1890`

### Step 6: Review Settings
Confirm:
- [ ] 8x A100 GPUs selected
- [ ] All 3 config files uploaded
- [ ] Model name correct
- [ ] Environment ID correct
- [ ] Max steps: 500

### Step 7: Launch
- [ ] Click "Launch Training"
- [ ] Wait for instance provisioning (10-30 min)

## Post-Launch Monitoring

### Prime Intellect Dashboard
- [ ] Monitor job status (provisioning -> running -> completed)
- [ ] Check logs for errors
- [ ] Monitor GPU utilization

### Weights & Biases (Optional)
- [ ] Open WandB project: `dakota-rl-grammar`
- [ ] Monitor metrics: reward/mean, char_accuracy
- [ ] Track training progress

## Quick Reference

**File Locations:**
```
dakota_rl_training/configs/orch_30b.toml
dakota_rl_training/configs/train_30b.toml
dakota_rl_training/configs/infer_30b.toml
```

**Key Settings:**
- Model: `qwen/qwen3-30b-a3b-instruct-2507`
- Environment: `harleycooper/dakota1890`
- GPU: 8x A100 (4 inference + 4 trainer)
- Steps: 500
- Batch: 512
- LR: 1e-6

**Expected Timeline:**
- Provisioning: 10-30 minutes
- Training: ~90 minutes (1.5 hours)
- Total: ~2 hours

**GPU Allocation:**
```
GPUs 0-3: Inference Server (dp=4, tp=1)
GPUs 4-7: Trainer (FSDP2)
```

## Troubleshooting

**Config file not found:**
- Verify all 3 files uploaded
- Check file names match exactly

**Model not found:**
- Verify: `qwen/qwen3-30b-a3b-instruct-2507`
- Check model availability in Prime Intellect registry

**Environment not found:**
- Verify: `harleycooper/dakota1890`
- Confirm environment is published

**Out of memory:**
- Reduce batch_size to 256 in orch_30b.toml
- Reduce rollouts_per_example to 8

## After Training

- [ ] Download checkpoints from dashboard
- [ ] Run evaluation with trained weights
- [ ] Compare with baseline results
- [ ] Document findings

## Notes

- Evals can run in parallel during training
- Training is fully cloud-based (no local setup needed)
- Checkpoints saved every 100 steps
- Eval runs every 50 steps
