# Launching Prime-RL Training for Qwen 30B-A3B

This guide explains how to launch RL training using the Prime-RL framework.

## Quick Start

### On Prime Intellect Platform (Recommended)

1. **Navigate to Prime Intellect Dashboard**
   - Go to: https://app.primeintellect.ai
   - Log in with your account

2. **Create a New Training Job**
   - Click "New Training Job" or "Create Job"
   - Select "RL Training" or "Prime-RL"

3. **Configure Instance**
   - **Instance Type**: 8x A100 GPUs (or equivalent)
   - **GPU Allocation**:
     - Inference: GPUs 0-3 (4 GPUs)
     - Trainer: GPUs 4-7 (4 GPUs)

4. **Upload Config Files**
   Upload these 3 files from `dakota_rl_training/configs/`:
   ```
    train_30b.toml      (trainer config)
    orch_30b.toml       (orchestrator config)
    infer_30b.toml      (inference config)
   ```

5. **Set Environment Variables** (if needed)
   - Model: `qwen/qwen3-30b-a3b-instruct-2507`
   - Environment ID: `harleycooper/dakota1890`
   - W&B Project: `dakota-rl-grammar`

6. **Launch Training**
   - Click "Launch Training"
   - Monitor logs in the dashboard

### Local/Multi-Node Setup

If running on a Linux machine with NVIDIA GPUs:

#### Single Node (8 GPUs)

```bash
cd prime-rl-framework

uv run rl \
    --trainer @ ../dakota_rl_training/configs/train_30b.toml \
    --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml \
    --inference @ ../dakota_rl_training/configs/infer_30b.toml \
    --trainer-gpu-ids 4,5,6,7 \
    --inference-gpu-ids 0,1,2,3 \
    --output-dir ../dakota_rl_training/outputs/grpo_30b
```

#### Multi-Node Setup

See [Prime-RL Deployment Guide](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/deployment.md) for multi-node instructions.

## Configuration Files

### Trainer Config (`train_30b.toml`)
- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **Max Steps**: 500
- **Learning Rate**: 1e-6
- **Checkpoint Interval**: Every 100 steps
- **W&B Project**: `dakota-rl-grammar`

### Orchestrator Config (`orch_30b.toml`)
- **Environment**: `harleycooper/dakota1890`
- **Batch Size**: 256
- **Micro Batch Size**: 2
- **Sequence Length**: 1536
- **Rollouts per Example**: 8
- **Max Tokens**: 512

### Inference Config (`infer_30b.toml`)
- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **Tensor Parallelism**: 2 (tp=2)
- **Data Parallelism**: 2 (dp=2)
- **Enforce Eager**: true (fixes CUDA graph issues)

## Monitoring

### Weights & Biases
- Project: `dakota-rl-grammar`
- Dashboard: https://wandb.ai/your-username/dakota-rl-grammar

### Logs
Logs are written to:
```
outputs/grpo_30b/logs/
  ├── trainer/
  ├── orchestrator.log
  └── inference.stdout
```

### Checkpoints
Checkpoints are saved to:
```
outputs/grpo_30b/
  ├── checkpoints/step_100/
  ├── checkpoints/step_200/
  └── weights/
```

## Expected Timeline

- **Provisioning**: 10-30 minutes (on Prime Intellect)
- **Training**: ~90 minutes (1.5 hours)
- **Total**: ~2 hours

## Troubleshooting

### "Config file not found"
→ Make sure you're in the `prime-rl-framework` directory and paths are relative

### "Model not found"
→ Verify model name: `qwen/qwen3-30b-a3b-instruct-2507`

### "Environment not found"
→ Verify environment ID: `harleycooper/dakota1890` is published

### "Out of memory"
→ Reduce `batch_size` in `orch_30b.toml` from 256 to 128

### "CUDA error: invalid argument"
→ The `enforce_eager = true` setting in `infer_30b.toml` should fix this

## References

- [Prime-RL GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [Prime-RL Documentation](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/docs)
- [Prime Intellect Dashboard](https://app.primeintellect.ai)

