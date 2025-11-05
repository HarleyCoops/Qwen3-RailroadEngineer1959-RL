# Prime-RL Launch Guide - Quick Reference

## ✅ Setup Complete

Your Prime-RL training configuration is ready! Here's what's been set up:

### Config Files (Ready to Use)
- ✅ `train_30b.toml` - Trainer configuration
- ✅ `orch_30b.toml` - Orchestrator configuration  
- ✅ `infer_30b.toml` - Inference configuration

### Launch Script
- ✅ `launch_prime_rl_30b.ps1` - PowerShell launch script

## 🚀 How to Launch

### Option 1: Prime Intellect Platform (Recommended)

1. Go to https://app.primeintellect.ai
2. Create new RL training job
3. Upload the 3 config files from `dakota_rl_training/configs/`
4. Set GPU allocation: Inference (0-3), Trainer (4-7)
5. Launch!

### Option 2: Command Line (Linux/Multi-Node)

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

### Option 3: PowerShell Script (Windows → Linux Instance)

```powershell
cd dakota_rl_training
.\launch_prime_rl_30b.ps1
```

## 📋 Configuration Summary

| Setting | Value |
|---------|-------|
| Model | `qwen/qwen3-30b-a3b-instruct-2507` |
| Environment | `harleycooper/dakota1890` |
| Max Steps | 500 |
| Batch Size | 256 |
| Micro Batch Size | 2 |
| Sequence Length | 1536 |
| Rollouts per Example | 8 |
| Learning Rate | 1e-6 |
| Checkpoint Interval | Every 100 steps |
| W&B Project | `dakota-rl-grammar` |

## 📊 GPU Allocation (8 A100s)

- **Inference**: GPUs 0-3 (4 GPUs, dp=2, tp=2)
- **Trainer**: GPUs 4-7 (4 GPUs)

## 📁 Output Structure

```
outputs/grpo_30b/
├── checkpoints/
│   ├── step_100/
│   ├── step_200/
│   └── ...
├── weights/
├── rollouts/
└── logs/
    ├── trainer/
    ├── orchestrator.log
    └── inference.stdout
```

## 🔍 Monitoring

- **W&B Dashboard**: https://wandb.ai/your-username/dakota-rl-grammar
- **Logs**: `outputs/grpo_30b/logs/`
- **Prime Intellect Dashboard**: https://app.primeintellect.ai

## ⏱️ Expected Timeline

- Provisioning: 10-30 min
- Training: ~90 min
- Total: ~2 hours

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Config file not found | Run from `prime-rl-framework` directory |
| Model not found | Verify: `qwen/qwen3-30b-a3b-instruct-2507` |
| Environment not found | Verify: `harleycooper/dakota1890` is published |
| Out of memory | Reduce `batch_size` in `orch_30b.toml` |
| CUDA error | `enforce_eager = true` already set in `infer_30b.toml` |

## 📚 Documentation

- Full guide: `dakota_rl_training/LAUNCH_PRIME_RL.md`
- Prime-RL docs: https://github.com/PrimeIntellect-ai/prime-rl/tree/main/docs
- Environment: https://app.primeintellect.ai/dashboard/environments/harleycooper/dakota1890

## ✅ Pre-Launch Checklist

- [ ] All 3 config files exist
- [ ] Environment `harleycooper/dakota1890` is published
- [ ] Model `qwen/qwen3-30b-a3b-instruct-2507` is accessible
- [ ] W&B project `dakota-rl-grammar` exists (or will be created)
- [ ] Have access to 8 A100 GPUs (or equivalent)
- [ ] Output directory has sufficient disk space

---

**Ready to launch!** 🚀

For detailed instructions, see `LAUNCH_PRIME_RL.md`

