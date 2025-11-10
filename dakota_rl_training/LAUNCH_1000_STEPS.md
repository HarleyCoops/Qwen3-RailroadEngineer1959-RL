# Launch RL Training - 1000 Steps (Disk Error Prevention)

## Quick Launch Command

**On your Prime Intellect instance (SSH):**

```bash
cd ~/prime-rl

# Set environment variables (if not already set)
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="dakota-rl-grammar"

# Launch training with config files
uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b_1000steps \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-1000steps"
```

## What Changed

- **max_steps**: Updated from 500 → **1000** in both `train_30b.toml` and `orch_30b.toml`
- **Checkpoint interval**: Still 100 steps (will create 10 checkpoints total)
- **Environment**: Using `harleycooper/dakota1890` (version 0.1.8 with verbose penalty)
- **Output directory**: Changed to `grpo_30b_1000steps` to avoid conflicts

## Upload Configs First

**From Windows PowerShell:**

```powershell
# Replace <instance-ip> with your Prime Intellect instance IP
scp dakota_rl_training\configs\*.toml root@<instance-ip>:/root/dakota_rl_training/configs/
```

## Monitor Training

1. **W&B Dashboard**: https://wandb.ai
   - Project: `dakota-rl-grammar`
   - Run: `dakota-30b-1000steps`

2. **Check logs on instance:**
   ```bash
   tail -f ~/dakota_rl_training/outputs/grpo_30b_1000steps/logs/trainer.log
   ```

3. **Check GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

## Expected Timeline

- **1000 steps** with checkpoint every 100 steps
- **10 checkpoints** total
- Estimated time: ~8-12 hours (depending on instance speed)

## If Disk Error Occurs

The training will stop at the last successful checkpoint. To resume:

```bash
# The config already has resume_step = -1 (auto-resume)
# Just re-run the same launch command - it will auto-resume from latest checkpoint
```

## Checkpoints Location

```
~/dakota_rl_training/outputs/grpo_30b_1000steps/weights/
  - step_100/
  - step_200/
  - step_300/
  ...
  - step_1000/
```





