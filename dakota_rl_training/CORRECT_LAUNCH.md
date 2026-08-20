# CORRECT LAUNCH COMMAND - Based on Prime-RL Docs

## Upload Fixed Configs

```powershell
# From Windows PowerShell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
```

## Launch Training

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

**Key changes:**
- Removed `recompute_logprobs` from trainer config
- Removed `[wandb]` sections from configs (set via CLI)
- Removed `[ckpt]` from orchestrator config (uses trainer's ckpt)
- Added `max_steps` to both configs
- Set `wandb.project` and `wandb.name` via CLI

This matches the exact structure from the working examples in Prime-RL docs.

