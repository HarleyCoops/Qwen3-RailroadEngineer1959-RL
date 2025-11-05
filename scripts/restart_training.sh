#!/bin/bash
# Restart GRPO training with actual API keys from .env

cd ~/prime-rl

export WANDB_API_KEY="cee32d77c7edb39a3857ede1c44fa2c7d7f89bb1"
export WANDB_PROJECT="dakota-rl-grammar"
export HF_TOKEN="your_huggingface_token_here"

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs

