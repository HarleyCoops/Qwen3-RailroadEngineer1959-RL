# Quick Launch - Lambda Labs / Hyperbolic Labs

## 1. Reserve Instance

**Lambda Labs**: https://lambdalabs.com
- Reserve 8x A100 GPUs
- Get SSH command

**Hyperbolic Labs**: Check their website
- Reserve 8x A100 GPUs  
- Get SSH command

## 2. SSH In

```bash
ssh ubuntu@<instance-ip>
```

## 3. Quick Setup

```bash
# Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Verify GPUs
nvidia-smi

# Create config directory
mkdir -p ~/dakota_rl_training/configs
```

## 4. Upload Configs (From Windows PowerShell)

```powershell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml ubuntu@<instance-ip>:~/dakota_rl_training/configs/
```

## 5. Launch Training

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

**That's it!** Much simpler than Prime Intellect's platform.










