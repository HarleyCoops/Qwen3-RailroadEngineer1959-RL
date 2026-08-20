# COPY-PASTE READY - Your Exact Instance

## Terminal 1: Setup (Copy All at Once)

```bash
# SSH in
ssh ubuntu@69.19.136.157

# Create directories
mkdir -p ~/dakota_rl_training/configs
mkdir -p ~/dakota_rl_training/outputs

# Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Verify GPU
nvidia-smi
```

## Windows PowerShell: Upload Files

```powershell
# Upload config files
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/

# Verify upload
ssh ubuntu@69.19.136.157 "ls -la ~/dakota_rl_training/configs/"
```

## Terminal 2: Launch Training

```bash
# SSH in
ssh ubuntu@69.19.136.157

# Launch training
cd ~/prime-rl
uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b
```

## Terminal 3: Monitor (Optional)

```bash
# SSH in
ssh ubuntu@69.19.136.157

# Watch logs
tail -f ~/dakota_rl_training/outputs/grpo_30b/logs/trainer/rank_0.log
```

---

**SSH: `ssh ubuntu@69.19.136.157`**

