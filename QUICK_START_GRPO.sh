# QUICK START - Copy-Paste These Commands

## Step 1: Create Directory & Install prime-rl

```bash
# Create project directory
mkdir -p ~/dakota-rl-training/configs
cd ~/dakota-rl-training/configs

# Go back and install prime-rl
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync && uv sync --all-extras
```

## Step 2: Create Config Files

```bash
cd ~/dakota-rl-training/configs

# Create orch_30b.toml
cat > orch_30b.toml << 'EOF'
max_steps = 500
batch_size = 512
micro_batch_size = 2
seq_len = 2048
rollouts_per_example = 16
mask_truncated_completions = false

[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[sampling]
max_tokens = 512

[wandb]
project = "dakota-rl-grammar"

[wandb.log_extras]
interval = 10

[environment]
id = "harleycooper/dakota1890"

[eval]
interval = 50
environment_ids = ["harleycooper/dakota1890"]
rollouts_per_example = [1]

[ckpt]
interval = 100
EOF

# Create train_30b.toml
cat > train_30b.toml << 'EOF'
max_steps = 500

[wandb]
project = "dakota-rl-grammar"

[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[optim]
lr = 1e-6

[ckpt]
interval = 100
EOF

# Create infer_30b.toml
cat > infer_30b.toml << 'EOF'
[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[parallel]
dp = 4
tp = 1
EOF
```

## Step 3: LAUNCH TRAINING!

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
```

## That's it! Training will start immediately.

Monitor in:
- Terminal output (real-time logs)
- Weights & Biases: dakota-rl-grammar project
- Prime Intellect dashboard

Expected time: ~1.5 hours

