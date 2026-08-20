# RUSH SETUP - Copy everything below into SSH terminal

# Step 1: Setup directories
mkdir -p ~/dakota-rl-training/configs
cd ~/dakota-rl-training/configs

# Step 2: Create orch_30b.toml
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

# Step 3: Create train_30b.toml
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

# Step 4: Create infer_30b.toml
cat > infer_30b.toml << 'EOF'
[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[parallel]
dp = 4
tp = 1
EOF

# Step 5: Go back and install prime-rl
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies (this takes a few minutes)
uv sync && uv sync --all-extras

# Step 6: LAUNCH TRAINING!
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs

