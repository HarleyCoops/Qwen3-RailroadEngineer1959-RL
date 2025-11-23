#!/bin/bash
# Launch Dakota RL Training - 1000 Steps
# Based on CORRECT_LAUNCH.md and Prime Intellect RFT docs

set -e

echo "======================================================================"
echo "DAKOTA RL TRAINING - 1000 STEPS"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "/workspace/prime-rl" ]; then
    echo "Error: /workspace/prime-rl not found"
    echo "Make sure you're on a Prime Intellect instance"
    exit 1
fi

cd /workspace/prime-rl

# Check if wandb is logged in
if ! uv run wandb whoami &>/dev/null; then
    echo "  W&B not logged in. Logging in..."
    uv run wandb login
fi

# Create output directory
mkdir -p /workspace_1/dakota_rl_outputs

# Install Dakota environment if not already installed
echo "Checking Dakota environment..."
if ! uv run python -c "import dakota_grammar_translation" 2>/dev/null; then
    echo "Installing Dakota environment..."
    uv run prime env install harleycooper/dakota1890
fi

echo ""
echo "Configuration:"
echo "  Model: qwen/qwen3-30b-a3b-instruct-2507"
echo "  Environment: harleycooper/dakota1890"
echo "  Max Steps: 1000"
echo "  Checkpoints: Every 100 steps (keep 3 most recent)"
echo "  Inference GPUs: 0,1,2,3"
echo "  Trainer GPUs: 4,5,6,7"
echo "  Output Directory: /workspace_1/dakota_rl_outputs"
echo "  W&B Project: dakota-rl-grammar"
echo "  W&B Run Name: dakota-30b-rl-1000steps"
echo ""

# Launch training
uv run rl \
  --trainer @ ../dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml \
  --inference @ ../dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir /workspace_1/dakota_rl_outputs \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl-1000steps"

