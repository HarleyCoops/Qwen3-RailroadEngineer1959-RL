#!/bin/bash
# Dakota RL Training Launch Script
# Launches training using PrimeIntellect prime-rl framework

set -e

echo "======================================================================"
echo "DAKOTA GRAMMAR RL TRAINING - PRIMEINTELLECT LAUNCH"
echo "======================================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv package manager not found"
    echo ""
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source \$HOME/.local/bin/env"
    exit 1
fi

# Check for PI_API_KEY
if [ -z "$PI_API_KEY" ]; then
    echo "[ERROR] PI_API_KEY not set"
    echo ""
    echo "Please set your PrimeIntellect API key:"
    echo "  export PI_API_KEY=your_key_here"
    echo ""
    echo "Or add to .env file:"
    echo "  PI_API_KEY=your_key_here"
    exit 1
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  Algorithm: GRPO (Group Relative Policy Optimization)"
echo "  Dataset: 1,998 easy tasks (curriculum stage 1)"
echo "  Epochs: 3"
echo "  TOPLOC Verification: Enabled"
echo ""

# Check if datasets exist
if [ ! -f "datasets/grammar_tasks_easy.jsonl" ]; then
    echo "[ERROR] Dataset not found: datasets/grammar_tasks_easy.jsonl"
    exit 1
fi

echo "[INFO] Datasets verified"
echo ""

# Option 1: Single GPU training (recommended for local testing)
echo "======================================================================"
echo "OPTION 1: Single GPU Training (Local)"
echo "======================================================================"
echo ""
echo "Run this command:"
echo ""
echo "uv run rl \\"
echo "  --trainer @ configs/train.toml \\"
echo "  --orchestrator @ configs/orch.toml \\"
echo "  --inference @ configs/infer.toml \\"
echo "  --trainer-gpu-ids 0 \\"
echo "  --inference-gpu-ids 0"
echo ""

# Option 2: Multi-node distributed training
echo "======================================================================"
echo "OPTION 2: Multi-Node Distributed Training (PrimeIntellect Cloud)"
echo "======================================================================"
echo ""
echo "1. Upload to PrimeIntellect platform:"
echo "   - configs/train.toml"
echo "   - configs/infer.toml"
echo "   - configs/orch.toml"
echo "   - datasets/grammar_tasks_easy.jsonl"
echo ""
echo "2. Launch via PrimeIntellect dashboard:"
echo "   https://app.primeintellect.ai"
echo ""

# Option 3: Component-based execution
echo "======================================================================"
echo "OPTION 3: Component-Based Execution (Advanced)"
echo "======================================================================"
echo ""
echo "Terminal 1 - Inference server:"
echo "  uv run inference @ configs/infer.toml"
echo ""
echo "Terminal 2 - Trainer:"
echo "  uv run trainer @ configs/train.toml"
echo ""
echo "Terminal 3 - Orchestrator:"
echo "  uv run orchestrator @ configs/orch.toml"
echo ""

echo "======================================================================"
echo "MONITORING"
echo "======================================================================"
echo ""
echo "Track training progress:"
echo "  - Weights & Biases: https://wandb.ai"
echo "  - Checkpoints: dakota_rl_training/checkpoints/"
echo "  - Logs: View in terminal or W&B dashboard"
echo ""
echo "Key metrics:"
echo "  - reward/mean: Average reward per episode"
echo "  - char_accuracy: Dakota character preservation"
echo "  - affix_accuracy: Morphology accuracy"
echo "  - semantic_accuracy: Translation correctness"
echo ""

echo "======================================================================"
echo "CURRICULUM STAGES"
echo "======================================================================"
echo ""
echo "Stage 1: Easy tasks (1,998 tasks) - Current"
echo "  Min accuracy: 80%"
echo "  Expected time: 2-4 hours"
echo ""
echo "Stage 2: Medium tasks (2,155 tasks)"
echo "  Min accuracy: 75%"
echo "  Expected time: 3-5 hours"
echo ""
echo "Stage 3: Hard tasks (398 tasks)"
echo "  Min accuracy: 70%"
echo "  Expected time: 1-2 hours"
echo ""
echo "Total expected time: 6-11 hours"
echo ""

echo "======================================================================"
echo "READY TO LAUNCH"
echo "======================================================================"
echo ""
echo "Choose an option above and run the corresponding command"
echo ""
