#!/bin/bash
# Prime Intellect RFT Validation Script
# Based on: https://docs.primeintellect.ai/reinforcement-fine-tuning

set -e

echo "=== Prime Intellect RFT Validation ==="
echo ""

# Step 1: Check Python version (should be 3.12)
echo "1. Checking Python version..."
cd /workspace/prime-rl
uv run python -V
echo ""

# Step 2: Check flash-attn
echo "2. Checking flash-attn installation..."
uv run python -c "import flash_attn; print('✓ flash-attn installed')" || echo "✗ flash-attn NOT installed"
echo ""

# Step 3: Check SFT trainer debug (requires 1 GPU)
echo "3. Testing SFT trainer debug mode (1 GPU)..."
echo "   Run: uv run sft @ configs/debug/sft/train.toml"
echo ""

# Step 4: Check RL trainer debug (requires 1 GPU)
echo "4. Testing RL trainer debug mode (1 GPU)..."
echo "   Run: uv run trainer @ configs/debug/rl/train.toml"
echo ""

# Step 5: Check orchestrator + inference (requires 1 GPU each)
echo "5. Testing orchestrator + inference..."
echo "   Terminal 1: uv run inference @ configs/debug/infer.toml"
echo "   Terminal 2: uv run orchestrator @ configs/debug/orch.toml"
echo ""

# Step 6: Check SFT warmup (requires 1 GPU)
echo "6. Testing SFT warmup (1 GPU)..."
echo "   Run: uv run sft @ configs/reverse_text/sft/train.toml"
echo ""

# Step 7: Check toy RL run (requires 2 GPUs)
echo "7. Testing toy RL run (2 GPUs)..."
echo "   Run: uv run rl \\"
echo "     --trainer @ configs/reverse_text/rl/train.toml \\"
echo "     --orchestrator @ configs/reverse_text/rl/orch.toml \\"
echo "     --inference @ configs/reverse_text/rl/infer.toml"
echo ""

echo "=== Validation Checklist Complete ==="
echo ""
echo "Next steps:"
echo "1. Run the tests above to validate your setup"
echo "2. Install your Dakota environment: uv run prime env install harleycooper/dakota1890"
echo "3. Configure your RFT run with your config files"
echo "4. Launch: uv run rl --trainer @ configs/train.toml --orchestrator @ configs/orch.toml --inference @ configs/infer.toml"

