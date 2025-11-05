#!/bin/bash
# Debug script for Prime RL inference server issues
# Run this on your remote server to investigate the crash

echo "=== Checking Inference Server Logs ==="
echo ""
echo "--- Last 200 lines of inference.stdout ---"
tail -200 ~/dakota-rl-training/outputs/logs/inference.stdout
echo ""
echo ""

echo "=== Searching for Errors ==="
echo "--- CUDA/OOM/Memory errors ---"
grep -i "cuda\|oom\|out of memory\|torch.cuda.OutOfMemoryError\|RuntimeError" ~/dakota-rl-training/outputs/logs/inference.stdout | tail -50
echo ""
echo ""

echo "=== Searching for vLLM-specific errors ==="
grep -i "error\|exception\|failed\|traceback" ~/dakota-rl-training/outputs/logs/inference.stdout | tail -50
echo ""
echo ""

echo "=== Current GPU Memory Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo ""

echo "=== Inference Config Check ==="
echo "--- infer_30b.toml ---"
cat ~/dakota-rl-training/configs/infer_30b.toml
echo ""
echo ""

echo "=== Orchestrator Logs ==="
echo "--- Last 100 lines ---"
tail -100 ~/dakota-rl-training/outputs/logs/orchestrator.log 2>/dev/null || echo "Orchestrator log not found"
echo ""
echo ""

echo "=== Trainer Logs (Rank 0) ==="
echo "--- Last 50 lines ---"
tail -50 ~/dakota-rl-training/outputs/logs/trainer/rank_0.log 2>/dev/null || echo "Trainer logs not found"
