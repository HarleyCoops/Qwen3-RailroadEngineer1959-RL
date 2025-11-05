#!/bin/bash
# Fix OOM by reducing memory usage

cd ~/dakota-rl-training/configs

# 1. Reduce batch_size from 256 to 128
sed -i 's/batch_size = 256/batch_size = 128/' orch_30b.toml

# 2. Reduce seq_len from 1536 to 1024
sed -i 's/seq_len = 1536/seq_len = 1024/' orch_30b.toml

# 3. Reduce rollouts_per_example from 8 to 4 (less memory during rollout collection)
sed -i 's/rollouts_per_example = 8/rollouts_per_example = 4/' orch_30b.toml

# 4. Add gpu_memory_utilization to infer_30b.toml if not present
if ! grep -q "gpu_memory_utilization" infer_30b.toml; then
    sed -i '/\[parallel\]/a gpu_memory_utilization = 0.85' infer_30b.toml
fi

echo "Fixed configs:"
echo "=== orch_30b.toml ==="
cat orch_30b.toml | grep -E "batch_size|seq_len|rollouts"
echo ""
echo "=== infer_30b.toml ==="
cat infer_30b.toml

