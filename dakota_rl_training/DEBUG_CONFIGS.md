# DEBUG COMMANDS - Run These on Instance

## Check Prime-RL Version

```bash
cd ~/prime-rl
git log --oneline -1
git branch
git status
```

## Check Config Structure

```bash
# See what the parser expects
cd ~/prime-rl
uv run python -c "from prime_rl.trainer.rl.config import RLTrainerConfig; import inspect; print([f.name for f in RLTrainerConfig.model_fields.values()])"
```

## Try Minimal Configs

Create minimal test configs to see what works:

```bash
# Test trainer config
cat > /tmp/test_train.toml << EOF
[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[optim]
lr = 1e-6
EOF

# Test orchestrator config  
cat > /tmp/test_orch.toml << EOF
batch_size = 256
seq_len = 1536
rollouts_per_example = 8

[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[sampling]
max_tokens = 512

[environment]
id = "harleycooper/dakota1890"
EOF

# Try running with minimal configs
cd ~/prime-rl
uv run rl \
  --trainer @ /tmp/test_train.toml \
  --orchestrator @ /tmp/test_orch.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --max-steps 500
```

If this works, gradually add fields back to find which ones cause issues.

