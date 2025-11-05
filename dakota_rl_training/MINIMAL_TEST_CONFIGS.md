# MINIMAL TEST CONFIGS - Test These on Instance

The error suggests Prime-RL is parsing configs incorrectly. Let's test with minimal configs that match the working examples exactly.

## Step 1: Create Minimal Test Configs on Instance

```bash
# SSH into instance
ssh ubuntu@69.19.136.157

# Create minimal trainer config (exactly like working example)
cat > ~/dakota_rl_training/configs/train_30b_minimal.toml << 'EOF'
max_steps = 500

[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"
trust_remote_code = true

[model.ac]
freq = 1

[optim]
lr = 1e-6

[ckpt]
interval = 100
EOF

# Create minimal orchestrator config (exactly like working example)
cat > ~/dakota_rl_training/configs/orch_30b_minimal.toml << 'EOF'
batch_size = 256
micro_batch_size = 2
rollouts_per_example = 8
seq_len = 1536
mask_truncated_completions = false
max_steps = 500

[model]
name = "qwen/qwen3-30b-a3b-instruct-2507"

[sampling]
max_tokens = 512

[environment]
id = "harleycooper/dakota1890"
EOF
```

## Step 2: Test Launch

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b_minimal.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b_minimal.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

## Step 3: If Still Fails, Check Prime-RL Version

The instance might have a different version. Check:

```bash
cd ~/prime-rl
git log --oneline -5
git diff HEAD~1 src/prime_rl/orchestrator/config.py
```

## Step 4: Alternative - Use CLI Args Only

If config files still fail, use CLI args directly (from help output):

```bash
cd ~/prime-rl

uv run rl \
  --trainer.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --trainer.model.trust-remote-code \
  --trainer.model.ac.freq 1 \
  --trainer.optim.lr 1e-6 \
  --trainer.ckpt.interval 100 \
  --trainer.max-steps 500 \
  --orchestrator.batch-size 256 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.seq-len 1536 \
  --orchestrator.mask-truncated-completions \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.env '[{"id": "harleycooper/dakota1890"}]' \
  --orchestrator.max-steps 500 \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

**Note:** Help shows `--orchestrator.env` (not `environment`), so use `--orchestrator.env '[{"id": "harleycooper/dakota1890"}]'`

