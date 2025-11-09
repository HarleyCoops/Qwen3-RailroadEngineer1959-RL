# Debug Orchestrator Failure

## Check Orchestrator Logs

The error shown is just the cleanup signal. We need to see the actual orchestrator error:

```bash
# SSH into instance
ssh ubuntu@69.19.136.157

# Check orchestrator logs
cat ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.log

# OR check stdout
cat ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.stdout

# OR check for errors
grep -i error ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.log
grep -i error ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.stdout
```

## Common Orchestrator Failures

1. **Environment not found**: `harleycooper/dakota1890` might not be installed or accessible
2. **Inference server not ready**: Orchestrator connects before inference is ready
3. **API key missing**: If inference server requires authentication
4. **Environment import error**: Python import failure for the environment

## Quick Test - Check Environment

```bash
# Test if environment is accessible
uv run python -c "from verifiers import load_environment; env = load_environment('harleycooper/dakota1890'); print('OK')"
```

If this fails, the environment isn't installed or accessible.

## Install Environment If Needed

```bash
# Install via prime CLI
prime env install harleycooper/dakota1890

# OR via pip
pip install dakota-grammar-translation
```

## Check Inference Server Status

```bash
# Check if inference server started
tail -50 ~/dakota_rl_training/outputs/grpo_30b/logs/inference.stdout

# Check if it's responding
curl http://localhost:8000/v1/models
```

## Fixed Command (Complete)

```bash
cd ~/prime-rl

uv run rl \
  --trainer.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --trainer.model.trust-remote-code \
  --trainer.model.ac.freq 1 \
  --trainer.optim.lr 1e-6 \
  --trainer.ckpt.interval 100 \
  --trainer.max-steps 500 \
  --trainer.wandb.project "dakota-rl-grammar" \
  --orchestrator.batch-size 256 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.seq-len 1536 \
  --orchestrator.mask-truncated-completions \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.env '[{"id": "harleycooper/dakota1890"}]' \
  --orchestrator.max-steps 500 \
  --orchestrator.wandb.project "dakota-rl-grammar" \
  --orchestrator.ckpt.interval 100 \
  --inference.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --inference.model.enforce-eager \
  --inference.model.trust-remote-code \
  --inference.parallel.tp 2 \
  --inference.parallel.dp 2 \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

**Run the log check commands first to see what actually failed!**










