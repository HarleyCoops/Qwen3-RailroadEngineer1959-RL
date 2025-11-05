# TRAINING STARTED! But Orchestrator Failed

## What Happened

✅ Training started successfully!
- Inference process started on GPUs 0-3
- Trainer process started on GPUs 4-7
- Model loading began

❌ Orchestrator failed with exit code 1

## Check Orchestrator Logs

```bash
# SSH into instance
ssh ubuntu@69.19.136.157

# Check orchestrator logs for error
cat ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.log
# OR
tail -100 ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.stdout
```

## Fixed Command (Missing --inference.parallel.dp)

The command you ran was missing `--inference.parallel.dp 2`. Here's the complete corrected command:

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
  --trainer.wandb.name "dakota-30b-rl-trainer" \
  --orchestrator.batch-size 256 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.seq-len 1536 \
  --orchestrator.mask-truncated-completions \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.env '[{"id": "harleycooper/dakota1890"}]' \
  --orchestrator.max-steps 500 \
  --orchestrator.wandb.project "dakota-rl-grammar" \
  --orchestrator.wandb.name "dakota-30b-rl-orchestrator" \
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

**Key fixes:**
- Added `--inference.parallel.dp 2` (was missing)
- Added `--inference.model.trust-remote-code`
- Fixed orchestrator wandb name

## Common Orchestrator Errors

1. **Environment not found**: Check `harleycooper/dakota1890` is published
2. **Inference server not ready**: Wait longer for inference to start
3. **API key missing**: Check if inference server needs API key
4. **Model loading timeout**: Model might be too large or slow to load

## Next Steps

1. Check orchestrator logs to see exact error
2. Fix the issue
3. Re-run the corrected command above

