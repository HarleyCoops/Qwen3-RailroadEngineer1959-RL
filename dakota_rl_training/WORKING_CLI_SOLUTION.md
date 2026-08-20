# WORKING SOLUTION - Use CLI Args (Based on Help Output)

The config file parsing has issues. Use CLI args directly from the help output.

## Launch Command (Copy-Paste Ready)

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
  --trainer.wandb.name "dakota-30b-rl" \
  --orchestrator.batch-size 256 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.seq-len 1536 \
  --orchestrator.mask-truncated-completions \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.env '[{"id": "harleycooper/dakota1890"}]' \
  --orchestrator.max-steps 500 \
  --orchestrator.wandb.project "dakota-rl-grammar" \
  --orchestrator.wandb.name "dakota-30b-rl" \
  --orchestrator.ckpt.interval 100 \
  --inference.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --inference.model.enforce-eager \
  --inference.parallel.tp 2 \
  --inference.parallel.dp 2 \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --wandb.project "dakota-rl-grammar" \
  --wandb.name "dakota-30b-rl"
```

**Key changes:**
- Removed `micro_batch_size` (not in help output - might use default)
- Used `--orchestrator.env` with JSON array format (not `--orchestrator.environment`)
- Set all fields via CLI args instead of config files

## If micro_batch_size is Needed

If you need `micro_batch_size`, try adding it after other orchestrator args:
```bash
--orchestrator.micro-batch-size 2 \
```

But since it's not in the help output, it might not be supported via CLI or might use a default.

