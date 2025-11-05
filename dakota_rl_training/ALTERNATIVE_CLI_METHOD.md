# ALTERNATIVE: Try Without Nested Config Parsing

If the `--trainer @` syntax is causing issues, try passing fields directly via CLI:

```bash
cd ~/prime-rl

uv run rl \
  --trainer.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --trainer.model.trust-remote-code true \
  --trainer.optim.lr 1e-6 \
  --trainer.ckpt.interval 100 \
  --trainer.wandb.project "dakota-rl-grammar" \
  --orchestrator.batch-size 256 \
  --orchestrator.seq-len 1536 \
  --orchestrator.rollouts-per-example 8 \
  --orchestrator.micro-batch-size 2 \
  --orchestrator.model.name "qwen/qwen3-30b-a3b-instruct-2507" \
  --orchestrator.sampling.max-tokens 512 \
  --orchestrator.environment.id "harleycooper/dakota1890" \
  --orchestrator.wandb.project "dakota-rl-grammar" \
  --orchestrator.ckpt.interval 100 \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --max-steps 500
```

**Note:** This bypasses the config file parsing issue but is more verbose.

