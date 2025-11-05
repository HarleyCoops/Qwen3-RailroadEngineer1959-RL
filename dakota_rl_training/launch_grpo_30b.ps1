# Launch GRPO Training on Prime Intellect
# Model: qwen3-30b-a3b-instruct-2507 on 8 A100s
# 
# GPU Allocation:
#   - Inference: GPUs 0-3 (4 GPUs, data parallelism)
#   - Trainer: GPUs 4-7 (4 GPUs)
#
# Usage:
#   cd prime-rl-framework
#   uv run rl `
#     --trainer @ ../dakota_rl_training/configs/train_30b.toml `
#     --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml `
#     --inference @ ../dakota_rl_training/configs/infer_30b.toml `
#     --trainer-gpu-ids 4,5,6,7 `
#     --inference-gpu-ids 0,1,2,3 `
#     --output-dir ../dakota_rl_training/outputs/grpo_30b

# PowerShell command (Windows):
cd prime-rl-framework
uv run rl `
  --trainer @ ../dakota_rl_training/configs/train_30b.toml `
  --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml `
  --inference @ ../dakota_rl_training/configs/infer_30b.toml `
  --trainer-gpu-ids 4,5,6,7 `
  --inference-gpu-ids 0,1,2,3 `
  --output-dir ../dakota_rl_training/outputs/grpo_30b

# Expected:
#   - Training time: ~1.5 hours (90 minutes)
#   - Dataset: 10,576 examples
#   - Epochs: 3 (via max_steps)
#   - Cost: ~$12-60

