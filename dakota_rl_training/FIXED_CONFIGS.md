# Fixed Config Files - Upload These to Instance

The configs have been fixed to remove `max_steps` which will be set via CLI.

## Updated Files:

### train_30b.toml
- Removed `max_steps` (will be set via CLI with `--max-steps` or in orchestrator)

### orch_30b.toml  
- Removed `max_steps` (will be set via CLI)

## Upload Fixed Configs:

```powershell
# From Windows PowerShell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml ubuntu@69.19.136.157:~/dakota_rl_training/configs/
```

## Updated Launch Command:

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b \
  --max-steps 500
```

**Note:** Added `--max-steps 500` to the CLI command to set it at the top level.

