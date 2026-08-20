# Restart Training After OOM Fix

## Confirmed: Training Crashed

The logs show:
- **OOM Error**: CUDA out of memory during first training step
- **Exit Code**: 1 (trainer failed)
- **Time**: Crashed ~30 minutes after startup
- **Status**: All processes terminated

## Fix Applied

I've reduced memory usage in `orch_30b.toml`:
- `batch_size`: 512 → **256** (50% reduction)
- `rollouts_per_example`: 16 → **8** (50% reduction)  
- `seq_len`: 2048 → **1536** (25% reduction)

## Restart Steps

### 1. Kill Any Hanging Processes (SSH)

```bash
# Kill all Python processes
pkill -9 python

# Verify GPUs are free
nvidia-smi
```

### 2. Upload Fixed Config (Local PowerShell)

```powershell
scp dakota_rl_training/configs/orch_30b.toml ubuntu@204.52.25.227:~/dakota-rl-training/configs/
```

### 3. Set Environment Variables (SSH)

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="dakota-rl-grammar"
export HF_TOKEN="your_huggingface_token_here"
```

### 4. Restart Training (SSH)

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
```

## Expected Memory Usage (After Fix)

- **Before**: ~80GB per GPU (OOM!)
- **After**: ~60-65GB per GPU (should fit!)
- **Free memory**: ~15-20GB per GPU

## Monitoring

Watch GPU memory:
```bash
watch -n 1 nvidia-smi
```

Should see GPUs stabilizing around 60-65GB usage.

## If Still OOM

Try even more aggressive reduction:

```toml
batch_size = 128
rollouts_per_example = 4
seq_len = 1024
```

But let's try the current fix first - it should work!

