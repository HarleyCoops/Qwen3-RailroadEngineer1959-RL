# Troubleshooting Checkpoint Save Error

## Error Description

The training is failing with:
```
File "/workspace/prime-rl/src/prime_rl/trainer/weights.py", line 238, in save_state_dict
    torch.save(state_dict, save_dir / weights_name)
  File "/workspace/prime-rl/.venv/lib/python3.12/site-packages/torch/serialization.py", line 966, in save
    with _open_zipfile_writer(f) as opened_zipfile:
```

This error occurs when `torch.save()` cannot write the checkpoint file, typically due to:

1. **Disk space full** (most common)
2. **Permission issues**
3. **Inode exhaustion**
4. **Concurrent write conflicts**

## Quick Diagnosis

SSH into your server and run:

```bash
# Check disk space
df -h

# Check your output directory
du -sh ~/dakota_rl_training/outputs/grpo_30b/weights 2>/dev/null || echo "Weights directory not found"

# Check for temporary files that might be stuck
find ~/dakota_rl_training/outputs/grpo_30b -name "*.tmp" -ls

# Check inodes (if filesystem is full of small files)
df -i
```

## Solutions

### Solution 1: Free Up Disk Space

**Clean up old checkpoints:**

```bash
# Navigate to output directory
cd ~/dakota_rl_training/outputs/grpo_30b

# List checkpoint sizes
du -sh weights/step_* | sort -h

# Remove old checkpoints (keep only recent ones)
# WARNING: This will delete old checkpoints!
# Adjust the step numbers based on what you want to keep
rm -rf weights/step_0 weights/step_1 weights/step_2  # etc.

# Or remove all except the last N steps
ls -t weights/step_* | tail -n +11 | xargs rm -rf  # Keeps last 10 steps
```

**Clean up temporary files:**

```bash
# Remove any stuck .tmp files
find ~/dakota_rl_training/outputs -name "*.tmp" -delete

# Clean up Python cache
find ~/prime-rl -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
```

### Solution 2: Reduce Checkpoint Frequency

Edit your `train_30b.toml` to save checkpoints less frequently:

```toml
[ckpt]
interval = 200  # Instead of 100, saves every 200 steps

# If you have a [weights] section, increase its interval too
[weights]
interval = 200
save_async = true  # Keep async to avoid blocking training
```

### Solution 3: Use Ephemeral Storage

If you're running out of space on the main disk, use `/ephemeral` storage:

1. **Stop the current training** (if running)

2. **Move existing outputs to ephemeral:**
```bash
mkdir -p /ephemeral/dakota_rl_training/outputs
mv ~/dakota_rl_training/outputs/grpo_30b /ephemeral/dakota_rl_training/outputs/
```

3. **Update your launch command** to use ephemeral storage:
```bash
uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir /ephemeral/dakota_rl_training/outputs/grpo_30b
```

**Note:** `/ephemeral` data may be lost if the instance is stopped, so copy important checkpoints elsewhere periodically.

### Solution 4: Disable Async Saving (Temporary Fix)

If async saving is causing issues, you can disable it. However, this will slow down training:

Edit `train_30b.toml`:
```toml
[weights]
interval = 100
save_async = false  # Disable async saving
```

### Solution 5: Check and Fix Permissions

```bash
# Ensure output directory is writable
chmod -R 755 ~/dakota_rl_training/outputs
chown -R $USER:$USER ~/dakota_rl_training/outputs

# If running as root, ensure proper ownership
# chown -R root:root ~/dakota_rl_training/outputs
```

## Prevention

1. **Monitor disk space regularly:**
```bash
watch -n 60 'df -h && echo "" && du -sh ~/dakota_rl_training/outputs/grpo_30b/weights'
```

2. **Set up automatic cleanup** in your training script or use cron

3. **Use checkpoint cleanup** - The framework has built-in cleanup, ensure `keep` is set appropriately in your config

4. **Increase checkpoint interval** for long training runs

## Resuming Training

If training crashed due to this error:

1. **Fix the disk space issue** using solutions above

2. **Check if you have a valid checkpoint:**
```bash
ls -lah ~/dakota_rl_training/outputs/grpo_30b/checkpoints/
```

3. **Resume from the last checkpoint** (the framework should auto-detect, or specify explicitly)

## Getting More Information

Run the diagnostic script:
```bash
bash scripts/diagnose_checkpoint_error.sh ~/dakota_rl_training/outputs/grpo_30b
```

Or check the full error logs:
```bash
tail -100 ~/dakota_rl_training/outputs/grpo_30b/logs/trainer/rank_0.log
```

