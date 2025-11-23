# Where Are Training Logs Stored?

## Log Directory Structure

Based on your launch command (`--output-dir ~/dakota-rl-training/outputs`), logs are stored in:

```
~/dakota-rl-training/outputs/
├── logs/
│   ├── rl.log                    # Main RL coordinator log
│   ├── trainer.stdout            # Trainer stdout/stderr (MAIN TRAINING LOG)
│   ├── orchestrator.stdout       # Orchestrator logs
│   └── inference.stdout          # Inference server logs
├── torchrun/                     # Distributed training logs (per rank)
│   └── {rdzv_id}/
│       └── attempt_0/
│           └── {rank}/
│               ├── stdout.log
│               └── stderr.log
└── wandb/                        # WandB run data
    └── run-{run_id}/
```

## Quick Check Commands (SSH)

### 1. Check if logs exist:
```bash
ls -lah ~/dakota-rl-training/outputs/logs/
```

### 2. View recent trainer activity (LAST 50 LINES):
```bash
tail -50 ~/dakota-rl-training/outputs/logs/trainer.stdout
```

### 3. Watch logs in real-time:
```bash
tail -f ~/dakota-rl-training/outputs/logs/trainer.stdout
```
(Press Ctrl+C to stop watching)

### 4. Check when logs were last updated:
```bash
ls -lh ~/dakota-rl-training/outputs/logs/trainer.stdout
```
Look at the timestamp - if it's recent (last few minutes), training is likely still running!

### 5. Search for recent "step" updates:
```bash
grep -i "step" ~/dakota-rl-training/outputs/logs/trainer.stdout | tail -20
```

### 6. Check for errors:
```bash
grep -i "error\|exception\|oom\|failed" ~/dakota-rl-training/outputs/logs/trainer.stdout | tail -20
```

### 7. Check main RL coordinator log:
```bash
tail -30 ~/dakota-rl-training/outputs/logs/rl.log
```

## Alternative: Check WandB Dashboard

**Real-time activity check:**
1. Go to: https://wandb.ai/christian-cooper-us/dakota-rl-grammar
2. Click on your run (e.g., "clean-brook-2")
3. Check the **"System"** tab:
   - Look at "GPU Utilization" - should show activity
   - Check "Memory" - should show memory usage
4. Check the **"Charts"** tab:
   - Look for "step" metric - should be increasing
   - Check timestamp of last update

**If WandB shows:**
-  Recent updates (last few minutes) → Training is running!
-  Last update was 30+ minutes ago → Training likely crashed

## What to Look For

**Signs training is RUNNING:**
- Log file timestamps are recent
- Logs show "step X" messages increasing
- WandB dashboard shows recent updates
- GPU utilization is high (100% on some GPUs)

**Signs training CRASHED:**
- Log file hasn't updated in 30+ minutes
- Last log entry is an error/OOM message
- No "step" messages after the error
- Processes are gone (`ps aux | grep python` shows nothing)

## One-Liner Status Check

```bash
# Quick status check
echo "=== Last log update ===" && \
ls -lh ~/dakota-rl-training/outputs/logs/trainer.stdout && \
echo -e "\n=== Last 10 log lines ===" && \
tail -10 ~/dakota-rl-training/outputs/logs/trainer.stdout
```

This will show you:
1. When the log file was last modified
2. The last 10 lines of output

If the timestamp is recent and logs show training progress, you're good!

