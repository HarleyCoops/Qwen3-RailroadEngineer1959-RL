# Monitoring Dakota GRPO Training Runs

This guide explains how to monitor your active GRPO training runs for the Dakota language model.

## Quick Monitoring

Run the monitoring script:

```bash
python scripts/monitor_training.py
```

This will show:
- **Active W&B runs** - Current training runs with runtime and status
- **Latest metrics** - Reward, loss, solve rates (when available)
- **Failed runs** - Recent failures for debugging
- **System resources** - CPU and memory usage
- **Python processes** - Active training processes

## Understanding the Output

### Run Status

- **State: running** - Training is actively running
- **Runtime** - How long the run has been active
- **URL** - Direct link to W&B dashboard for detailed metrics

### Key Metrics to Watch

#### For Orchestrator Runs:
- `reward/mean` - Average reward across rollouts (higher is better)
- `reward/std` - Reward variance (lower = more stable)
- `batch/solve_all` - Percentage of problems solved completely
- `batch/solve_none` - Percentage of problems not solved

#### For Trainer Runs:
- `loss/mean` - Training loss (should decrease over time)
- `optim/lr` - Learning rate
- `optim/grad_norm` - Gradient norm (for stability)

### System Resources

- **CPU Usage** - Should be high during active training
- **Memory Usage** - Watch for memory pressure (>90% is concerning)
- **Python Processes** - Should see trainer + orchestrator pairs

## Monitoring Best Practices

### 1. Regular Checks

Run the monitor script every 10-15 minutes during training:

```bash
# Windows PowerShell
while ($true) { python scripts/monitor_training.py; Start-Sleep 600 }
```

### 2. Watch for Issues

**Red Flags:**
- Memory usage >90%
- CPU usage consistently <5% (may indicate stalled training)
- Failed runs increasing
- Metrics not updating (runs may be stuck)

**Good Signs:**
- Reward mean increasing over time
- Loss decreasing (for trainer)
- Solve rates improving
- Stable memory usage

### 3. W&B Dashboard

For detailed visualization, visit the W&B URLs shown in the monitor output:
- View reward curves
- Check loss trends
- Examine sample outputs
- Compare runs side-by-side

## Current Active Runs

Based on your W&B export, you have:

1. **qwen3-0.6b-rl-20251106-180601** (older pair)
   - Trainer: Running ~58 minutes
   - Orchestrator: Running ~58 minutes
   - URL: https://wandb.ai/christian-cooper-us/grammar-gym/runs/bqbyvpm2

2. **qwen3-0.6b-rl-20251106-175347** (newer pair)
   - Trainer: Running ~1 hour 9 minutes
   - Orchestrator: Running ~1 hour 11 minutes
   - URL: https://wandb.ai/christian-cooper-us/grammar-gym/runs/f6cb75js

## Troubleshooting

### No Metrics Showing

If metrics show "No metrics available yet":
- Runs may still be initializing (first few minutes)
- Check W&B dashboard directly - metrics may be logged but not fetched yet
- Verify training is actually progressing (check logs)

### High Memory Usage

If memory is >90%:
- Consider reducing batch size
- Check for memory leaks in logs
- May need to restart training with lower memory settings

### Failed Runs

Check the failed run URL for error details:
- Connection issues (vLLM server down?)
- Configuration errors
- Resource exhaustion

## Advanced Monitoring

### Check Logs Directly

Training logs are typically in:
- `outputs/` directory
- W&B run files in `wandb/` directory
- Console output where training was started

### Real-time Metrics

For real-time monitoring, use W&B's live dashboard:
1. Open the run URL from monitor output
2. Watch metrics update in real-time
3. Set up alerts for failures or anomalies

### Custom Monitoring

You can extend `scripts/monitor_training.py` to:
- Add custom metrics
- Send alerts (email, Slack, etc.)
- Generate reports
- Track specific reward components

## Next Steps

1. **Monitor regularly** - Run the script every 10-15 minutes
2. **Check W&B dashboard** - Use URLs for detailed visualization
3. **Watch for convergence** - Look for reward plateauing (may indicate training complete)
4. **Review failed runs** - Learn from failures to improve configs

For questions or issues, check the training logs or W&B dashboard for detailed error messages.

