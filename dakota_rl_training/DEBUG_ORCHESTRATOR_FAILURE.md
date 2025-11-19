# Debug Orchestrator Failure

## Problem
Orchestrator failed before logs were written. Need to check what exists and find the error.

## Solution: Check Output Directory

**On Server:**

```bash
# Check what files exist in the output directory
ls -la ~/dakota_rl_training/outputs/ledger_test_400/

# Check if there's a logs directory
ls -la ~/dakota_rl_training/outputs/ledger_test_400/logs/ 2>/dev/null || echo "No logs directory"

# Check for any error files or stderr output
find ~/dakota_rl_training/outputs/ledger_test_400/ -name "*.log" -o -name "*.err" -o -name "stderr*"

# Try running orchestrator directly to see the error
cd /workspace/prime-rl
uv run orchestrator @ ~/dakota_rl_training/configs/orch_test_400.toml
```

This will show the actual error when running the orchestrator directly.

