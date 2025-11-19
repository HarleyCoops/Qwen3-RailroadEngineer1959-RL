# Check Orchestrator Log

## Problem
Orchestrator failed with exit code 1. Need to check the log to see the actual error.

## Solution: Check Orchestrator Log

**On Server:**

```bash
# Check the orchestrator log
cat ~/dakota_rl_training/outputs/ledger_test_400/logs/orchestrator.log

# Or check the last 50 lines
tail -50 ~/dakota_rl_training/outputs/ledger_test_400/logs/orchestrator.log
```

This will show the actual error that caused the orchestrator to fail.

