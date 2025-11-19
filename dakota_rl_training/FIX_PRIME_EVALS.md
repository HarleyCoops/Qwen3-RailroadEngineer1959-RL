# Fix: Install prime_evals

## Problem
Orchestrator fails with `ModuleNotFoundError: No module named 'prime_evals'`

## Solution: Install prime_evals

**On Server:**

```bash
cd /workspace/prime-rl

# Install prime_evals
uv pip install prime_evals

# Or if it's from a specific index:
# uv pip install prime_evals --index-url https://hub.primeintellect.ai/primeintellect/simple/

# Verify orchestrator can import it
uv run python -c "from prime_evals import AsyncEvalsClient; print('✓ prime_evals OK')"

# Now try orchestrator again
uv run orchestrator @ ~/dakota_rl_training/configs/orch_test_400.toml
```

## Why This Works

- `prime_evals` is a dependency of prime-rl but wasn't installed
- Installing it resolves the import error
- Orchestrator should now start successfully

