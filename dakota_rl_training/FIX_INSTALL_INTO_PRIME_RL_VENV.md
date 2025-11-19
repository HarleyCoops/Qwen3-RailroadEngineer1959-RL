# Fix: Install Into Prime-RL Venv

## Problem
Package was installed in a different venv. `uv run` uses the prime-rl venv, so we need to install it there.

## Solution: Install From Prime-RL Directory

**On Server:**

```bash
cd /workspace/prime-rl

# Install directly into prime-rl's venv (this is the key!)
uv pip install -e ../Dakota1890/environments/dakota_grammar_translation

# Install dependencies manually
uv pip install verifiers>=0.1.7.post0 datasets>=2.18

# Now verify it works
uv run python -c "import dakota_grammar_translation; print('✓ Environment OK')"
```

## Why This Works

- Installing from `/workspace/prime-rl` ensures it goes into the prime-rl venv
- `uv run` uses the prime-rl venv, so the package will be found
- The `-e` flag makes it editable so changes are picked up

