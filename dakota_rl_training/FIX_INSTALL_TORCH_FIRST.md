# Fix: Install Torch First, Then Package

## Problem
`uv pip install -e` fails because torch isn't available during build dependency resolution.

## Solution: Install Torch and Dependencies First

**On Server:**

```bash
cd /workspace/prime-rl

# Step 1: Install torch first (make it available to resolver)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install dependencies first (they might pull in torch)
uv pip install verifiers>=0.1.7.post0 datasets>=2.18

# Step 3: Now install the package (torch is available, resolution succeeds)
uv pip install -e ../Dakota1890/environments/dakota_grammar_translation

# Step 4: Verify it works
uv run python -c "import dakota_grammar_translation; print('✓ Environment OK')"
```

## Why This Works

- Installing torch first makes it available to uv's resolver
- Installing verifiers/datasets first ensures their dependencies are satisfied
- Then installing the package can resolve successfully

