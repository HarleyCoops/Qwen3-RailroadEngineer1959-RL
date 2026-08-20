# Fixing Prime Intellect Hosted Eval Dataset Issue

## Problem

The error shows:
```
Using default args: dataset_path=None
Unable to locate any Dakota grammar tasks. Checked: /dakota_rl_training/datasets/grammar_tasks_complete.jsonl and fallback sample.
```

The environment isn't receiving `dataset_path` via environment arguments, and the fallback paths don't exist.

## Solutions

### Option 1: Pass dataset_path via UI Environment Arguments (Recommended)

In the Prime Intellect UI, when configuring the eval:

1. Go to **Environment Arguments** section
2. Click **"+ Add Entry"**
3. Add:
   - **Key**: `dataset_path`
   - **Value**: Must be a URL or path accessible from Prime Intellect servers
   
**Options for dataset_path value:**
- **GitHub Raw URL**: `https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/dakota_rl_training/datasets/grammar_tasks_complete.jsonl`
- **Public URL**: If you host the dataset somewhere publicly accessible
- **Relative path**: If packaged with environment (see Option 2)

### Option 2: Package Dataset with Environment

Update `pyproject.toml` to include the dataset:

```toml
[tool.hatch.build.targets.wheel]
packages = ["dakota_grammar_translation", "dakota1890"]
include = [
    "dakota_grammar_translation/data/*.jsonl",
    "dakota_rl_training/datasets/grammar_tasks_complete.jsonl",  # If relative
]
```

Then copy the dataset to the environment package:
```bash
cp dakota_rl_training/datasets/grammar_tasks_complete.jsonl \
   environments/dakota_grammar_translation/dakota_grammar_translation/data/
```

Then republish the environment.

### Option 3: Create Fallback Sample Dataset

Create a small sample dataset in the packaged environment:
```bash
# Copy first 100 examples as sample
head -n 100 dakota_rl_training/datasets/grammar_tasks_complete.jsonl > \
  environments/dakota_grammar_translation/dakota_grammar_translation/data/sample_tasks.jsonl
```

Then republish the environment.

## Quick Fix (UI Only)

**In Prime Intellect UI Environment Arguments:**

Add entry:
- Key: `dataset_path`
- Value: `https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/dakota_rl_training/datasets/grammar_tasks_complete.jsonl`

This assumes your repo is public. If private, use Option 2 or host the dataset publicly.

