# Publishing Dakota1890 Environment to PrimeIntellect

## Quick Publish Command

```bash
python publish_dakota_environment.py --owner <your-username>
```

The script defaults to:
- **Environment name**: `Dakota1890`
- **Version**: `0.1.0`
- **Package directory**: `environments/dakota_grammar_translation`

## Prerequisites

1. **Install PrimeIntellect CLI**:
   ```bash
   pip install prime-intellect-cli
   ```

2. **Login to PrimeIntellect**:
   ```bash
   prime login
   ```

3. **Install build tools** (if not already installed):
   ```bash
   pip install build
   ```

## What Gets Published

- **Environment Package**: `dakota-grammar-translation`
- **Environment Name**: `Dakota1890`
- **Components**:
  - `DakotaGrammarEnv` (multi-turn grammar tasks)
  - `DakotaMorphologyEnv` (single-turn morphology tasks)
  - `DakotaGrammarRubric` (reward functions)
  - Rules loaded from `data/rl_training_rules/all_rl_rules.json`
  - 10,576 training tasks ready for RL

## Example Usage

```bash
# Publish with default name (Dakota1890)
python publish_dakota_environment.py --owner HarleyCoops

# Publish with custom version
python publish_dakota_environment.py --owner HarleyCoops --version 0.2.0

# Dry run (test without publishing)
python publish_dakota_environment.py --owner HarleyCoops --dry-run
```

## After Publishing

Your environment will be available at:
```
https://primeintellect.ai/env/<owner>/Dakota1890
```

To use in training configs:
```yaml
environment:
  name: <owner>/Dakota1890
  version: 0.1.0
```

## Verification

After publishing, verify it was uploaded:
```bash
prime env info <owner>/Dakota1890
```

