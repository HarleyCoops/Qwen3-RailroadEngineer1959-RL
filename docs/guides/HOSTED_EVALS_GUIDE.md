# Hosted Evals Guide - Prime Intellect Environment Hub

## What Are Hosted Evals?

**Hosted evals** are a new feature on Prime Intellect's Environment Hub that allows you to:
- Run evaluations directly in the cloud (no local setup needed)
- Use your favorite models (Claude, GPT-4, open-source models)
- Share benchmarks with the community
- Eliminate GPU/infrastructure requirements

**Announced**: Recently (within last 48 hours)  
**Status**: Live on Environment Hub

---

## Prerequisites

### 1. Prime Intellect Account & API Key

Get your API key from: https://app.primeintellect.ai

```bash
# Set in PowerShell
$env:PI_API_KEY="your_api_key_here"
```

### 2. Install Prime CLI

**IMPORTANT**: The Prime CLI is installed via `uv`, not pip!

```powershell
# Make sure uv is installed (usually already installed)
# If not: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install prime CLI
uv tool install prime

# Verify .local\bin is in your PATH
# Run this script to fix PATH issues:
powershell -ExecutionPolicy Bypass -File scripts/setup_prime_cli.ps1
```

**Alternative**: If you have persistent PATH issues, use the setup script:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_prime_cli.ps1
```

### 3. Login

```bash
prime login
```

---

## Running Your First Hosted Eval

### Option 1: Prime CLI (Recommended)

**First, ensure Prime CLI is installed and accessible:**

```powershell
# Check if prime is accessible
prime --version

# If not found, install with:
uv tool install prime

# If still not found, ensure .local\bin is in PATH
# Run the setup script:
powershell -ExecutionPolicy Bypass -File scripts/setup_prime_cli.ps1
```

#### Basic Command Structure

```powershell
# PowerShell syntax (use backticks for line continuation)
prime env eval <environment-name> `
  -m <model-name> `
  -n <num-examples> `
  -r <rollouts-per-example>
```

**Key Points**:
- Environment name is a **positional argument** (not `--env`)
- Use `-m` or `--model` for model selection
- Use `-n` or `--num-examples` for number of examples
- Use `-r` or `--rollouts-per-example` for rollouts per example
- The environment loads its own dataset (no `--dataset` flag needed)

#### Example: Dakota Grammar Eval (Minimal Parameters)

**Start with smallest model and minimal parameters for testing:**

```powershell
# First, install the environment if not already installed
# Note: Use owner/name format for installation
prime env install harleycooper/dakota1890

# Get the absolute path to your dataset
$datasetPath = (Resolve-Path dakota_rl_training/datasets/grammar_tasks_complete.jsonl).Path -replace '\\', '/'

# Create environment args JSON string
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 10}'

# Minimal test run with smallest model
prime env eval dakota1890 `
  -m openai/gpt-5-nano `
  -n 5 `
  -r 1 `
  -t 256 `
  -T 0.7 `
  --env-args $envArgs

# Slightly larger test with 7B model
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 20}'
prime env eval dakota1890 `
  -m mistralai/mistral-7b-instruct-v0.3 `
  -n 10 `
  -r 2 `
  -t 512 `
  -T 0.6 `
  --env-args $envArgs

# Production run with more examples
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 100}'
prime env eval dakota1890 `
  -m meta-llama/llama-3.1-8b-instruct `
  -n 100 `
  -r 3 `
  -t 1024 `
  -T 0.7 `
  --env-args $envArgs
```

**Parameter Guidelines**:
- **Small test**: `-n 5`, `-r 1`, `-t 256` (fastest, cheapest)
- **Medium test**: `-n 10`, `-r 2`, `-t 512` (balanced)
- **Full eval**: `-n 100`, `-r 3`, `-t 1024` (comprehensive)

**Important**: You must provide the `dataset_path` via `--env-args` because the environment needs to know where to find your dataset file.

**Note**: The environment name is just `dakota1890` (not `harleycooper/dakota1890`) after installation.

### Option 2: Via Environment Hub Web UI

1. **Navigate to Environment Hub**
   - Go to: https://app.primeintellect.ai/dashboard/environments

2. **Find Your Environment**
   - Search for: `dakota-grammar-env` or `HarleyCoops/dakota-grammar-env`
   - Click on the environment card

3. **Launch Hosted Eval**
   - Click "Run Evaluation" or "Hosted Eval" button
   - Select:
     - **Model**: Choose from available models (Claude, GPT-4, custom, etc.)
     - **Dataset**: Upload or select from your datasets
     - **Configuration**: Number of examples, timeout, etc.

4. **Monitor Results**
   - View real-time progress in dashboard
   - See results and metrics
   - Share benchmarks with community

---

## Available Models

### Recommended: Small Open-Source Models for Testing

For initial testing, use the smallest available open-source models:

**Smallest/Cheapest Options**:
- `openai/gpt-5-nano` - Smallest, cheapest ($0.05/$0.4 per 1M tokens)
- `openai/gpt-oss-20b` - Open-source 20B model ($0.07/$0.3 per 1M tokens)
- `mistralai/mistral-7b-instruct-v0.3` - Mistral 7B ($0.1/$0.25 per 1M tokens)
- `meta-llama/llama-3.1-8b-instruct` - Llama 3.1 8B ($0.9/$0.9 per 1M tokens)

**Check Available Models**:
```powershell
# List all available models on Prime Inference
prime inference models
```

### API Models (Require API Keys)

These require separate API keys and may not be available:
- `anthropic/claude-*` - Requires ANTHROPIC_API_KEY
- `openai/gpt-*` - Requires OPENAI_API_KEY
- `google/gemini-*` - Requires GOOGLE_API_KEY

---

## Dataset Format

Your Dakota grammar datasets are already in the correct format:

```jsonl
{"id": "task_001", "prompt": "Translate to Dakota: ...", "expected": "..."}
{"id": "task_002", "prompt": "Apply grammar rule: ...", "expected": "..."}
```

Located in:
- `dakota_rl_training/datasets/grammar_tasks_easy.jsonl` (1,998 tasks)
- `dakota_rl_training/datasets/grammar_tasks_medium.jsonl` (2,155 tasks)
- `dakota_rl_training/datasets/grammar_tasks_hard.jsonl` (398 tasks)

---

## Evaluation Process

### What Happens During Hosted Eval

1. **Environment Setup**: Prime Intellect provisions compute resources
2. **Model Loading**: Selected model is loaded/configured
3. **Task Execution**: For each task in dataset:
   - Environment presents prompt to model
   - Model generates response
   - Environment evaluates response using rubric
   - Reward/score calculated
4. **Results Aggregation**: Metrics calculated across all tasks
5. **Report Generation**: Results available in dashboard

### Metrics Returned

For Dakota grammar environment, metrics include:
- **Accuracy**: % of tasks completed correctly
- **Character Preservation**: % of Dakota special characters preserved
- **Affix Accuracy**: % of affixes applied correctly
- **Semantic Accuracy**: % of semantically correct translations
- **Composite Reward**: Overall score (0.0 to 1.0)

---

## Sharing Benchmarks

After running an eval:

1. **View Results**: Check dashboard for detailed metrics
2. **Share**: Click "Share Benchmark" to make results public
3. **Compare**: Compare your model's performance with others
4. **Export**: Download results as JSON/CSV

---

## Example: Complete Workflow

### Step 1: Verify Environment is Published

```bash
# Check if environment exists
prime env info harleycooper/dakota1890
```

If not published, see: `docs/guides/PRIMEINTELLECT_PUBLISHING_GUIDE.md`

### Step 2: Install Environment

```powershell
# Install the Dakota grammar environment
prime env install harleycooper/dakota1890

# Verify installation
python -c "import dakota1890"
```

The environment includes its own dataset, so no separate dataset file is needed.

### Step 3: Run Hosted Eval (Minimal Parameters First)

**Start with smallest model and minimal parameters**:

```powershell
# Set API key (only needed for Prime Inference API access)
$env:PI_API_KEY="your_key"

# Verify prime is accessible
prime --version

# Install the environment first if not already installed
prime env install harleycooper/dakota1890

# Get the absolute path to your dataset
$datasetPath = (Resolve-Path dakota_rl_training/datasets/grammar_tasks_complete.jsonl).Path -replace '\\', '/'

# Create environment args JSON string
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 10}'

# MINIMAL TEST RUN - Smallest model, minimal parameters
prime env eval dakota1890 `
  -m openai/gpt-5-nano `
  -n 5 `
  -r 1 `
  -t 256 `
  -T 0.7 `
  --env-args $envArgs

# If successful, try slightly larger:
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 20}'
prime env eval dakota1890 `
  -m mistralai/mistral-7b-instruct-v0.3 `
  -n 10 `
  -r 2 `
  -t 512 `
  -T 0.6 `
  --env-args $envArgs
```

**Parameter Guide**:
- `-m, --model`: Model to use (start with `openai/gpt-5-nano` for smallest)
- `-n, --num-examples`: Number of examples (start with `5` for testing)
- `-r, --rollouts-per-example`: Rollouts per example (start with `1` for testing)
- `-c, --max-concurrent`: Max concurrent requests (default: 32)
- `-t, --max-tokens`: Max tokens to generate (start with `256` for testing)
- `-T, --temperature`: Temperature for sampling (0.6-0.7 recommended)
- `-a, --env-args`: Environment arguments as JSON (REQUIRED: must include `dataset_path`)
- `-s, --save-results`: Save results to disk (default: True)
- `-v, --verbose`: Verbose output
- `-P, --push-to-hub`: Push results to Prime Evals Hub

### Step 4: Monitor Progress

```powershell
# View eval results (saved to disk by default)
# Results are saved in the current directory

# List recent evaluations
prime eval list

# Get specific evaluation details
prime eval get <eval-id>

# View samples from an evaluation
prime eval samples <eval-id>

# Or view in dashboard
Start-Process "https://app.primeintellect.ai/dashboard/evals"
```

### Step 5: Review Results

Results available at:
- Dashboard: https://app.primeintellect.ai/dashboard/evals
- Local: `eval_results/` directory

---

## Troubleshooting

### Prime CLI Not Found / PATH Issues

**Symptoms**: `prime: command not found` or `prime-intellect-cli` package not found

**Solution**:
```powershell
# 1. Prime CLI is installed via uv, NOT pip
# Wrong: pip install prime-intellect-cli
# Right: uv tool install prime

# 2. Run the setup script to fix PATH issues:
powershell -ExecutionPolicy Bypass -File scripts/setup_prime_cli.ps1

# 3. Verify installation:
prime --version
# Should show: Prime CLI version: 0.4.9 (or similar)

# 4. If still not found, manually add to PATH:
$localBinPath = "$env:USERPROFILE\.local\bin"
$env:PATH = "$localBinPath;$env:PATH"
```

**Persistent PATH Fix** (adds to user PATH permanently):
```powershell
# Add .local\bin to user PATH permanently
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
$localBinPath = "$env:USERPROFILE\.local\bin"
if ($currentPath -notlike "*$localBinPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$localBinPath", "User")
    Write-Host "Added to PATH. Restart PowerShell for changes to take effect." -ForegroundColor Green
}
```

### Environment Not Found

If you get "environment not found":
1. Verify environment is published: `prime env info harleycooper/dakota1890`
2. If not published, follow: `docs/guides/PRIMEINTELLECT_PUBLISHING_GUIDE.md`

### API Key Issues

```powershell
# Set API key
$env:PI_API_KEY="your_key_here"

# Verify it's set
Write-Host $env:PI_API_KEY

# Or use PRIME_API_KEY (alternative name)
$env:PRIME_API_KEY="your_key_here"
```

### Dataset Format Issues

**CRITICAL**: You must provide the dataset path via `--env-args` flag:

```powershell
# Get absolute path to dataset
$datasetPath = (Resolve-Path dakota_rl_training/datasets/grammar_tasks_complete.jsonl).Path -replace '\\', '/'

# Create environment args JSON
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 10}'

# Use in eval command
prime env eval dakota1890 -m openai/gpt-5-nano -n 5 -r 1 --env-args $envArgs
```

**Available Environment Arguments**:
- `dataset_path`: Path to JSONL file (REQUIRED)
- `max_examples`: Limit number of examples (-1 for all)
- `eval_examples`: Limit eval examples
- `difficulty_filter`: Filter by difficulty `["easy"]`, `["medium"]`, `["hard"]`
- `task_filter`: Filter by task type `["morphology"]`, `["translation"]`
- `eval_fraction`: Fraction for eval split (default: 0.1)
- `include_hints`: Include hint metadata (default: true)

**Example with filters**:
```powershell
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"difficulty_filter\": [\"easy\"], \"max_examples\": 50}'
prime env eval dakota1890 -m openai/gpt-5-nano -n 5 -r 1 --env-args $envArgs
```

---

## Resources

- **Environment Hub**: https://app.primeintellect.ai/dashboard/environments
- **Prime Intellect Docs**: https://docs.primeintellect.ai
- **Environments Tutorial**: https://docs.primeintellect.ai/tutorials-environments/environments
- **HUD Platform**: https://hud.so (alternative eval platform)

---

## Next Steps

1. **Verify Environment**: Ensure `dakota-grammar-env` is published
2. **Set API Key**: Configure `PI_API_KEY`
3. **Run First Eval**: Start with easy dataset + Claude
4. **Compare Models**: Run same eval with different models
5. **Share Results**: Make benchmarks public

---

**Status**: Ready to run hosted evals!  
**Last Updated**: Based on latest announcements (within 48 hours)

