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
prime env eval `
  --env <owner>/<env-name> `
  --model <model-name> `
  --dataset <path-to-dataset> `
  --num-examples <count>
```

**Note**: According to Prime Intellect docs, the command is `prime env eval` (not `prime eval run`)

#### Example: Dakota Grammar Eval

```powershell
# Using Claude Sonnet
prime env eval `
  --env HarleyCoops/dakota-grammar-env `
  --model claude-3-5-sonnet `
  --dataset dakota_rl_training/datasets/grammar_tasks_easy.jsonl `
  --num-examples 100

# Using GPT-4
prime env eval `
  --env HarleyCoops/dakota-grammar-env `
  --model gpt-4 `
  --dataset dakota_rl_training/datasets/grammar_tasks_medium.jsonl `
  --num-examples 100

# Using open-source model (e.g., Qwen)
prime env eval `
  --env HarleyCoops/dakota-grammar-env `
  --model Qwen/Qwen2.5-7B-Instruct `
  --dataset dakota_rl_training/datasets/grammar_tasks_hard.jsonl `
  --num-examples 50
```

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

### API Models (No Setup Required)
- `claude-3-5-sonnet` - Claude 3.5 Sonnet
- `claude-3-opus` - Claude 3 Opus
- `gpt-4` - GPT-4
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### Open-Source Models (Requires Model Spec)
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- Custom models via HuggingFace or model path

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
prime env info HarleyCoops/dakota-grammar-env
```

If not published, see: `docs/guides/PRIMEINTELLECT_PUBLISHING_GUIDE.md`

### Step 2: Prepare Dataset

Your datasets are ready:
- `dakota_rl_training/datasets/grammar_tasks_easy.jsonl`
- `dakota_rl_training/datasets/grammar_tasks_medium.jsonl`
- `dakota_rl_training/datasets/grammar_tasks_hard.jsonl`

### Step 3: Run Hosted Eval

```powershell
# Set API key
$env:PI_API_KEY="your_key"

# Verify prime is accessible
prime --version

# Run eval on easy tasks with Claude
prime env eval `
  --env HarleyCoops/dakota-grammar-env `
  --model claude-3-5-sonnet `
  --dataset dakota_rl_training/datasets/grammar_tasks_easy.jsonl `
  --num-examples 100 `
  --output-dir eval_results
```

### Step 4: Monitor Progress

```powershell
# Check eval status (command may vary - check Prime docs)
prime env eval status <eval-id>

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
1. Verify environment is published: `prime env info HarleyCoops/dakota-grammar-env`
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

Verify your JSONL format:
```powershell
# Check first few lines
Get-Content dakota_rl_training/datasets/grammar_tasks_easy.jsonl -Head 3
```

Each line should be valid JSON with `id`, `prompt`, and optional `expected` fields.

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

