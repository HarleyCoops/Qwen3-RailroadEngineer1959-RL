# Environment Variables Setup for Remote Instance

## Required Variables

Run these commands in your SSH session to set up environment variables for GRPO training:

```bash
# WandB API Key (REQUIRED)
export WANDB_API_KEY="your_wandb_api_key_here"

# WandB Project (optional, defaults to dakota-rl-grammar)
export WANDB_PROJECT="dakota-rl-grammar"

# WandB Entity (optional - your W&B username)
export WANDB_ENTITY="your_wandb_username"

# Hugging Face Token (optional but recommended for model access)
export HF_TOKEN="your_hf_token_here"

# Prime Intellect API Key (optional, for CLI)
export PI_API_KEY="your_pi_api_key_here"
```

## Quick Setup Script

If you have these variables in your local `.env` file, you can extract them:

1. **From PowerShell (local machine):**
```powershell
# Extract WANDB_API_KEY from .env
$wandbKey = (Get-Content .env | Select-String "^WANDB_API_KEY=").ToString().Split('=')[1].Trim('"').Trim("'")
Write-Host "export WANDB_API_KEY=`"$wandbKey`""

# Extract HF_TOKEN (or HUGGINGFACE_TOKEN)
$hfToken = (Get-Content .env | Select-String "^HF_TOKEN=|^HUGGINGFACE_TOKEN=").ToString().Split('=')[1].Trim('"').Trim("'")
if ($hfToken) { Write-Host "export HF_TOKEN=`"$hfToken`"" }

# Extract WANDB_PROJECT
$wandbProj = (Get-Content .env | Select-String "^WANDB_PROJECT=").ToString().Split('=')[1].Trim('"').Trim("'")
if ($wandbProj) { Write-Host "export WANDB_PROJECT=`"$wandbProj`"" } else { Write-Host "export WANDB_PROJECT=`"dakota-rl-grammar`"" }
```

2. **Copy the output and paste into SSH session**

## Make Variables Permanent

To make these variables persist across sessions, add them to `~/.bashrc`:

```bash
echo 'export WANDB_API_KEY="your_key_here"' >> ~/.bashrc
echo 'export WANDB_PROJECT="dakota-rl-grammar"' >> ~/.bashrc
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## Verify Setup

After setting variables, verify they're set:

```bash
echo $WANDB_API_KEY
echo $WANDB_PROJECT
echo $HF_TOKEN
```

## Then Launch Training

Once variables are set, launch training:

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
```

