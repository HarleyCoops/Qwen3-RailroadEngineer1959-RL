# Extract environment variables from .env and generate export commands for remote instance
# Run this locally to generate commands for SSH session

$envFile = ".env"
$requiredVars = @("WANDB_API_KEY")
$optionalVars = @("WANDB_PROJECT", "WANDB_ENTITY", "HF_TOKEN", "HUGGINGFACE_TOKEN", "PI_API_KEY", "PRIME_API_KEY")

if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Please create a .env file with your API keys" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Extracting environment variables from .env ===" -ForegroundColor Green

# Read .env file and parse variables
$envVars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    # Skip comments and empty lines
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim().Trim('"').Trim("'")
            $envVars[$key] = $value
        }
    }
}

Write-Host "`n=== Generated export commands for remote instance ===" -ForegroundColor Cyan
Write-Host "`nCopy and paste these into your SSH session:`n" -ForegroundColor Yellow

# Check required variables
$missing = @()
foreach ($var in $requiredVars) {
    if (-not $envVars.ContainsKey($var)) {
        $missing += $var
    }
}

if ($missing.Count -gt 0) {
    Write-Host "WARNING: Missing required variables:" -ForegroundColor Red
    foreach ($var in $missing) {
        Write-Host "  - $var" -ForegroundColor Red
    }
    Write-Host ""
}

# Generate export commands
$exports = @()

# WandB API Key (REQUIRED)
if ($envVars.ContainsKey("WANDB_API_KEY")) {
    $exports += "export WANDB_API_KEY=`"$($envVars['WANDB_API_KEY'])`""
    Write-Host "[OK] WANDB_API_KEY found" -ForegroundColor Green
} else {
    Write-Host "[ERROR] WANDB_API_KEY missing (REQUIRED!)" -ForegroundColor Red
    $exports += "# export WANDB_API_KEY=`"your_wandb_api_key_here`""
}

# WandB Project
if ($envVars.ContainsKey("WANDB_PROJECT")) {
    $exports += "export WANDB_PROJECT=`"$($envVars['WANDB_PROJECT'])`""
    Write-Host "[OK] WANDB_PROJECT found" -ForegroundColor Green
} else {
    $exports += "export WANDB_PROJECT=`"dakota-rl-grammar`"  # Default"
    Write-Host "[WARNING] WANDB_PROJECT not found, using default: dakota-rl-grammar" -ForegroundColor Yellow
}

# WandB Entity
if ($envVars.ContainsKey("WANDB_ENTITY")) {
    $exports += "export WANDB_ENTITY=`"$($envVars['WANDB_ENTITY'])`""
    Write-Host "[OK] WANDB_ENTITY found" -ForegroundColor Green
} else {
    Write-Host "[INFO] WANDB_ENTITY not set (optional)" -ForegroundColor Gray
}

# Hugging Face Token
if ($envVars.ContainsKey("HF_TOKEN")) {
    $exports += "export HF_TOKEN=`"$($envVars['HF_TOKEN'])`""
    Write-Host "[OK] HF_TOKEN found" -ForegroundColor Green
} elseif ($envVars.ContainsKey("HUGGINGFACE_TOKEN")) {
    $exports += "export HF_TOKEN=`"$($envVars['HUGGINGFACE_TOKEN'])`""
    Write-Host "[OK] HF_TOKEN found (from HUGGINGFACE_TOKEN)" -ForegroundColor Green
} else {
    Write-Host "[WARNING] HF_TOKEN not found (optional but recommended)" -ForegroundColor Yellow
    $exports += "# export HF_TOKEN=`"your_hf_token_here`"  # Optional but recommended"
}

# Prime Intellect API Key
if ($envVars.ContainsKey("PI_API_KEY")) {
    $exports += "export PI_API_KEY=`"$($envVars['PI_API_KEY'])`""
    Write-Host "[OK] PI_API_KEY found" -ForegroundColor Green
} elseif ($envVars.ContainsKey("PRIME_API_KEY")) {
    $exports += "export PI_API_KEY=`"$($envVars['PRIME_API_KEY'])`""
    Write-Host "[OK] PI_API_KEY found (from PRIME_API_KEY)" -ForegroundColor Green
} else {
    Write-Host "[INFO] PI_API_KEY not set (optional)" -ForegroundColor Gray
}

# Output the export commands
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "COPY THESE COMMANDS TO SSH SESSION:" -ForegroundColor Yellow
Write-Host "================================================`n" -ForegroundColor Cyan

foreach ($export in $exports) {
    Write-Host $export -ForegroundColor White
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "`nAfter running these exports, launch training:" -ForegroundColor Yellow
$trainCmd = @"
cd ~/prime-rl
uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
"@
Write-Host $trainCmd -ForegroundColor White
Write-Host ""

# Also save to file for easy copy-paste
$outputFile = "remote_env_exports.sh"
$exports | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "[OK] Saved export commands to: $outputFile" -ForegroundColor Green
Write-Host "  You can copy this file to the remote instance and source it:`n" -ForegroundColor Gray
Write-Host "  scp $outputFile ubuntu@your-instance-ip:~/" -ForegroundColor Cyan
Write-Host "  # Then on SSH: source ~/$outputFile`n" -ForegroundColor Cyan

