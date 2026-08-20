# Simple script to extract environment variables from .env
# Run: powershell -ExecutionPolicy Bypass -File scripts/get_env_exports.ps1

$envFile = ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Environment Variable Export Commands ===" -ForegroundColor Green
Write-Host "Copy these to your SSH session:`n" -ForegroundColor Yellow

# Read and parse .env file
$vars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim().Trim('"').Trim("'")
        $vars[$key] = $value
    }
}

# Required: WANDB_API_KEY
if ($vars.ContainsKey("WANDB_API_KEY")) {
    Write-Host "export WANDB_API_KEY=`"$($vars['WANDB_API_KEY'])`"" -ForegroundColor White
} else {
    Write-Host "# export WANDB_API_KEY=`"your_wandb_api_key_here`"  # REQUIRED!" -ForegroundColor Red
}

# Optional: WANDB_PROJECT
if ($vars.ContainsKey("WANDB_PROJECT")) {
    Write-Host "export WANDB_PROJECT=`"$($vars['WANDB_PROJECT'])`"" -ForegroundColor White
} else {
    Write-Host "export WANDB_PROJECT=`"dakota-rl-grammar`"  # Default" -ForegroundColor Gray
}

# Optional: WANDB_ENTITY
if ($vars.ContainsKey("WANDB_ENTITY")) {
    Write-Host "export WANDB_ENTITY=`"$($vars['WANDB_ENTITY'])`"" -ForegroundColor White
}

# Optional: HF_TOKEN or HUGGINGFACE_TOKEN
if ($vars.ContainsKey("HF_TOKEN")) {
    Write-Host "export HF_TOKEN=`"$($vars['HF_TOKEN'])`"" -ForegroundColor White
} elseif ($vars.ContainsKey("HUGGINGFACE_TOKEN")) {
    Write-Host "export HF_TOKEN=`"$($vars['HUGGINGFACE_TOKEN'])`"" -ForegroundColor White
} else {
    Write-Host "# export HF_TOKEN=`"your_hf_token_here`"  # Optional but recommended" -ForegroundColor Gray
}

# Optional: PI_API_KEY or PRIME_API_KEY
if ($vars.ContainsKey("PI_API_KEY")) {
    Write-Host "export PI_API_KEY=`"$($vars['PI_API_KEY'])`"" -ForegroundColor White
} elseif ($vars.ContainsKey("PRIME_API_KEY")) {
    Write-Host "export PI_API_KEY=`"$($vars['PRIME_API_KEY'])`"" -ForegroundColor White
}

Write-Host "`n=== Copy the above commands to your SSH session ===" -ForegroundColor Green

