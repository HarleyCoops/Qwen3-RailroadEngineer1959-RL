# Setup Prime CLI for Windows PowerShell
# This script ensures Prime Intellect CLI is installed and accessible

$ErrorActionPreference = "Stop"

Write-Host "Setting up Prime Intellect CLI..." -ForegroundColor Green

# Check if uv is installed
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Host "ERROR: 'uv' is not found in PATH" -ForegroundColor Red
    Write-Host "Install uv from: https://docs.astral.sh/uv/" -ForegroundColor Yellow
    Write-Host 'Or run: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"' -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] uv found: $($uvPath.Source)" -ForegroundColor Green

# Check if .local/bin is in PATH
$localBinPath = Join-Path $env:USERPROFILE ".local\bin"
$pathEntries = $env:PATH -split ';'
$inPath = $pathEntries | Where-Object { $_ -eq $localBinPath -or $_ -like "*\.local\bin*" }

if (-not $inPath) {
    Write-Host "WARNING: $localBinPath is not in PATH" -ForegroundColor Yellow
    Write-Host "Adding to PATH for this session..." -ForegroundColor Yellow
    
    # Add to current session PATH
    $env:PATH = "$localBinPath;$env:PATH"
    
    # Add to user PATH permanently
    Write-Host "Adding to user PATH permanently..." -ForegroundColor Yellow
    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentUserPath -notlike "*$localBinPath*") {
        $newPath = "$currentUserPath;$localBinPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "[OK] Added to user PATH permanently" -ForegroundColor Green
        Write-Host "NOTE: You may need to restart PowerShell for changes to take effect" -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] .local\bin is in PATH" -ForegroundColor Green
}

# Install prime CLI
Write-Host "Installing Prime CLI..." -ForegroundColor Cyan
try {
    uv tool install prime
    Write-Host "[OK] Prime CLI installed successfully" -ForegroundColor Green
} catch {
    Write-Host "Note: Prime CLI may already be installed" -ForegroundColor Yellow
}

# Verify prime is accessible
$primePath = Join-Path $localBinPath "prime.exe"
if (Test-Path $primePath) {
    Write-Host "[OK] Prime CLI found at: $primePath" -ForegroundColor Green
    
    # Test prime command
    try {
        $version = & prime --version 2>&1
        Write-Host "[OK] Prime CLI working: $version" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Prime CLI found but may not be executable" -ForegroundColor Yellow
        Write-Host "Try running: $primePath --version" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING: Prime CLI executable not found at expected location" -ForegroundColor Yellow
    Write-Host "Expected location: $primePath" -ForegroundColor Yellow
}

# Check for API key
if (-not $env:PI_API_KEY -and -not $env:PRIME_API_KEY) {
    Write-Host ""
    Write-Host "[!] API Key not set" -ForegroundColor Yellow
    Write-Host "Set your Prime Intellect API key:" -ForegroundColor Yellow
    Write-Host '  $env:PI_API_KEY = "your_api_key_here"' -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Get your API key from: https://app.primeintellect.ai" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Setup complete! Use 'prime' command to interact with Prime Intellect." -ForegroundColor Green
Write-Host "Example: prime --version" -ForegroundColor Cyan
Write-Host "         prime login" -ForegroundColor Cyan
Write-Host "         prime env info owner/env-name" -ForegroundColor Cyan
