# Quick Guide: Upload Model to Hugging Face

## The Problem
Your model files are on the **remote Prime Intellect instance**, not on your local machine. You need to download them first, then upload to Hugging Face.

## Complete Workflow

### Step 1: Find Your Model Files on the Instance

SSH into your instance and find where the weights are:

```powershell
# Connect to instance
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43

# Once connected, find weights
ls -lh ~/dakota_rl_training/outputs/*/weights/step_*/
```

Look for something like:
- `~/dakota_rl_training/outputs/ledger_test_400/weights/step_400/`
- `~/dakota_rl_training/outputs/grpo_30b/weights/step_400/`

### Step 2: Download Model Files

**Exit SSH** (type `exit`), then from PowerShell:

```powershell
python scripts/conversion/download_model_from_instance.py `
    --instance-ip 65.109.75.43 `
    --remote-path "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400" `
    --local-path "downloaded_model" `
    --ssh-key "$env:USERPROFILE\.ssh\DakotaRL3" `
    --ssh-port 1234
```

Replace `ledger_test_400` and `step_400` with your actual run name and step number.

### Step 3: Verify Downloaded Files

```powershell
python scripts/conversion/prepare_model_for_hf.py --model-dir "downloaded_model"
```

Should show:
```
✓ Config: ✓
✓ Tokenizer: ✓
✓ Weights: ✓
```

### Step 4: Upload to Hugging Face

```powershell
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "downloaded_model" `
    --repo-id "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"
```

Done! Your model is now on Hugging Face.

## One-Liner (If You Know the Path)

```powershell
# Download
python scripts/conversion/download_model_from_instance.py --instance-ip 65.109.75.43 --remote-path "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400" --ssh-key "$env:USERPROFILE\.ssh\DakotaRL3" --ssh-port 1234

# Upload
python scripts/conversion/upload_model_to_hf.py --model-dir "downloaded_model"
```

## Troubleshooting

### "Permission denied" when downloading
- Check SSH key path: `Test-Path $env:USERPROFILE\.ssh\DakotaRL3`
- Try without `--ssh-key` if key is in default location

### "No such file or directory" on remote
- SSH into instance and verify the path exists
- Check you're using the correct run name and step number

### "No weights directory found" locally
- You haven't downloaded yet! Run Step 2 first.

## Files Created

- `scripts/conversion/download_model_from_instance.py` - Downloads from remote instance
- `scripts/conversion/upload_model_to_hf.py` - Uploads to Hugging Face
- `scripts/conversion/prepare_model_for_hf.py` - Checks files before upload

