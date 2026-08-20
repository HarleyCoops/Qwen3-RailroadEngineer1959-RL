# Complete Guide: Uploading Model to Hugging Face

This guide walks you through uploading your trained Dakota Grammar RL model to Hugging Face Hub.

## Prerequisites

1. **Hugging Face Account**: Make sure you have an account at https://huggingface.co
2. **Authentication**: Login using one of these methods:
   ```powershell
   # Option 1: CLI login (recommended)
   huggingface-cli login
   
   # Option 2: Set token in .env file
   # Add to .env: HF_TOKEN=your_token_here
   # Get token from: https://huggingface.co/settings/tokens
   ```
3. **Model Files**: Your trained model weights should be available

## Step 1: Download Model Files from Remote Instance

**IMPORTANT**: Your model files are on the Prime Intellect training instance, not locally! You need to download them first.

### Quick Download

```powershell
python scripts/conversion/download_model_from_instance.py `
    --instance-ip 65.109.75.43 `
    --remote-path "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400" `
    --ssh-key "$env:USERPROFILE\.ssh\DakotaRL3" `
    --ssh-port 1234
```

### Find Your Model Files on the Instance

First, SSH into your instance to find the exact path:

```powershell
# SSH into instance
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43

# Once connected, find your weights
ls -lh ~/dakota_rl_training/outputs/*/weights/step_*/
```

Common locations:
- `~/dakota_rl_training/outputs/ledger_test_400/weights/step_400/`
- `~/dakota_rl_training/outputs/grpo_30b/weights/step_400/`
- `~/dakota_rl_training/outputs/{run_name}/weights/step_{step_number}/`

### Download Weights Directory

```powershell
python scripts/conversion/download_model_from_instance.py `
    --instance-ip <your-instance-ip> `
    --remote-path "~/dakota_rl_training/outputs/<run_name>/weights/step_400" `
    --local-path "downloaded_model" `
    --ssh-key "$env:USERPROFILE\.ssh\DakotaRL3" `
    --ssh-port 1234
```

This downloads the entire weights directory (config, tokenizer, weights) to `downloaded_model/` locally.

## Step 2: Locate Your Model Files (After Download)

After downloading, model files are saved locally. The exact location depends on what you specified with `--local-path`.

### Option A: Check Latest Weights Directory

Run the preparation script to find your model files:

```powershell
python scripts/conversion/prepare_model_for_hf.py
```

This will:
- Search for the latest weights directory
- Check if all required files are present
- Show you what's ready to upload

### Option B: Manual Location

Model files are typically in one of these locations:

```
dakota_rl_training/outputs/weights/step_400/
dakota_rl_training/outputs/{run_name}/weights/step_400/
```

The directory should contain:
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer file
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merges (if applicable)
- `model.safetensors` or `pytorch_model.bin` - Model weights
- Other tokenizer files as needed

## Step 2: Verify Model Files

Before uploading, verify all files are present:

```powershell
python scripts/conversion/prepare_model_for_hf.py --model-dir "path/to/your/weights/step_400"
```

Expected output:
```
 Config: 
 Tokenizer: 
 Weights: 

Weight files found:
  - model.safetensors (1234.5 MB)

 All required files found!
```

## Step 3: Convert Checkpoint (If Needed)

If your model is in checkpoint format (`.distcp` files), you need to convert it first:

```powershell
# Navigate to prime-rl-framework
cd prime-rl-framework

# Convert checkpoint to HuggingFace format
python scripts/extract_hf_from_ckpt.py `
    --ckpt-dir "path/to/checkpoints/step_400/trainer" `
    --output-dir "hf_model" `
    --utils-repo-id "Qwen/Qwen3-0.6B" `
    --dtype "bfloat16"
```

This will:
- Extract model weights from checkpoint
- Download config/tokenizer from base model repo
- Save everything in HuggingFace-compatible format

Then use the `hf_model` directory for upload.

## Step 4: Upload Model to Hugging Face

### Basic Upload

```powershell
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "dakota_rl_training/outputs/weights/step_400" `
    --repo-id "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"
```

### Upload with Custom Options

```powershell
# Upload to different repo
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "path/to/weights/step_400" `
    --repo-id "YourUsername/YourModelName"

# Upload as private repo
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "path/to/weights/step_400" `
    --private

# Upload without model card
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "path/to/weights/step_400" `
    --no-model-card

# Custom commit message
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "path/to/weights/step_400" `
    --commit-message "Initial model upload - 400 steps training"
```

## Step 5: Upload Model Card (If Not Already Uploaded)

The model card (README.md) is automatically uploaded with the model. To update it separately:

```powershell
python scripts/conversion/upload_model_card.py `
    --repo-id "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL" `
    --model-card-path "MODEL_CARD.md"
```

## What Gets Uploaded

The upload script uploads:

1. **Model Configuration**
   - `config.json` - Model architecture and hyperparameters
   - `generation_config.json` - Generation settings (if present)

2. **Tokenizer Files**
   - `tokenizer_config.json` - Tokenizer configuration
   - `tokenizer.json` - Main tokenizer file
   - `vocab.json` - Vocabulary mappings
   - `merges.txt` - BPE merges (for BPE tokenizers)
   - `special_tokens_map.json` - Special token definitions
   - `chat_template.jinja` - Chat template (if present)
   - `added_tokens.json` - Additional tokens (if present)

3. **Model Weights**
   - `model.safetensors` or `pytorch_model.bin` - Model weights
   - `model.safetensors.index.json` - Weight index (for sharded models)

4. **Documentation**
   - `README.md` - Model card (from MODEL_CARD.md)

## Verification

After upload, verify your model:

1. **Visit your model page**: https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL
2. **Check files**: All files should be listed
3. **Test inference**: Use the "Hosted inference API" tab to test

## Troubleshooting

### "No model weight files found"

**Problem**: The script can't find model weight files.

**Solutions**:
1. Check that weights directory contains `.safetensors` or `.bin` files
2. If using checkpoints, convert them first (see Step 3)
3. Verify the model directory path is correct

### "config.json not found"

**Problem**: Model config is missing.

**Solutions**:
1. Copy `config.json` from base model repository:
   ```powershell
   huggingface-cli download Qwen/Qwen3-0.6B config.json --local-dir "your_model_dir"
   ```
2. Or use `extract_hf_from_ckpt.py` which downloads it automatically

### "Tokenizer files not found"

**Problem**: Tokenizer files are missing.

**Solutions**:
1. Download tokenizer files from base model:
   ```powershell
   huggingface-cli download Qwen/Qwen3-0.6B tokenizer* vocab.json merges.txt --local-dir "your_model_dir"
   ```
2. Or use `extract_hf_from_ckpt.py` which downloads them automatically

### "Authentication failed"

**Problem**: Can't authenticate with Hugging Face.

**Solutions**:
1. Run `huggingface-cli login` and enter your token
2. Or set `HF_TOKEN` in `.env` file
3. Get token from: https://huggingface.co/settings/tokens

### "Repository creation failed"

**Problem**: Can't create the repository.

**Solutions**:
1. Check repository name is valid (lowercase, hyphens/underscores only)
2. Verify you have permission to create repos under that namespace
3. Repository might already exist - that's OK, files will be updated

## Complete Example Workflow

```powershell
# 1. Find your model files
python scripts/conversion/prepare_model_for_hf.py

# 2. Upload model (if files are ready)
python scripts/conversion/upload_model_to_hf.py `
    --model-dir "dakota_rl_training/outputs/weights/step_400" `
    --repo-id "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"

# 3. Verify upload
# Visit: https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL
```

## Next Steps

After successful upload:

1. **Test the model**: Use Hugging Face's inference API or download locally
2. **Update documentation**: Add any additional information to MODEL_CARD.md
3. **Share the model**: Share the Hugging Face link with others
4. **Create a Space**: Deploy a Gradio interface (see `huggingface_space/`)

## Files Reference

- **Upload script**: `scripts/conversion/upload_model_to_hf.py`
- **Preparation script**: `scripts/conversion/prepare_model_for_hf.py`
- **Model card upload**: `scripts/conversion/upload_model_card.py`
- **Checkpoint converter**: `prime-rl-framework/scripts/extract_hf_from_ckpt.py`
- **Model card**: `MODEL_CARD.md`

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the script output for specific error messages
3. Verify all prerequisites are met
4. Check Hugging Face Hub status: https://status.huggingface.co

