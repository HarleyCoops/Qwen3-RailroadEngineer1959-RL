# Hugging Face Space Debut Guide

Complete step-by-step guide for deploying your Dakota Grammar RL model to Hugging Face Spaces.

## Prerequisites

- ✅ Model published on HuggingFace Hub: `HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL`
- ✅ HuggingFace account with write access
- ✅ Files prepared in `huggingface_space/` directory

## Quick Start (5 Minutes)

### Step 1: Create the Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `HarleyCooper/Dakota-.6B` (or your preferred name)
   - **SDK**: Select **"Gradio"**
   - **Hardware**: Select **"GPU"** → Choose **"T4"** (sufficient for 0.6B model)
   - **Visibility**: Public (for free GPU) or Private
4. Click **"Create Space"**

### Step 2: Upload Files

**Option A: Automatic Push (Recommended)**
```powershell
# Make sure you're logged in
huggingface-cli login

# Push files automatically
python scripts/push_space.py --space-name "HarleyCooper/Dakota-.6B"
```

**Option B: Manual Upload**
1. In your Space's file browser, click **"Add file"** → **"Upload files"**
2. Upload these files from `huggingface_space/`:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Click **"Commit changes"**

### Step 3: Watch It Build

1. Go to the **"Deploy"** tab in your Space
2. Watch the build logs:
   - Installing dependencies (`pip install`)
   - Downloading model from HuggingFace Hub (~1.5GB)
   - Launching Gradio interface
3. Wait for **"Running"** status (usually 5-10 minutes first time)

### Step 4: Test Your Space

Visit your Space URL: `https://huggingface.co/spaces/HarleyCooper/Dakota-.6B`

Try example prompts:
- "Translate to Dakota: Hello"
- "Translate to Dakota: That is my horse"
- "What does 'Háu' mean?"

## What Gets Deployed

```
huggingface_space/
├── app.py              # Gradio web interface
├── requirements.txt    # Python dependencies
└── README.md          # Space description & metadata
```

## Features Included

- ✅ Automatic model loading from HuggingFace Hub
- ✅ Proper chat formatting (matches RL training)
- ✅ Adjustable generation parameters (temperature, max tokens)
- ✅ Example prompts
- ✅ Clean response extraction
- ✅ Debug mode (set `DEBUG_INFERENCE=true` env var)

## Updating Your Space

After making changes to `huggingface_space/app.py`:

```powershell
# Automatic update
python scripts/push_space.py --space-name "HarleyCooper/Dakota-.6B"
```

The Space will automatically rebuild with your changes.

## Troubleshooting

### Build Fails

**Check logs in "Deploy" tab:**
- Missing dependencies? Add to `requirements.txt`
- Import errors? Check Python syntax
- Model download fails? Verify model name is correct

### Space Won't Start

- Check GPU availability (may need to wait for free GPU)
- Verify `app.py` has no syntax errors
- Check `requirements.txt` has all dependencies

### Model Output Issues

- Enable debug mode: Set `DEBUG_INFERENCE=true` in Space settings
- Check generation parameters in `app.py`
- Verify chat formatting matches training format

### Upload Script Fails

```powershell
# Re-login to HuggingFace
huggingface-cli login

# Check Space name is correct
# Verify Space exists: https://huggingface.co/spaces/YOUR-SPACE-NAME
```

## Space Settings

**Recommended Settings:**
- **Hardware**: GPU T4 (free tier)
- **Auto-deploy**: Enabled (rebuilds on file changes)
- **Environment Variables**: 
  - `DEBUG_INFERENCE=false` (set to `true` for debugging)

## Cost

- **Free Tier**: T4 GPU available for public Spaces
- **Paid Tier**: A10 GPU for better performance (optional)

## Next Steps

1. **Share your Space**: Copy the URL and share with others
2. **Monitor Usage**: Check Space metrics in HuggingFace dashboard
3. **Iterate**: Update `app.py` and push changes as needed
4. **Documentation**: Update `README.md` with usage examples

## Files Reference

- **Deployment Checklist**: `huggingface_space/DEPLOYMENT.md`
- **Auto-Push Script**: `docs/AUTO_PUSH_SPACE.md`
- **Debug Guide**: `docs/SPACE_DEBUG_GUIDE.md`
- **Inference Details**: `docs/INFERENCE_DEPLOYMENT.md`

## Support

- **HuggingFace Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://gradio.app/docs/
- **Space Logs**: Check "Deploy" tab for detailed build logs

---

**You're ready to debut your model!** 🚀

