# Automatic Space Deployment Script

## Quick Push to HuggingFace Space

I've created a script that automatically pushes files from `huggingface_space/` to your HuggingFace Space!

## Usage

### First Time Setup

1. **Update the Space name** in `scripts/push_space.py`:
   ```python
   DEFAULT_SPACE_NAME = "HarleyCooper/YOUR-ACTUAL-SPACE-NAME"
   ```

2. **Make sure you're logged in**:
   ```powershell
   huggingface-cli login
   ```

### Push Files

Simply run:
```powershell
python scripts/push_space.py
```

This will automatically:
-  Upload `app.py`
-  Upload `requirements.txt`
-  Upload `README.md`
-  Trigger Space rebuild

### Custom Space Name

If your Space has a different name:
```powershell
python scripts/push_space.py --space-name "HarleyCooper/Dakota-Grammar-Demo"
```

## What It Does

1. Checks for HF token (from login or `HF_TOKEN` env var)
2. Uploads each file to your Space
3. Space automatically rebuilds with new files
4. Shows upload progress and results

## After Pushing

1. Go to your Space: https://huggingface.co/spaces/YOUR-SPACE-NAME
2. Check the **"Deploy"** tab to watch rebuild
3. Test the updated Space!

## Files Pushed

- `app.py` - Your Gradio interface
- `requirements.txt` - Dependencies
- `README.md` - Space description

## Troubleshooting

**If upload fails:**
- Make sure you're logged in: `huggingface-cli login`
- Check Space name is correct
- Verify Space exists on HuggingFace
- Check you have write access to the Space

**If Space doesn't rebuild:**
- Check Space Settings → Auto-deploy is enabled
- Manually trigger rebuild in Deploy tab

