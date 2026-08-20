# Push to HuggingFace Space

## Option 1: If Space Already Exists

If you've already created the Space on HuggingFace, run:

```powershell
cd C:\Users\chris\Dakota1890\huggingface_space
git remote add origin https://huggingface.co/spaces/HarleyCooper/YOUR-SPACE-NAME
git push -u origin main
```

Replace `YOUR-SPACE-NAME` with your actual Space name (e.g., `Dakota-Grammar-Demo`).

## Option 2: Push via Web Interface

1. Go to your Space: https://huggingface.co/spaces/HarleyCooper/YOUR-SPACE-NAME
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Drag and drop:
   - `app.py`
   - `README.md`
   - `requirements.txt`
5. Commit changes

## Option 3: Clone Space First (Recommended)

If the Space exists, clone it first:

```powershell
cd C:\Users\chris\Dakota1890
git clone https://huggingface.co/spaces/HarleyCooper/YOUR-SPACE-NAME hf-space-repo
cd hf-space-repo
# Copy files from huggingface_space/
Copy-Item ..\huggingface_space\* .
git add .
git commit -m "Update app.py and README"
git push
```

## Files Ready to Push

 `app.py` - Updated with error handling
 `README.md` - Updated with new metadata
 `requirements.txt` - Dependencies

All files are in: `C:\Users\chris\Dakota1890\huggingface_space\`

