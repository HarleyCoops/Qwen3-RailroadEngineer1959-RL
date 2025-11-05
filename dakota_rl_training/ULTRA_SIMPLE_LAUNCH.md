# ULTRA-SIMPLE LAUNCH GUIDE

## Files to Upload (Copy These Exact Paths)

```
C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml
```

## Step-by-Step (Copy-Paste Ready)

### 1. Go to Website
```
https://app.primeintellect.ai
```

### 2. Log In
- Enter your credentials
- Click "Sign In"

### 3. Find "Create Job" Button
- Look for button that says: "New Training Job", "Create Job", "Launch Training", or "+"
- **Click it**

### 4. Select Training Type
- Choose: **"RL Training"** or **"GRPO"** or **"Prime-RL"**

### 5. Upload Files
- Click **"Upload Files"** or **"Choose Files"**
- Navigate to: `C:\Users\chris\Dakota1890\dakota_rl_training\configs\`
- Select these 3 files:
  - `train_30b.toml`
  - `orch_30b.toml`  
  - `infer_30b.toml`
- Click **"Open"** or **"Upload"**

### 6. Set Model
- Find field: **"Model"** or **"Base Model"**
- Type: `qwen/qwen3-30b-a3b-instruct-2507`

### 7. Set Environment
- Find field: **"Environment"** or **"Environment ID"**
- Type: `harleycooper/dakota1890`

### 8. Set Instance
- Find: **"Instance Type"** or **"GPUs"**
- Select: **8x A100** (or equivalent)

### 9. Launch
- Click: **"Launch Training"** or **"Start"** or **"Submit"**

## Done!

Monitor at: https://app.primeintellect.ai

---

## If You Can't Find Upload Button

**Method 1: Look for "Config" Tab**
- Click "Config" tab
- Upload files there

**Method 2: Look for "Advanced"**
- Click "Advanced Settings"
- Upload files there

**Method 3: Use SSH Instead**
- See `EXACT_LAUNCH_COMMANDS.md` for manual SSH method

