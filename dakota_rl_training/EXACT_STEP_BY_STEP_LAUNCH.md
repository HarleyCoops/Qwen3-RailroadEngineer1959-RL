# HOW TO LAUNCH RL TRAINING ON PRIME INTELLECT - STEP BY STEP

## ️ IMPORTANT: Exact File Locations

Your config files are here:
```
C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml
C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml
```

---

## STEP 1: Open Prime Intellect Dashboard

1. Open your web browser
2. Go to: **https://app.primeintellect.ai**
3. **Log in** with your account (if not already logged in)

---

## STEP 2: Find the "Create Job" Button

Look for one of these buttons (may vary by UI):
- **"New Training Job"** button (usually top-right or main dashboard)
- **"Create Job"** button
- **"Launch Training"** button
- **"Start Training"** button
- **"+"** button (then select "Training Job")
- **"Train"** menu item → "New Training"

**If you can't find it:**
- Look for a **sidebar menu** on the left
- Look for a **"Workbench"** or **"Training"** section
- Look for a **"Jobs"** section → then click "Create New Job"

---

## STEP 3: Select Training Type

When you click create, you'll see options:
1. **Select "RL Training"** or **"Reinforcement Learning"**
2. OR select **"Custom Training"** → then choose **"Prime-RL"**
3. OR select **"GRPO"** if that's an option

**If you see "Algorithm" dropdown:**
- Choose **"GRPO"** (Group Relative Policy Optimization)

---

## STEP 4: Configure Instance/Resources

Look for a section called:
- **"Instance Configuration"**
- **"GPU Configuration"**
- **"Resources"**
- **"Hardware"**

**Set these:**
- **Instance Type**: Select **8x A100** (or equivalent)
- **GPU Count**: **8 GPUs**
- **GPU Allocation** (if there's a separate setting):
  - **Inference GPUs**: **4** (or GPUs 0-3)
  - **Trainer GPUs**: **4** (or GPUs 4-7)

---

## STEP 5: Upload Configuration Files

**Find the file upload section.** It might be labeled:
- **"Upload Config Files"**
- **"Configuration Files"**
- **"Training Config"**
- **"Config Files"**
- **"Upload Files"** button

### Option A: Drag and Drop
1. Open File Explorer
2. Navigate to: `C:\Users\chris\Dakota1890\dakota_rl_training\configs\`
3. Select all 3 files:
   - `train_30b.toml`
   - `orch_30b.toml`
   - `infer_30b.toml`
4. **Drag and drop** them into the upload area on the website

### Option B: Click to Upload
1. Click **"Choose Files"** or **"Upload"** button
2. In the file picker:
   - Navigate to: `C:\Users\chris\Dakota1890\dakota_rl_training\configs\`
   - Select: `train_30b.toml`
   - Then upload `orch_30b.toml`
   - Then upload `infer_30b.toml`

### Option C: If There Are Separate Fields
If the UI has separate fields for each config:
- **Trainer Config**: Upload `train_30b.toml`
- **Orchestrator Config**: Upload `orch_30b.toml`
- **Inference Config**: Upload `infer_30b.toml`

---

## STEP 6: Set Model and Environment

Look for fields labeled:
- **"Base Model"** or **"Model Name"**
  - Enter: `qwen/qwen3-30b-a3b-instruct-2507`

- **"Environment ID"** or **"Environment"**
  - Enter: `harleycooper/dakota1890`

**If there's a dropdown:**
- Look for your environment in the list
- Select: `harleycooper/dakota1890`

---

## STEP 7: Set Additional Settings (if available)

Look for these optional fields:

- **"Output Directory"** or **"Output Path"**
  - Leave default or set to: `outputs/grpo_30b`

- **"W&B Project"** or **"Weights & Biases Project"**
  - Enter: `dakota-rl-grammar`

- **"Max Steps"**
  - Should already be in config (500), but verify

- **"Checkpoint Interval"**
  - Should already be in config (100), but verify

---

## STEP 8: Review and Launch

1. **Scroll down** to review all settings
2. **Check** that all 3 config files are uploaded
3. **Verify** model name and environment ID
4. **Click** one of these buttons:
   - **"Launch Training"**
   - **"Start Training"**
   - **"Submit Job"**
   - **"Create Job"**
   - **"Run"**

---

## STEP 9: Monitor Training

After launching:
1. You'll be redirected to a **job details page**
2. **Status** will show: "Provisioning" → "Running" → "Completed"
3. **Logs** will appear in the dashboard
4. **W&B Dashboard**: https://wandb.ai/your-username/dakota-rl-grammar

---

##  IF YOU CAN'T FIND THE UPLOAD BUTTON

### Try This:
1. Look for a **"Advanced"** or **"Advanced Settings"** toggle
2. Look for a **"Config"** tab or section
3. Look for **"Edit Config"** or **"Configuration"** link
4. Check if there's a **"Code"** section where you can paste config

### Alternative: Use SSH Method
If the web UI doesn't work, you'll need to:
1. **Create/Reserve an instance** on Prime Intellect
2. **SSH into that instance**
3. **Upload files** via SCP or git
4. **Run the command** manually

See `../docs/root/EXACT_LAUNCH_COMMANDS.md` for SSH instructions.

---

##  Still Stuck?

**Take screenshots:**
1. Screenshot of the main dashboard
2. Screenshot of the "Create Job" page
3. Screenshot of any error messages

**Common Issues:**
- **"File not found"**: Make sure you're uploading from the correct path
- **"Invalid config"**: Check that all 3 files are .toml files
- **"Environment not found"**: Verify `harleycooper/dakota1890` is published
- **"Model not found"**: Verify `qwen/qwen3-30b-a3b-instruct-2507` exists

---

##  QUICK CHECKLIST

Before clicking Launch:
- [ ] All 3 config files uploaded (train_30b.toml, orch_30b.toml, infer_30b.toml)
- [ ] Model name set: `qwen/qwen3-30b-a3b-instruct-2507`
- [ ] Environment ID set: `harleycooper/dakota1890`
- [ ] Instance type: 8x A100 GPUs (or equivalent)
- [ ] W&B project: `dakota-rl-grammar` (if field exists)

---

**That's it!** Once you click Launch, the job will start provisioning and training.

