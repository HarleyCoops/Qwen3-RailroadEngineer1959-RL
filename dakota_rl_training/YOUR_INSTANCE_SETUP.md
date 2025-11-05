# YOUR EXACT SETUP - Step by Step

## Your Instance Details
- **SSH**: `ssh ubuntu@69.19.136.157`
- **Storage**: `/ephemeral` (6500GB)
- **User**: `ubuntu`

---

## STEP 1: SSH Into Instance

**From Windows PowerShell:**
```powershell
ssh ubuntu@69.19.136.157
```

**First time?** You'll see a prompt like:
```
The authenticity of host '69.19.136.157' can't be established...
Are you sure you want to continue connecting (yes/no)?
```
Type `yes` and press Enter.

---

## STEP 2: Upload Config Files

**While still in PowerShell (before SSH or in a new window):**

```powershell
# Upload all 3 config files
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml ubuntu@69.19.136.157:/home/ubuntu/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml ubuntu@69.19.136.157:/home/ubuntu/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml ubuntu@69.19.136.157:/home/ubuntu/dakota_rl_training/configs/
```

**OR upload all at once:**
```powershell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml ubuntu@69.19.136.157:/home/ubuntu/dakota_rl_training/configs/
```

**If directory doesn't exist, create it first (after SSH):**
```bash
mkdir -p ~/dakota_rl_training/configs
```

---

## STEP 3: Setup on Instance (After SSH)

**SSH in first:**
```powershell
ssh ubuntu@69.19.136.157
```

**Then run these commands on the instance:**

```bash
# Create directories
mkdir -p ~/dakota_rl_training/configs
mkdir -p ~/dakota_rl_training/outputs

# Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Verify GPU access
nvidia-smi
```

---

## STEP 4: Verify Config Files Are Uploaded

```bash
ls -la ~/dakota_rl_training/configs/
```

You should see:
```
train_30b.toml
orch_30b.toml
infer_30b.toml
```

---

## STEP 5: Launch Training!

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b
```

---

## STEP 6: Monitor Training

**In the same terminal**, you'll see logs streaming.

**Or open a new terminal and SSH again:**
```powershell
# From Windows PowerShell (new window)
ssh ubuntu@69.19.136.157
```

**Then monitor:**
```bash
tail -f ~/dakota_rl_training/outputs/grpo_30b/logs/trainer/rank_0.log
```

**Or check orchestrator logs:**
```bash
tail -f ~/dakota_rl_training/outputs/grpo_30b/logs/orchestrator.log
```

---

## 💾 About /ephemeral Storage

The `/ephemeral` path is likely for temporary storage. You have two options:

### Option A: Use Home Directory (Recommended)
Use `~/dakota_rl_training/` as shown above - it's simpler.

### Option B: Use /ephemeral (If you want more space)
```bash
# Create directories on ephemeral storage
mkdir -p /ephemeral/dakota_rl_training/configs
mkdir -p /ephemeral/dakota_rl_training/outputs

# Upload configs to ephemeral
# (Change scp destination to /ephemeral/dakota_rl_training/configs/)

# Launch with ephemeral output
uv run rl \
  --trainer @ /ephemeral/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ /ephemeral/dakota_rl_training/configs/orch_30b.toml \
  --inference @ /ephemeral/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir /ephemeral/dakota_rl_training/outputs/grpo_30b
```

**Note:** `/ephemeral` data may be lost if instance is stopped. Use for temporary outputs.

---

## 🔍 Troubleshooting

**If scp fails:**
- Make sure you're in PowerShell (not CMD)
- Make sure SSH worked first
- Try uploading one file at a time

**If "config file not found":**
- Check path: `ls ~/dakota_rl_training/configs/`
- Verify file names match exactly

**If "Permission denied":**
- Check file permissions: `chmod 644 ~/dakota_rl_training/configs/*.toml`

**If "uv: command not found":**
- Make sure you ran: `source $HOME/.local/bin/env`
- Or use full path: `$HOME/.local/bin/uv run rl ...`

---

## ✅ Quick Checklist

- [ ] SSH'd into instance: `ssh ubuntu@69.19.136.157`
- [ ] Uploaded 3 config files via scp
- [ ] Cloned prime-rl: `git clone https://github.com/PrimeIntellect-ai/prime-rl.git`
- [ ] Installed uv and dependencies
- [ ] Verified GPUs: `nvidia-smi`
- [ ] Launched training command
- [ ] Training is running!

---

**Start with SSH, then upload files, then run the setup commands!**

