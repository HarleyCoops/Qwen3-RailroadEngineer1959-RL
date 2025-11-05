# IMMEDIATE NEXT STEPS - After Instance Provisions

## ✅ Step-by-Step Checklist

### 1. Wait for Instance to Be Ready
- Status should change from "Provisioning" → "Running" → "Ready"
- Usually takes 10-30 minutes
- You'll see an IP address or hostname when ready

### 2. Get SSH Connection Info
Once instance is ready, look for:
- **"SSH"** button
- **"Connect"** button  
- **"Terminal"** button
- **SSH command** showing: `ssh user@ip-address`
- **SSH key** download link

**Copy the SSH command** - you'll need it!

### 3. SSH Into Instance

**On Windows PowerShell:**
```powershell
# Use the SSH command from Prime Intellect dashboard
ssh user@your-instance-ip
```

**Or if they give you a key file:**
```powershell
ssh -i path/to/key.pem user@your-instance-ip
```

### 4. Upload Your Config Files

**While still in PowerShell on Windows**, upload files:

```powershell
# Upload all 3 config files at once
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml user@your-instance-ip:/home/user/dakota_rl_training/configs/
```

**If the directory doesn't exist yet, create it first:**
```bash
# On the instance (after SSH)
mkdir -p ~/dakota_rl_training/configs
```

**Then upload:**
```powershell
# From Windows PowerShell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\train_30b.toml user@your-instance-ip:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\orch_30b.toml user@your-instance-ip:~/dakota_rl_training/configs/
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\infer_30b.toml user@your-instance-ip:~/dakota_rl_training/configs/
```

### 5. Verify Files Are Uploaded

**On the instance (after SSH):**
```bash
ls -la ~/dakota_rl_training/configs/
```

You should see:
```
train_30b.toml
orch_30b.toml
infer_30b.toml
```

### 6. Clone/Install Prime-RL

**On the instance:**
```bash
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### 7. Verify GPU Access

```bash
nvidia-smi
```

You should see your GPUs listed.

### 8. Launch Training!

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

### 9. Monitor Training

**In the same terminal**, you'll see logs streaming.

**Or in a separate terminal (SSH again):**
```bash
tail -f ~/dakota_rl_training/outputs/grpo_30b/logs/trainer/rank_0.log
```

---

## 📋 Quick Copy-Paste Commands

**Once instance is ready, SSH in and run:**

```bash
# 1. Create directory
mkdir -p ~/dakota_rl_training/configs

# 2. Clone Prime-RL
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 4. Install dependencies
uv sync

# 5. Verify GPUs
nvidia-smi

# 6. Launch training (after uploading configs!)
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

## 🔍 What to Look For

**Good signs:**
- ✅ "Initializing model..."
- ✅ "Starting RL trainer..."
- ✅ GPU utilization > 0% in `nvidia-smi`
- ✅ "Step 1/500" messages

**Bad signs:**
- ❌ "Config file not found" → Check file paths
- ❌ "Out of memory" → Reduce batch_size in orch_30b.toml
- ❌ "Connection refused" → Check inference server started

---

## ⚠️ If Upload Fails

**Alternative: Copy-paste config contents**

1. Open each `.toml` file on Windows
2. Copy all contents
3. SSH into instance
4. Create file: `nano ~/dakota_rl_training/configs/train_30b.toml`
5. Paste contents
6. Save: `Ctrl+X`, then `Y`, then `Enter`
7. Repeat for other configs

---

**Once your instance is ready, start with Step 2!**

