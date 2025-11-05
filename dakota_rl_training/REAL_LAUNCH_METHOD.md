# HOW TO ACTUALLY LAUNCH PRIME-RL TRAINING

## THE REALITY: Prime-RL Uses Command Line, NOT Web UI

Prime-RL does NOT have a "Create Training Job" button in the web UI. 
You need to **SSH into a Linux instance** and run the command.

---

## OPTION 1: Use Prime Intellect Compute Platform (Provision Instance First)

### Step 1: Provision/Rent an Instance

1. Go to: https://app.primeintellect.ai
2. Look for:
   - **"Compute"** or **"Instances"** 
   - **"Rent GPU"** or **"Provision Instance"**
   - **"Workbench"** → **"New Instance"**
3. Select: **8x A100 GPUs** (or equivalent)
4. **Start/Launch** the instance
5. **Wait** for it to provision (10-30 minutes)

### Step 2: Get SSH Access

Once instance is running:
1. Find **"SSH"** button or **"Connect"** button
2. Copy the SSH command they give you (looks like: `ssh user@instance-ip`)
3. Or get SSH key/credentials from dashboard

### Step 3: SSH Into Instance

```bash
ssh user@your-instance-ip
# Or whatever command Prime Intellect gives you
```

### Step 4: Upload Your Config Files

**On your Windows machine**, use SCP or copy files manually:

```powershell
# From PowerShell on Windows
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml user@instance-ip:/home/user/dakota_rl_training/configs/
```

OR manually copy/paste the files via SSH.

### Step 5: Install Prime-RL on Instance

```bash
# Clone Prime-RL if not already installed
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### Step 6: Run Training Command

```bash
cd prime-rl

uv run rl \
  --trainer @ ../dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml \
  --inference @ ../dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ../dakota_rl_training/outputs/grpo_30b
```

---

## OPTION 2: Use Your Own Linux Machine/Server

If you have access to a Linux machine with GPUs:

### Step 1: Transfer Files

```powershell
# From Windows PowerShell
scp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml user@linux-server:/path/to/configs/
```

### Step 2: SSH In

```bash
ssh user@your-linux-server
```

### Step 3: Install Prime-RL

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### Step 4: Run Training

```bash
cd prime-rl

uv run rl \
  --trainer @ /path/to/configs/train_30b.toml \
  --orchestrator @ /path/to/configs/orch_30b.toml \
  --inference @ /path/to/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir /path/to/outputs/grpo_30b
```

---

## OPTION 3: Use WSL2 on Windows (Single GPU Only)

If you have WSL2 with GPU access:

### Step 1: Open WSL2

```powershell
wsl
```

### Step 2: Install Prime-RL in WSL

```bash
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies
uv sync
```

### Step 3: Copy Config Files to WSL

```powershell
# From PowerShell
wsl cp C:\Users\chris\Dakota1890\dakota_rl_training\configs\*.toml ~/dakota_rl_training/configs/
```

### Step 4: Run Training (Single GPU)

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --output-dir ~/dakota_rl_training/outputs/grpo_30b
```

---

## 📋 QUICK REFERENCE: The Command You Need

```bash
cd prime-rl

uv run rl \
  --trainer @ ../dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml \
  --inference @ ../dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ../dakota_rl_training/outputs/grpo_30b
```

---

## 🎯 What Prime Intellect Web UI Actually Does

The https://app.primeintellect.ai website is for:
- ✅ **Managing/renting compute instances** (GPUs)
- ✅ **Viewing environments** (like your `harleycooper/dakota1890`)
- ✅ **Managing SSH keys** for instances
- ✅ **Monitoring running instances**

It does NOT:
- ❌ Launch training jobs via UI
- ❌ Have a "Create Training Job" button
- ❌ Upload config files for training

---

## 🚨 Summary

**You MUST:**
1. Provision/rent a Linux instance with GPUs (via Prime Intellect or elsewhere)
2. SSH into that instance
3. Upload your config files
4. Run the `uv run rl` command

**There is NO web UI button to launch training.**

---

## Need Help Finding Instance Provisioning?

Look on https://app.primeintellect.ai for:
- "Compute" section
- "Instances" section  
- "Workbench" section
- "Rent GPU" or "Provision" buttons

If you can't find it, Prime Intellect might use a different interface. Contact their support or check their docs.

