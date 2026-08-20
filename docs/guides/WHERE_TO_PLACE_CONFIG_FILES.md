# Where to Place Config Files on Prime Intellect Instance

## Overview

When you SSH into your Prime Intellect instance, you need to upload the 3 config files and place them in the correct directory structure for `prime-rl` to use them.

## Directory Structure

**DO NOT** put files at the root (`/`)! Instead, create a project directory:

```
/home/ubuntu/dakota-rl-training/
├── configs/
│   ├── orch_30b.toml
│   ├── train_30b.toml
│   └── infer_30b.toml
└── (optional: datasets, outputs, etc.)
```

## Step-by-Step Setup

### Step 1: SSH into Instance (Once Ready)

Once SSH is available (2-4 minutes after deployment):

```bash
# Click "SSH Connection" button in Prime Intellect dashboard
# Or use the SSH command shown in the dashboard
```

### Step 2: Create Project Directory

```bash
cd ~
mkdir -p dakota-rl-training/configs
cd dakota-rl-training
```

### Step 3: Upload Config Files

**Option A: Using SCP (from your Windows machine)**

Open a **new PowerShell window** on your Windows machine:

```powershell
# Navigate to your project
cd C:\Users\chris\Dakota1890

# Upload config files
scp dakota_rl_training\configs\orch_30b.toml ubuntu@<instance-ip>:~/dakota-rl-training/configs/
scp dakota_rl_training\configs\train_30b.toml ubuntu@<instance-ip>:~/dakota-rl-training/configs/
scp dakota_rl_training\configs\infer_30b.toml ubuntu@<instance-ip>:~/dakota-rl-training/configs/
```

**Option B: Copy-Paste Content (Simpler)**

1. SSH into instance
2. Create files directly:

```bash
cd ~
mkdir -p dakota-rl-training/configs
cd dakota-rl-training/configs

# Create orch_30b.toml
nano orch_30b.toml
# Paste content, Ctrl+X, Y, Enter to save

# Create train_30b.toml
nano train_30b.toml
# Paste content, Ctrl+X, Y, Enter to save

# Create infer_30b.toml
nano infer_30b.toml
# Paste content, Ctrl+X, Y, Enter to save
```

**Option C: Clone Your Repo (Best for Full Setup)**

If you want to clone your entire repo:

```bash
cd ~
git clone https://github.com/HarleyCoops/Dakota1890.git
cd Dakota1890
# Configs are already at: dakota_rl_training/configs/orch_30b.toml
```

### Step 4: Install prime-rl Framework

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone prime-rl framework
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install dependencies
uv sync && uv sync --all-extras
```

### Step 5: Verify Config Files

```bash
# From your project directory
ls -la ~/dakota-rl-training/configs/
# Should show: orch_30b.toml, train_30b.toml, infer_30b.toml

# Verify they're readable
cat ~/dakota-rl-training/configs/orch_30b.toml
```

### Step 6: Launch Training

```bash
cd ~/prime-rl

# Launch GRPO training
uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota-rl-training/outputs
```

## Important Notes

### File Paths

- **Config paths**: Use `@` symbol before paths (e.g., `@ ~/dakota-rl-training/configs/orch_30b.toml`)
- **Absolute paths**: Use `~/` for home directory or full paths like `/home/ubuntu/dakota-rl-training/configs/`
- **Relative paths**: Can be relative to current directory if you're in the right location

### Directory Structure Best Practice

```
/home/ubuntu/
├── dakota-rl-training/          # Your project
│   ├── configs/                 # Config files here
│   │   ├── orch_30b.toml
│   │   ├── train_30b.toml
│   │   └── infer_30b.toml
│   └── outputs/                 # Training outputs (created automatically)
│       ├── checkpoints/
│       ├── weights/
│       └── rollouts/
└── prime-rl/                    # prime-rl framework
    ├── src/
    ├── configs/
    └── ...
```

### Why Not Root?

- **Root (`/`)**: System directory, protected, bad practice
- **Home (`~/`)**: User directory, safe, recommended
- **Project directory**: Organized, easy to find, best practice

## Quick Reference

**Upload configs to:**
```
~/dakota-rl-training/configs/
```

**Then run:**
```bash
cd ~/prime-rl
uv run rl \
  --trainer @ ~/dakota-rl-training/configs/train_30b.toml \
  --orchestrator @ ~/dakota-rl-training/configs/orch_30b.toml \
  --inference @ ~/dakota-rl-training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3
```

## Alternative: Use Prime Intellect Web UI

If Prime Intellect's web UI supports file uploads:

1. Go to your instance dashboard
2. Look for "Upload Files" or "File Manager"
3. Upload configs to `/home/ubuntu/dakota-rl-training/configs/`
4. Use SSH to verify and launch training

## Summary

**Do NOT** put files at root (`/`)  
**DO** put files in a project directory like `~/dakota-rl-training/configs/`  
**Then** reference them with `@` symbol in the `uv run rl` command

The `@` symbol tells prime-rl to load the config from that file path.

