# SSH Key Information for Prime Intellect

## Your SSH Keys Location

**Windows Path:** `C:\Users\chris\.ssh\`

## Available Keys

### 1. Your Local Key (id_ed25519)
- **Private Key:** `C:\Users\chris\.ssh\id_ed25519`
- **Public Key:** `C:\Users\chris\.ssh\id_ed25519.pub`
- **Key Type:** ED25519
- **Created:** April 27, 2025

### 2. Prime Intellect Key (DakotaRL3)
- **Status:** NOT found in your `.ssh` directory
- **Note:** This key is deployed BY Prime Intellect when you create an instance
- **Location:** Should be downloaded from Prime Intellect dashboard

---

## What Public Key to Paste

### Option A: Use Your Local Key (id_ed25519)

**Read your public key:**
```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

**Copy the entire output** - it will look like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... chris@PC
```

**Paste this into Prime Intellect:**
1. Go to Prime Intellect dashboard
2. Find "SSH Keys" or "Public Keys" section
3. Click "Add SSH Key" or "Add Public Key"
4. Paste the entire public key
5. Give it a name: `chris-local-key` or `id_ed25519`

---

## Option B: Use Prime Intellect's Key (DakotaRL3)

**If Prime Intellect provides a key:**

1. **Download the key** from Prime Intellect dashboard
   - Look for "Download SSH Key" or "DakotaRL3" button
   - Save it as: `C:\Users\chris\.ssh\DakotaRL3`

2. **Use it to connect:**
   ```powershell
   ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@<instance-ip>
   ```

**Note:** This key is already authorized on the instance - you don't need to paste it anywhere.

---

## How to Add Your Public Key to Instance

### Method 1: Via Prime Intellect Web Console (Easiest)

1. Go to https://app.primeintellect.ai
2. Find your instance
3. Click **"Console"** or **"Terminal"** button (opens web terminal)
4. Run these commands in the web terminal:

```bash
# Create .ssh directory
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key (paste your id_ed25519.pub content)
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... chris@PC" >> ~/.ssh/authorized_keys

# Fix permissions
chmod 600 ~/.ssh/authorized_keys

# Verify
cat ~/.ssh/authorized_keys
```

### Method 2: Via SSH (if you already have access)

If you can already SSH in (using DakotaRL3 key or password):

```bash
# On the instance
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... chris@PC" >> ~/.ssh/authorized_keys

chmod 600 ~/.ssh/authorized_keys
```

---

## Quick Reference

### Your Public Key Location
```
C:\Users\chris\.ssh\id_ed25519.pub
```

### To Display Your Public Key
```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

### To Connect Using Your Key
```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 1234 root@<instance-ip>
```

### To Connect Using Prime Intellect Key (if downloaded)
```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@<instance-ip>
```

---

## Troubleshooting

### "Permission denied (publickey)"
- Your public key is not in `~/.ssh/authorized_keys` on the server
- Use web console to add it (Method 1 above)

### "No such file or directory" for DakotaRL3
- Prime Intellect hasn't provided this key yet
- Use your local `id_ed25519` key instead
- Add it via web console

### "Too many authentication failures"
- SSH is trying multiple keys
- Specify the exact key: `ssh -i $env:USERPROFILE\.ssh\id_ed25519 ...`

---

## Summary

**Key Name:** `id_ed25519` (your local key)  
**Public Key File:** `C:\Users\chris\.ssh\id_ed25519.pub`  
**Where to Paste:** `~/.ssh/authorized_keys` on the Prime Intellect instance  
**How to Add:** Use Prime Intellect web console (easiest method)





