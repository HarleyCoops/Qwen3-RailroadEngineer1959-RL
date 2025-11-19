# SSH Connection to Prime Intellect Instance

## Your Instance Details
- **IP:** 65.109.75.43
- **Port:** 1234
- **User:** root
- **Key Name:** DakotaRL3

---

## Option 1: Use DakotaRL3 Key (If You Have It)

### Step 1: Locate DakotaRL3 Key

**Check if you have it:**
```powershell
Test-Path $env:USERPROFILE\.ssh\DakotaRL3
```

**If it exists, connect:**
```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43
```

### Step 2: If You Don't Have DakotaRL3 Key

**Download it from Prime Intellect:**
1. Go to https://app.primeintellect.ai
2. Find your instance (65.109.75.43)
3. Look for:
   - **"Download SSH Key"** button
   - **"SSH Keys"** section
   - **"DakotaRL3"** download link
4. Download the key file
5. Save it as: `C:\Users\chris\.ssh\DakotaRL3`
6. Set permissions (if needed):
   ```powershell
   icacls $env:USERPROFILE\.ssh\DakotaRL3 /inheritance:r
   icacls $env:USERPROFILE\.ssh\DakotaRL3 /grant:r "$env:USERNAME:R"
   ```

**Then connect:**
```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43
```

---

## Option 2: Use Your Local Key (id_ed25519)

If DakotaRL3 isn't available, you can use your local key:

### Step 1: Add Your Public Key to Instance

**Via Prime Intellect Web Console:**
1. Go to https://app.primeintellect.ai
2. Find instance 65.109.75.43
3. Click **"Console"** or **"Terminal"** button
4. Run these commands:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINOqQ2A1bz/Cvvexy7qqGIolOCpjSkv9Tab8Cm3WorzG chris@PC" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Step 2: Connect Using Your Key

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 1234 root@65.109.75.43
```

---

## Quick Test Connection

**Try the simple command first (if DakotaRL3 is configured):**
```powershell
ssh root@65.109.75.43 -p 1234
```

**If that doesn't work, try with explicit key:**
```powershell
# Try DakotaRL3 first
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43

# Or try your local key
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 1234 root@65.109.75.43
```

---

## Troubleshooting

### "Permission denied (publickey)"
- The key isn't authorized on the server
- Use web console to add your public key (Option 2, Step 1)

### "No such file or directory" for DakotaRL3
- Download it from Prime Intellect dashboard
- Or use your local id_ed25519 key instead

### "Too many authentication failures"
- Specify the exact key: `ssh -i $env:USERPROFILE\.ssh\DakotaRL3 ...`

---

## Once Connected

After SSH works, you can:
1. Upload config files
2. Set up Prime-RL
3. Launch training

See `LAUNCH_1000_STEPS.md` for next steps.











