# SSH Key Download Clarification

## What You Actually Need

**For SSH client authentication, you only need the PRIVATE KEY.**

- **Private Key:** This is what you use to connect (`DakotaRL3`)
- **Public Key:** This is already on the Prime Intellect server (they added it when you deployed)

---

## What Prime Intellect Provides

Prime Intellect typically gives you **one file** - the **private key**.

### Option 1: Single Private Key File (Most Common)
- **File name:** `DakotaRL3` or `DakotaRL3.pem` or `id_rsa` or similar
- **What it is:** Private key
- **Save as:** `C:\Users\chris\.ssh\DakotaRL3` (no extension, or keep `.pem` if provided)
- **Use it:** `ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43`

### Option 2: Two Files (Less Common)
If Prime Intellect provides both:
- **Private key:** `DakotaRL3` or `DakotaRL3.pem` → Save as `C:\Users\chris\.ssh\DakotaRL3`
- **Public key:** `DakotaRL3.pub` → You DON'T need this (it's already on the server)

---

## How to Download from Prime Intellect

1. Go to https://app.primeintellect.ai
2. Find your instance (65.109.75.43)
3. Look for:
   - **"Download SSH Key"** button
   - **"SSH Keys"** → **"DakotaRL3"** → **"Download"**
   - **"Private Key"** download link

**You're downloading the PRIVATE key** (even if it's not labeled clearly).

---

## Save the Key

**If it's a single file:**
```powershell
# Save it as DakotaRL3 (no extension)
# Location: C:\Users\chris\.ssh\DakotaRL3
```

**If it has an extension (.pem, .key):**
```powershell
# You can keep the extension or remove it
# C:\Users\chris\.ssh\DakotaRL3.pem  OR  C:\Users\chris\.ssh\DakotaRL3
# Both work fine
```

---

## Set Permissions (Windows)

Windows doesn't enforce SSH key permissions like Linux, but you can set them:

```powershell
# Remove inheritance and grant only your user access
icacls $env:USERPROFILE\.ssh\DakotaRL3 /inheritance:r
icacls $env:USERPROFILE\.ssh\DakotaRL3 /grant:r "$env:USERNAME:R"
```

---

## Connect

```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43
```

**Or if it has an extension:**
```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3.pem -p 1234 root@65.109.75.43
```

---

## Summary

**What to download:** Just the **private key** file  
**What to save:** `C:\Users\chris\.ssh\DakotaRL3` (or `DakotaRL3.pem` if it has extension)  
**What you DON'T need:** Public key (already on server)  
**File count:** Usually **1 file** (the private key)

If Prime Intellect gives you 2 files, only use the **private key** (the one without `.pub` extension).





