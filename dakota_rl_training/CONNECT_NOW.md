# Connect to Prime Intellect Instance - Ready to Go!

## Your Setup is Complete ✅

- **Key File:** `C:\Users\chris\.ssh\DakotaRL3` ✅
- **Instance IP:** 65.109.75.43
- **Port:** 1234
- **User:** root

---

## Connect Now

**Run this command in PowerShell:**

```powershell
ssh -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43
```

**Or the simpler version (if SSH finds the key automatically):**

```powershell
ssh root@65.109.75.43 -p 1234
```

---

## First Connection

You might see a prompt like:
```
The authenticity of host '65.109.75.43' can't be established...
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter.

---

## Once Connected

You'll be on the Prime Intellect instance. Then you can:

1. **Upload config files:**
   ```powershell
   # From a NEW PowerShell window (keep SSH session open)
   scp dakota_rl_training\configs\*.toml root@65.109.75.43:/root/dakota_rl_training/configs/
   ```

2. **Set up Prime-RL** (see `LAUNCH_1000_STEPS.md`)

3. **Launch training** (see `LAUNCH_1000_STEPS.md`)

---

## Troubleshooting

### "Permission denied (publickey)"
- The key might need different permissions
- Try: `ssh -v -i $env:USERPROFILE\.ssh\DakotaRL3 -p 1234 root@65.109.75.43` (verbose mode)

### "No such file or directory"
- Make sure the path is correct: `$env:USERPROFILE\.ssh\DakotaRL3`
- Check: `Test-Path $env:USERPROFILE\.ssh\DakotaRL3` (should return `True`)

### Connection works but then drops
- Normal - you're connected! Start setting up the instance.

---

## Next Steps After SSH Works

See `LAUNCH_1000_STEPS.md` for:
- Uploading config files
- Setting up Prime-RL
- Launching training with 1000 steps







