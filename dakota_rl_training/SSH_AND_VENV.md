# SSH and Venv Commands

## Exit Venv in PowerShell

```powershell
deactivate
```

## SSH into Remote Server

```powershell
ssh -i C:\Users\chris\.ssh\prime_rl_key root@185.216.20.236 -p 1234
```

## Full Workflow

```powershell
# Exit venv if you're in one
deactivate

# SSH into server
ssh -i C:\Users\chris\.ssh\prime_rl_key root@185.216.20.236 -p 1234
```

