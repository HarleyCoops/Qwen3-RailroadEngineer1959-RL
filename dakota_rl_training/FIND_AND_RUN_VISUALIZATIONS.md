# Find Orchestrator Run ID and Run Visualizations

## Step 1: Find Orchestrator Run ID

**Option A: Check W&B Dashboard (Easiest)**
1. Go to: https://wandb.ai/christian-cooper-us/dakota-rl-grammar
2. Look for run named: `dakota-0.6b-ledger-test-400-orchestrator`
3. Copy the run ID from the URL (e.g., `abc123xyz`)

**Option B: Check Server Wandb Directory**
```bash
# SSH into server
ssh -i C:\Users\chris\.ssh\prime_rl_key root@185.216.20.236 -p 1234

# List wandb runs
ls -la ~/dakota_rl_training/outputs/ledger_test_400/wandb/

# Look for orchestrator run directory (will have different name than trainer)
# The run ID is in the directory name: run-YYYYMMDD_HHMMSS-<RUN_ID>
```

## Step 2: Run All Visualizations

**On Local Machine (PowerShell):**

```powershell
# Set run IDs
$env:TRAINER_ID = "yut26kcm"
$env:ORCHESTRATOR_ID = "<paste_orchestrator_id_here>"

# Main RL visualizations (4 plots)
python scripts/create_rl_visualizations.py `
    --trainer-id $env:TRAINER_ID `
    --orchestrator-id $env:ORCHESTRATOR_ID `
    --project dakota-rl-grammar `
    --entity christian-cooper-us `
    --output-dir wandb_visualizations

# Reward ledger plot (if CSV exists)
python scripts/analysis/plot_reward_ledger.py `
    --csv wandb_analysis/reward_ledger.csv `
    --out wandb_analysis/reward_ledger.png

# Generate ledger snippet table
python scripts/analysis/make_ledger_snippet.py

# Export comprehensive analysis
python scripts/analysis/export_comprehensive_analysis.py `
    --trainer-run $env:TRAINER_ID `
    --orchestrator-run $env:ORCHESTRATOR_ID

# Create detailed W&B report
python scripts/analysis/create_wandb_report.py `
    --trainer-run $env:TRAINER_ID `
    --orchestrator-run $env:ORCHESTRATOR_ID
```

## Step 3: Verify Outputs

After running, check these files exist:
- ✅ `wandb_visualizations/reward_progression.png`
- ✅ `wandb_visualizations/training_metrics.png`
- ✅ `wandb_visualizations/performance_metrics.png`
- ✅ `wandb_visualizations/comprehensive_dashboard.png`
- ✅ `wandb_analysis/reward_ledger.png` (if CSV exists)
- ✅ `wandb_analysis/reward_ledger_head_tail.md` (if CSV exists)

## Quick One-Liner to Find Orchestrator ID

**On Server:**
```bash
# Find orchestrator run ID from wandb directory
ls ~/dakota_rl_training/outputs/ledger_test_400/wandb/ | grep orchestrator | tail -1 | sed 's/run-.*-//'
```

Or check W&B dashboard - it's the fastest way!

