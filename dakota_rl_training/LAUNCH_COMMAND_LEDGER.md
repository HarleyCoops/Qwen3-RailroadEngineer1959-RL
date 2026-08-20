# Launch Command: Dakota RL Training with Reward Ledger Logging

## Complete Command for Clean Baseline Run

Starting from a clean Qwen baseline model with reward ledger logging enabled:

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --output-dir ~/dakota_rl_training/outputs/ledger_baseline_$(date +%Y%m%d-%H%M%S) \
  --wandb.project dakota-rl-grammar \
  --wandb.name dakota-0.6b-ledger-baseline
```

## Key Configuration Points

### 1. Clean Baseline (No Checkpoint Resume)
- **No `--ckpt.resume-step` flag** = starts from scratch
- Model loads fresh from HuggingFace: `Qwen/Qwen3-0.6B` (the small instruct model for RL)
- All weights reset to base model

### 2. Wandb Logging Enabled
The config files should have:
```toml
[wandb]
project = "dakota-rl-grammar"
offline = false
```

Or override via command line:
```bash
--wandb.project dakota-rl-grammar
  --wandb.name dakota-0.6b-ledger-baseline
```

### 3. Reward Ledger Integration

The ledger logging is **automatically enabled** because:
- `DakotaGrammarRubric.score()` computes and stores ledger in `_last_ledger`
- The ledger is accessible via `environment.get_reward_ledger()`
- The verifiers framework passes state dicts through `generate_outputs.state`
- The orchestrator can extract ledger from states and log it

**Note**: To fully enable ledger logging in the orchestrator, you need to patch the orchestrator to extract ledger data from states. See `dakota_rl_training/utils/orchestrator_ledger_patch.py` for the extraction logic.

## Expected Output

After running, you should see:

1. **W&B Runs**:
   - `dakota-0.6b-ledger-baseline-trainer` - Trainer metrics
   - `dakota-0.6b-ledger-baseline-orchestrator` - Orchestrator metrics + ledger data

2. **CSV File**: `wandb_analysis/reward_ledger.csv` with per-step ledger data

3. **W&B Metrics**: Under `ledger/*` namespace:
   - `ledger/char_overlap_raw`, `ledger/char_overlap_norm`
   - `ledger/affix_raw`, `ledger/affix_norm`
   - `ledger/w_char`, `ledger/w_affix`, etc.
   - `ledger/composite_predicted`, `ledger/reward_scalar`

## Post-Training Analysis

After training completes:

```bash
# Generate ledger visualization
python scripts/analysis/plot_reward_ledger.py

# Generate markdown table
python scripts/analysis/make_ledger_snippet.py

# Create comprehensive visualizations
python scripts/create_rl_visualizations.py \
    --trainer-id <trainer_run_id> \
    --orchestrator-id <orchestrator_run_id>
```

## Multi-GPU Example

For distributed training on multiple GPUs:

```bash
uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 4,5,6,7 \
  --inference-gpu-ids 0,1,2,3 \
  --output-dir ~/dakota_rl_training/outputs/ledger_baseline \
  --wandb.project dakota-rl-grammar \
  --wandb.name dakota-0.6b-ledger-baseline
```

## Model Selection

To use a different model, modify the config files or override:

```bash
--trainer.model.name Qwen/Qwen3-0.6B
--orchestrator.model.name Qwen/Qwen3-0.6B
--inference.model.name Qwen/Qwen3-0.6B
```

## Verification

After starting training, verify ledger logging is working:

1. Check W&B dashboard for `ledger/*` metrics
2. Check for `wandb_analysis/reward_ledger.csv` file creation
3. Verify ledger data appears in orchestrator run (not trainer run)

The ledger provides complete transparency into reward computation without changing training behavior.

