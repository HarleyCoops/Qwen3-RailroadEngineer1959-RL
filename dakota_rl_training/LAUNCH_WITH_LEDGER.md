# Dakota RL Training Launch Command with Reward Ledger Logging

## Prerequisites

1. **Clean baseline model**: Starting fresh (no checkpoint resume)
2. **Wandb logged in**: `wandb login` or `export WANDB_API_KEY=...`
3. **Environment installed**: `pip install -e environments/dakota_grammar_translation`
4. **Prime-RL installed**: In your prime-rl directory

## Basic Command Structure

The typical `uv run rl` command with wandb and ledger logging:

```bash
cd ~/prime-rl  # or wherever prime-rl is installed

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --output-dir ~/dakota_rl_training/outputs/ledger_run \
  --wandb.project dakota-rl-grammar \
  --wandb.name dakota-0.6b-ledger-baseline
```

## Revised Command with Clean Baseline

For a clean Qwen baseline model start with ledger logging:

```bash
cd ~/prime-rl

uv run rl \
  --trainer @ ~/dakota_rl_training/configs/train_30b.toml \
  --orchestrator @ ~/dakota_rl_training/configs/orch_30b.toml \
  --inference @ ~/dakota_rl_training/configs/infer_30b.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --output-dir ~/dakota_rl_training/outputs/ledger_baseline \
  --wandb.project dakota-rl-grammar \
  --wandb.name dakota-0.6b-ledger-baseline-$(date +%Y%m%d-%H%M%S) \
  --trainer.wandb.project dakota-rl-grammar \
  --trainer.wandb.name dakota-0.6b-ledger-baseline-trainer \
  --orchestrator.wandb.project dakota-rl-grammar \
  --orchestrator.wandb.name dakota-0.6b-ledger-baseline-orchestrator
```

## Key Points

1. **Clean baseline**: No `--ckpt.resume-step` flag = starts from scratch
2. **Model**: Uses model from config files (Qwen/Qwen3-0.6B - the small instruct model for RL)
3. **Wandb enabled**: Both trainer and orchestrator log to W&B
4. **Ledger logging**: Automatically enabled via DakotaGrammarRubric (ledger data in states)

## Ledger Integration

The ledger logging happens automatically:

1. **Rubric computes ledger**: `DakotaGrammarRubric.score()` builds ledger and stores it
2. **Environment exposes ledger**: `DakotaGrammarEnv.get_reward_ledger()` retrieves it
3. **States contain ledger**: Verifiers framework passes state dicts through
4. **Orchestrator extracts**: Ledger data extracted from `generate_outputs.state`
5. **Logging**: Logged to W&B (`ledger/*`) and CSV (`wandb_analysis/reward_ledger.csv`)

## Post-Training Analysis

After training completes:

```bash
# Generate ledger visualization
python scripts/analysis/plot_reward_ledger.py

# Generate markdown table for README
python scripts/analysis/make_ledger_snippet.py

# Create comprehensive visualizations
python scripts/create_rl_visualizations.py \
    --trainer-id <trainer_run_id> \
    --orchestrator-id <orchestrator_run_id> \
    --project dakota-rl-grammar \
    --entity christian-cooper-us
```

## Multi-GPU Example

For distributed training:

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

## Notes

- Ledger logging requires the updated `DakotaGrammarRubric` (already implemented)
- CSV file is written to `wandb_analysis/reward_ledger.csv` (relative to workspace root)
- W&B logs appear under `ledger/*` namespace in both trainer and orchestrator runs
- The ledger provides full transparency into reward computation without changing training behavior

