
from pathlib import Path
from dakota_rl_training.tinker_integration.ledger import export_reward_ledger

metrics_path = Path("dakota_rl_training/outputs/tinker_smoke/metrics.jsonl")
ledger_csv = Path("wandb_analysis/reward_ledger.csv")

if metrics_path.exists():
    print(f"Exporting ledger from {metrics_path} to {ledger_csv}...")
    export_reward_ledger(metrics_path, ledger_csv)
    print("Done.")
else:
    print(f"Error: {metrics_path} not found.")

