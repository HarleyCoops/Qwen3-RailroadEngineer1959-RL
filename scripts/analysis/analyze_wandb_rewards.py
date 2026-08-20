#!/usr/bin/env python3
"""
Comprehensive W&B Data Analysis Script for Dakota RL Training

Pulls data from:
- Orchestrator run (29hn8w98) - Contains reward data
- Trainer run (7nikv4vp) - Contains training metrics

Usage:
    python scripts/analysis/analyze_wandb_rewards.py
    
    # Analyze specific runs
    python scripts/analysis/analyze_wandb_rewards.py --orchestrator-run 29hn8w98 --trainer-run 7nikv4vp
    
    # Use local data only
    python scripts/analysis/analyze_wandb_rewards.py --local-only
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed")
    print("Install with: pip install wandb")
    sys.exit(1)

# Default run IDs
DEFAULT_ORCHESTRATOR_RUN = "29hn8w98"
DEFAULT_TRAINER_RUN = "7nikv4vp"
PROJECT = "dakota-rl-grammar"
ENTITY = "christian-cooper-us"


def load_local_data(run_id: str, data_dir: Path = Path("wandb_analysis")) -> Optional[Dict]:
    """Load data from local files if available."""
    run_dir = data_dir / run_id
    json_file = run_dir / f"{run_id}_summary.json"
    csv_file = run_dir / f"{run_id}_history.csv"
    
    if json_file.exists() and csv_file.exists():
        print(f"  Found local data for {run_id}")
        with open(json_file, 'r') as f:
            summary = json.load(f)
        history = pd.read_csv(csv_file)
        return {
            "summary": summary,
            "history": history,
            "run_id": run_id
        }
    return None


def fetch_wandb_run(run_id: str, entity: str = ENTITY, project: str = PROJECT) -> Optional[Dict]:
    """Fetch run data from W&B API."""
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key:
        print(f"  WARNING: No WANDB_API_KEY found, cannot fetch {run_id} from API")
        return None
    
    try:
        wandb.login(key=api_key)
        api = wandb.Api()
        run_path = f"{entity}/{project}/{run_id}"
        print(f"  Fetching {run_path} from W&B...")
        run = api.run(run_path)
        
        # Get history
        history = run.history(samples=50000)  # Get all available
        
        return {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": dict(run.summary),
            "config": dict(run.config),
            "history": history,
            "history_columns": history.columns.tolist() if not history.empty else [],
        }
    except Exception as e:
        print(f"  ERROR fetching {run_id}: {e}")
        return None


def analyze_reward_components(orchestrator_data: Dict) -> Dict[str, Any]:
    """Analyze reward components from orchestrator run."""
    if not orchestrator_data or "history" not in orchestrator_data:
        return {"error": "No orchestrator data available"}
    
    history = orchestrator_data["history"]
    if history.empty:
        return {"error": "Empty history data"}
    
    analysis = {
        "total_steps": len(history),
        "reward_columns": [],
        "reward_statistics": {},
        "reward_trends": {},
    }
    
    # Find all reward-related columns
    reward_cols = [col for col in history.columns if 'reward' in col.lower() or 'score' in col.lower()]
    analysis["reward_columns"] = reward_cols
    
    # Analyze each reward metric
    for col in reward_cols:
        if col in history.columns:
            values = history[col].dropna()
            if len(values) > 0:
                analysis["reward_statistics"][col] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "final": float(values.iloc[-1]) if len(values) > 0 else None,
                    "samples": len(values),
                }
                
                # Calculate trend (first half vs second half)
                if len(values) > 10:
                    mid = len(values) // 2
                    first_half = values[:mid].mean()
                    second_half = values[mid:].mean()
                    analysis["reward_trends"][col] = {
                        "first_half_mean": float(first_half),
                        "second_half_mean": float(second_half),
                        "improvement": float(second_half - first_half),
                        "improvement_pct": float((second_half - first_half) / abs(first_half) * 100) if first_half != 0 else 0,
                    }
    
    return analysis


def analyze_training_metrics(trainer_data: Dict) -> Dict[str, Any]:
    """Analyze training metrics from trainer run."""
    if not trainer_data or "history" not in trainer_data:
        return {"error": "No trainer data available"}
    
    history = trainer_data["history"]
    if history.empty:
        return {"error": "Empty history data"}
    
    analysis = {
        "total_steps": len(history),
        "key_metrics": {},
        "final_values": {},
    }
    
    # Key metrics to analyze
    key_metrics = [
        "loss/mean", "loss/std",
        "entropy/mean", "entropy/median",
        "inference_probs/mean", "trainer_probs/mean",
        "perf/throughput", "perf/mfu",
        "optim/lr", "optim/grad_norm",
    ]
    
    for metric in key_metrics:
        if metric in history.columns:
            values = history[metric].dropna()
            if len(values) > 0:
                analysis["key_metrics"][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "final": float(values.iloc[-1]),
                    "trend": "improving" if len(values) > 10 and values.iloc[-1] < values.iloc[0] else "degrading",
                }
                analysis["final_values"][metric] = float(values.iloc[-1])
    
    return analysis


def create_reward_summary(orchestrator_data: Dict, trainer_data: Dict, output_dir: Path):
    """Create comprehensive summary of rewards and training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze rewards
    reward_analysis = analyze_reward_components(orchestrator_data)
    training_analysis = analyze_training_metrics(trainer_data)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "orchestrator_run": {
            "run_id": orchestrator_data.get("run_id", "unknown"),
            "name": orchestrator_data.get("name", "unknown"),
            "url": orchestrator_data.get("url", ""),
            "state": orchestrator_data.get("state", "unknown"),
        },
        "trainer_run": {
            "run_id": trainer_data.get("run_id", "unknown"),
            "name": trainer_data.get("name", "unknown"),
            "url": trainer_data.get("url", ""),
            "state": trainer_data.get("state", "unknown"),
        },
        "reward_analysis": reward_analysis,
        "training_analysis": training_analysis,
    }
    
    # Save summary
    summary_file = output_dir / "reward_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_file}")
    
    # Save detailed reward data
    if orchestrator_data and "history" in orchestrator_data:
        reward_cols = [col for col in orchestrator_data["history"].columns 
                      if 'reward' in col.lower() or 'score' in col.lower()]
        if reward_cols:
            reward_df = orchestrator_data["history"][["_step"] + reward_cols]
            reward_csv = output_dir / "orchestrator_rewards.csv"
            reward_df.to_csv(reward_csv, index=False)
            print(f"Reward data saved to: {reward_csv}")
    
    return summary


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze W&B reward and training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--orchestrator-run",
        type=str,
        default=DEFAULT_ORCHESTRATOR_RUN,
        help=f"Orchestrator run ID (default: {DEFAULT_ORCHESTRATOR_RUN})"
    )
    parser.add_argument(
        "--trainer-run",
        type=str,
        default=DEFAULT_TRAINER_RUN,
        help=f"Trainer run ID (default: {DEFAULT_TRAINER_RUN})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wandb_analysis",
        help="Output directory for analysis files"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only use local data, don't fetch from W&B API"
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetch from W&B even if local data exists"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("W&B Reward & Training Data Analysis")
    print("="*80)
    print(f"Orchestrator Run: {args.orchestrator_run}")
    print(f"Trainer Run: {args.trainer_run}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Load orchestrator data
    print("Loading Orchestrator Data...")
    orchestrator_data = None
    
    if not args.force_fetch:
        orchestrator_data = load_local_data(args.orchestrator_run)
    
    if not orchestrator_data and not args.local_only:
        orchestrator_data = fetch_wandb_run(args.orchestrator_run)
        # Save fetched data locally
        if orchestrator_data:
            run_dir = Path(args.output_dir) / args.orchestrator_run
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / f"{args.orchestrator_run}_summary.json", 'w') as f:
                json.dump(orchestrator_data.get("summary", {}), f, indent=2, default=str)
            if "history" in orchestrator_data and not orchestrator_data["history"].empty:
                orchestrator_data["history"].to_csv(
                    run_dir / f"{args.orchestrator_run}_history.csv", 
                    index=False
                )
    
    if orchestrator_data:
        print(f"  [OK] Loaded orchestrator data ({len(orchestrator_data.get('history', pd.DataFrame()))} steps)")
    else:
        print(f"  [ERROR] Could not load orchestrator data")
    
    # Load trainer data
    print("\nLoading Trainer Data...")
    trainer_data = None
    
    if not args.force_fetch:
        trainer_data = load_local_data(args.trainer_run)
    
    if not trainer_data and not args.local_only:
        trainer_data = fetch_wandb_run(args.trainer_run)
        # Save fetched data locally
        if trainer_data:
            run_dir = Path(args.output_dir) / args.trainer_run
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / f"{args.trainer_run}_summary.json", 'w') as f:
                json.dump(trainer_data.get("summary", {}), f, indent=2, default=str)
            if "history" in trainer_data and not trainer_data["history"].empty:
                trainer_data["history"].to_csv(
                    run_dir / f"{args.trainer_run}_history.csv", 
                    index=False
                )
    
    if trainer_data:
        print(f"  [OK] Loaded trainer data ({len(trainer_data.get('history', pd.DataFrame()))} steps)")
    else:
        print(f"  [ERROR] Could not load trainer data")
    
    if not orchestrator_data and not trainer_data:
        print("\nERROR: No data available to analyze")
        print("  - Check that WANDB_API_KEY is set in .env file")
        print("  - Or ensure local data exists in wandb_analysis/")
        sys.exit(1)
    
    # Analyze and create summary
    print("\n" + "="*80)
    print("Analyzing Data...")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    summary = create_reward_summary(orchestrator_data or {}, trainer_data or {}, output_dir)
    
    # Print key findings
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)
    
    if "reward_analysis" in summary and "reward_columns" in summary["reward_analysis"]:
        reward_cols = summary["reward_analysis"]["reward_columns"]
        print(f"\nReward Metrics Found: {len(reward_cols)}")
        for col in reward_cols[:10]:
            print(f"  - {col}")
        if len(reward_cols) > 10:
            print(f"  ... and {len(reward_cols) - 10} more")
        
        if "reward_statistics" in summary["reward_analysis"]:
            print("\nReward Statistics:")
            for col, stats in list(summary["reward_analysis"]["reward_statistics"].items())[:5]:
                print(f"  {col}:")
                print(f"    Mean: {stats['mean']:.4f}, Final: {stats['final']:.4f}")
    
    if "training_analysis" in summary and "final_values" in summary["training_analysis"]:
        print("\nFinal Training Metrics:")
        for metric, value in list(summary["training_analysis"]["final_values"].items())[:5]:
            print(f"  {metric}: {value:.6f}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nView detailed results in: {output_dir}/reward_analysis_summary.json")


if __name__ == "__main__":
    main()

