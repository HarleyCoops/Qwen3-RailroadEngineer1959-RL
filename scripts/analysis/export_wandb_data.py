#!/usr/bin/env python3
"""
W&B Data Export Script for Dakota Grammar RL Run

Run this script on your local machine where you have access to W&B

Usage:
1. Install dependencies: pip install wandb pandas python-dotenv
2. Set WANDB_API_KEY in your .env file
3. Run: python scripts/analysis/export_wandb_data.py
4. This will create 'dakota_rl_wandb_export.json' with all your run data
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

try:
    import wandb
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install wandb pandas python-dotenv")
    sys.exit(1)


def export_wandb_run(
    run_path: str = "christian-cooper-us/dakota-rl-grammar/29hn8w98",
    output_dir: str = "wandb_analysis",
    max_samples: int = 10000,
):
    """Export W&B run data to JSON for visualization
    
    Args:
        run_path: W&B run path in format "entity/project/run_id"
        output_dir: Directory to save exported files
        max_samples: Maximum number of history samples to fetch
    """
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WANDB_API_KEY not found in environment")
        print("Please set it in your .env file or export it:")
        print("  export WANDB_API_KEY=your_key_here")
        print("\nGet your API key from: https://wandb.ai/authorize")
        sys.exit(1)
    
    print("="*70)
    print("W&B Data Export Script")
    print("="*70)
    print(f"Run path: {run_path}")
    print()
    
    # Initialize W&B API
    print("Initializing W&B API...")
    try:
        wandb.login(key=api_key)
        api = wandb.Api()
        print("✓ W&B API initialized")
    except Exception as e:
        print(f"ERROR: Failed to login to W&B: {e}")
        print("Check your WANDB_API_KEY in .env file")
        sys.exit(1)
    
    # Fetch the specific run
    print(f"\nFetching run data from: {run_path}...")
    try:
        run = api.run(run_path)
        print(f"✓ Run found: {run.name}")
    except Exception as e:
        print(f"ERROR: Failed to fetch run: {e}")
        print(f"Check that the run path is correct: {run_path}")
        sys.exit(1)
    
    # Display run info
    print(f"\nRun Information:")
    print(f"  Name: {run.name}")
    print(f"  ID: {run.id}")
    print(f"  State: {run.state}")
    print(f"  Created: {run.created_at}")
    print(f"  URL: {run.url}")
    
    # Get all history
    print(f"\nFetching complete history (up to {max_samples} samples)...")
    try:
        history = run.history(samples=max_samples)
        print(f"✓ Retrieved {len(history)} data points")
    except Exception as e:
        print(f"ERROR: Failed to fetch history: {e}")
        sys.exit(1)
    
    if len(history) == 0:
        print("WARNING: No history data found for this run")
        return None
    
    # Get summary metrics
    summary = dict(run.summary)
    
    # Get config
    config = dict(run.config)
    
    # Get available metrics
    try:
        available_metrics = run.history_keys
    except:
        available_metrics = list(history.columns)
    
    # Prepare export data
    export_data = {
        "run_info": {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "project": run.project,
            "entity": run.entity,
        },
        "config": config,
        "summary": summary,
        "available_metrics": available_metrics,
        "history": history.to_dict(orient='records'),
        "history_columns": history.columns.tolist(),
        "history_shape": {
            "rows": history.shape[0],
            "columns": history.shape[1]
        },
        "statistics": history.describe().to_dict(),
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    json_file = output_path / "dakota_rl_wandb_export.json"
    print(f"\nSaving to {json_file}...")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"✓ JSON export saved")
    except Exception as e:
        print(f"ERROR: Failed to save JSON: {e}")
        sys.exit(1)
    
    # Also save history as CSV for easier inspection
    csv_file = output_path / "dakota_rl_wandb_history.csv"
    print(f"Saving CSV to {csv_file}...")
    try:
        history.to_csv(csv_file, index=False)
        print(f"✓ CSV export saved")
    except Exception as e:
        print(f"WARNING: Failed to save CSV: {e}")
    
    print(f"\n{'='*70}")
    print("✅ Export complete!")
    print(f"{'='*70}")
    print(f"\nFiles created:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    
    # Print available metrics for reference
    print(f"\nAvailable metrics in history ({len(history.columns)} total):")
    metric_categories = {}
    for col in sorted(history.columns):
        if not col.startswith('_'):
            # Group by prefix
            prefix = col.split('/')[0] if '/' in col else 'other'
            if prefix not in metric_categories:
                metric_categories[prefix] = []
            metric_categories[prefix].append(col)
    
    for category, metrics in sorted(metric_categories.items()):
        print(f"\n  {category} ({len(metrics)} metrics):")
        for metric in metrics[:10]:  # Show first 10
            print(f"    - {metric}")
        if len(metrics) > 10:
            print(f"    ... and {len(metrics) - 10} more")
    
    # Show some key final metrics
    print(f"\n{'='*70}")
    print("Key Final Metrics:")
    print(f"{'='*70}")
    
    key_metrics = [
        'entropy/mean', 'entropy/median', 'entropy/std', 'entropy/max',
        'loss/mean', 'loss/median', 'loss/std',
        'reward', 'policy_loss', 'value_loss',
        'tokens_per_second', 'gpu_memory_gb',
        'inference_probs/mean', 'trainer_probs/mean',
    ]
    
    found_metrics = []
    for metric in key_metrics:
        # Try exact match first
        if metric in history.columns:
            final_value = history[metric].dropna()
            if len(final_value) > 0:
                val = final_value.iloc[-1]
                found_metrics.append((metric, val))
        else:
            # Try partial match
            matching = [col for col in history.columns if metric.split('/')[0] in col]
            if matching:
                for col in matching[:1]:  # Just show first match
                    final_value = history[col].dropna()
                    if len(final_value) > 0:
                        val = final_value.iloc[-1]
                        found_metrics.append((col, val))
    
    if found_metrics:
        for metric, value in found_metrics:
            if isinstance(value, (int, float)):
                print(f"  {metric:30s}: {value:.6f}")
            else:
                print(f"  {metric:30s}: {value}")
    else:
        print("  (No key metrics found - check available metrics above)")
    
    print(f"\n{'='*70}")
    print("📊 Data export complete! You can now analyze the exported files.")
    print(f"{'='*70}")
    
    return export_data


def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export W&B run data to JSON/CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default run
  python scripts/analysis/export_wandb_data.py
  
  # Export specific run
  python scripts/analysis/export_wandb_data.py --run-path "entity/project/run_id"
  
  # Export with more samples
  python scripts/analysis/export_wandb_data.py --max-samples 20000
        """
    )
    
    parser.add_argument(
        "--run-path",
        type=str,
        default="christian-cooper-us/dakota-rl-grammar/29hn8w98",
        help="W&B run path in format 'entity/project/run_id'"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wandb_analysis",
        help="Directory to save exported files"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of history samples to fetch"
    )
    
    args = parser.parse_args()
    
    try:
        export_wandb_run(
            run_path=args.run_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    except KeyboardInterrupt:
        print("\n\nExport cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

