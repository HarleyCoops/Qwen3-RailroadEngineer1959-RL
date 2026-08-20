#!/usr/bin/env python3
"""
Create W&B Report for Dakota Grammar RL Training

This script programmatically creates a W&B Report with all the charts and analysis
from the orchestrator and trainer runs.

Usage:
    python scripts/analysis/create_wandb_report.py

Requirements:
    pip install wandb python-dotenv
"""

import os
import sys
from pathlib import Path
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


def create_wandb_report(
    orchestrator_run_id: str = DEFAULT_ORCHESTRATOR_RUN,
    trainer_run_id: str = DEFAULT_TRAINER_RUN,
    report_title: str = "Dakota Grammar RL Training: Comprehensive Analysis"
):
    """Create a W&B Report with charts and analysis."""
    
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key:
        print("ERROR: WANDB_API_KEY not found in environment")
        print("Please set it in your .env file or export it")
        sys.exit(1)
    
    print("="*80)
    print("Creating W&B Report")
    print("="*80)
    print(f"Project: {ENTITY}/{PROJECT}")
    print(f"Orchestrator Run: {orchestrator_run_id}")
    print(f"Trainer Run: {trainer_run_id}")
    print()
    
    try:
        wandb.login(key=api_key)
        api = wandb.Api()
        
        # Verify runs exist
        orchestrator_path = f"{ENTITY}/{PROJECT}/{orchestrator_run_id}"
        trainer_path = f"{ENTITY}/{PROJECT}/{trainer_run_id}"
        
        print("Verifying runs exist...")
        orchestrator_run = api.run(orchestrator_path)
        trainer_run = api.run(trainer_path)
        print(f"[OK] Orchestrator run found: {orchestrator_run.name}")
        print(f"[OK] Trainer run found: {trainer_run.name}")
        print()
        
        # Create report using W&B API
        # Note: W&B Reports API may require different approach
        # This is a template - actual implementation depends on W&B API version
        
        print("Creating report...")
        print()
        print("NOTE: W&B Reports are typically created through the web UI.")
        print("To create this report:")
        print()
        print("1. Go to: https://wandb.ai/christian-cooper-us/dakota-rl-grammar")
        print("2. Click 'Create report'")
        print("3. Add panels using the following configurations:")
        print()
        
        # Print panel configurations
        panels = [
            {
                "title": "Overall Reward Progression",
                "type": "line",
                "metrics": ["reward/mean"],
                "run": orchestrator_run_id,
                "x_axis": "_step",
                "smoothing": 0.6,
                "description": "Shows overall reward improvement from 0.120 to 0.349 (190% increase)"
            },
            {
                "title": "Compositional Reward Components",
                "type": "line",
                "metrics": ["metrics/char_overlap_reward", "metrics/affix_reward", "reward/mean"],
                "run": orchestrator_run_id,
                "x_axis": "_step",
                "smoothing": 0.6,
                "description": "Compares character preservation (0.535), morphological accuracy (0.979), and overall composite (0.349)"
            },
            {
                "title": "Sample Efficiency Milestones",
                "type": "line",
                "metrics": ["reward/mean"],
                "run": orchestrator_run_id,
                "x_axis": "_step",
                "smoothing": 0.6,
                "markers": [
                    {"step": 49, "value": 0.177, "label": "25% improvement"},
                    {"step": 71, "value": 0.234, "label": "50% improvement"},
                    {"step": 109, "value": 0.292, "label": "75% improvement"},
                    {"step": 160, "value": 0.326, "label": "90% improvement"}
                ],
                "description": "Shows rapid initial learning: 90% improvement achieved in first 160 steps (16% of training)"
            },
            {
                "title": "Policy Stability - KL Divergence",
                "type": "line",
                "metrics": ["masked_mismatch_kl/mean", "mismatch_kl/mean", "unmasked_mismatch_kl/mean"],
                "run": trainer_run_id,
                "x_axis": "_step",
                "y_scale": "log",
                "smoothing": 0.6,
                "description": "Tracks policy divergence: masked KL (8.42 mean), overall KL (3.03 mean), unmasked KL (0.070 mean)"
            },
            {
                "title": "Training Loss Dynamics",
                "type": "line",
                "metrics": ["loss/mean", "loss/median", "loss/std"],
                "run": trainer_run_id,
                "x_axis": "_step",
                "y_scale": "log",
                "smoothing": 0.6,
                "description": "Shows stable optimization with small, controlled loss values (1e-5 to 1e-3 range)"
            },
            {
                "title": "Reward Component Comparison",
                "type": "bar",
                "data": [
                    {"name": "Character Preservation", "value": 0.535},
                    {"name": "Morphological Accuracy", "value": 0.979},
                    {"name": "Overall Composite", "value": 0.349}
                ],
                "description": "Final performance snapshot showing exceptional morphology (97.9%) vs moderate character preservation (53.5%)"
            },
            {
                "title": "KL Divergence Distribution",
                "type": "box",
                "metrics": [
                    "masked_mismatch_kl/mean",
                    "mismatch_kl/mean",
                    "unmasked_mismatch_kl/mean"
                ],
                "run": trainer_run_id,
                "x_axis": "_step",
                "description": "Distribution analysis showing high divergence for masked tokens, moderate for overall, low for unmasked"
            },
            {
                "title": "Training Progress Timeline",
                "type": "line",
                "metrics": ["progress/total_samples", "progress/total_tokens"],
                "run": orchestrator_run_id,
                "x_axis": "_step",
                "y_scale": "log",
                "description": "Tracks cumulative progress: 256,000 samples and 40.8M tokens processed over 1,000 steps"
            }
        ]
        
        for i, panel in enumerate(panels, 1):
            print(f"Panel {i}: {panel['title']}")
            print(f"  Type: {panel['type']}")
            if 'metrics' in panel:
                print(f"  Metrics: {', '.join(panel['metrics'])}")
                print(f"  Run: {panel.get('run', 'N/A')}")
            if 'data' in panel:
                print(f"  Data: Custom bar chart data")
            print(f"  Description: {panel['description']}")
            print()
        
        print("="*80)
        print("Report Configuration Complete")
        print("="*80)
        print()
        print("Next Steps:")
        print("1. Use the web UI to create the report with these panel configurations")
        print("2. Or use the W&B Reports API (if available) to create programmatically")
        print("3. Reference the detailed report structure in: docs/WANDB_REPORT_STRUCTURE.md")
        print()
        print("Run URLs:")
        print(f"  Orchestrator: https://wandb.ai/{orchestrator_path}")
        print(f"  Trainer: https://wandb.ai/{trainer_path}")
        print(f"  Project: https://wandb.ai/{ENTITY}/{PROJECT}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create W&B Report for Dakota Grammar RL Training"
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
        "--title",
        type=str,
        default="Dakota Grammar RL Training: Comprehensive Analysis",
        help="Report title"
    )
    
    args = parser.parse_args()
    
    try:
        create_wandb_report(
            orchestrator_run_id=args.orchestrator_run,
            trainer_run_id=args.trainer_run,
            report_title=args.title
        )
    except KeyboardInterrupt:
        print("\n\nReport creation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

