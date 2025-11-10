#!/usr/bin/env python3
"""
Analyze wandb run data - learning exercise to see what we can pull after a run.

This script demonstrates how to:
1. Query wandb API for recent runs
2. Extract metrics, config, summary, and history
3. Analyze the data structure
4. Export data for further analysis
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def get_recent_runs(project: str = "dakota-rl-grammar", entity: Optional[str] = None, limit: int = 5) -> List:
    """Get recent runs from wandb."""
    api = wandb.Api()
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=limit)
        else:
            runs = api.runs(project, order="-created_at", per_page=limit)
        return list(runs)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []


def analyze_run_structure(run) -> Dict[str, Any]:
    """Analyze the structure of a wandb run."""
    analysis = {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
        "project": run.project,
        "entity": run.entity if hasattr(run, 'entity') else None,
    }
    
    # Get summary (final metrics)
    print("\n" + "="*80)
    print("SUMMARY (Final Metrics)")
    print("="*80)
    if run.summary:
        summary_dict = dict(run.summary)
        analysis["summary"] = summary_dict
        print(f"Found {len(summary_dict)} summary metrics:")
        for key, value in sorted(summary_dict.items()):
            if not key.startswith("_"):  # Skip internal wandb keys
                print(f"  {key}: {value}")
    else:
        print("No summary data available")
        analysis["summary"] = {}
    
    # Get config (hyperparameters)
    print("\n" + "="*80)
    print("CONFIG (Hyperparameters)")
    print("="*80)
    if run.config:
        config_dict = dict(run.config)
        analysis["config"] = config_dict
        print(f"Found {len(config_dict)} config parameters:")
        for key, value in sorted(config_dict.items()):
            if not key.startswith("_"):  # Skip internal wandb keys
                print(f"  {key}: {value}")
    else:
        print("No config data available")
        analysis["config"] = {}
    
    # Get history (time series metrics)
    print("\n" + "="*80)
    print("HISTORY (Time Series Metrics)")
    print("="*80)
    try:
        import pandas as pd
        history = run.history()
        if not history.empty:
            analysis["history_columns"] = list(history.columns)
            analysis["history_rows"] = len(history)
            analysis["history_sample"] = history.head(5).to_dict('records') if len(history) > 0 else []
            
            print(f"Found {len(history)} logged steps")
            print(f"Metrics tracked: {', '.join([c for c in history.columns if not c.startswith('_')])}")
            
            # Show sample of latest metrics
            if len(history) > 0:
                print("\nLatest 5 steps:")
                latest = history.tail(5)
                for idx, row in latest.iterrows():
                    step = row.get('_step', idx)
                    print(f"\n  Step {step}:")
                    for col in history.columns:
                        if not col.startswith('_') and pd.notna(row[col]):
                            print(f"    {col}: {row[col]}")
        else:
            print("No history data available")
            analysis["history_columns"] = []
            analysis["history_rows"] = 0
    except Exception as e:
        print(f"Error fetching history: {e}")
        analysis["history_error"] = str(e)
    
    # Get files (artifacts, logs, etc.)
    print("\n" + "="*80)
    print("FILES (Artifacts & Logs)")
    print("="*80)
    try:
        files = run.files()
        file_list = [f.name for f in files]
        analysis["files"] = file_list
        print(f"Found {len(file_list)} files:")
        for fname in file_list[:20]:  # Show first 20
            print(f"  - {fname}")
        if len(file_list) > 20:
            print(f"  ... and {len(file_list) - 20} more")
    except Exception as e:
        print(f"Error fetching files: {e}")
        analysis["files_error"] = str(e)
    
    # Get tags
    print("\n" + "="*80)
    print("TAGS")
    print("="*80)
    if hasattr(run, 'tags') and run.tags:
        analysis["tags"] = run.tags
        print(f"Tags: {', '.join(run.tags)}")
    else:
        print("No tags")
        analysis["tags"] = []
    
    return analysis


def export_run_data(run, output_dir: Path):
    """Export run data to JSON files for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export summary
    if run.summary:
        summary_path = output_dir / f"{run.id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(dict(run.summary), f, indent=2, default=str)
        print(f"Exported summary to: {summary_path}")
    
    # Export config
    if run.config:
        config_path = output_dir / f"{run.id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(dict(run.config), f, indent=2, default=str)
        print(f"Exported config to: {config_path}")
    
    # Export history as CSV
    try:
        import pandas as pd
        history = run.history()
        if not history.empty:
            history_path = output_dir / f"{run.id}_history.csv"
            history.to_csv(history_path, index=False)
            print(f"Exported history ({len(history)} rows) to: {history_path}")
    except Exception as e:
        print(f"Could not export history: {e}")


def main():
    """Main analysis function."""
    import pandas as pd
    
    print("="*80)
    print("WANDB RUN DATA ANALYSIS")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get project and entity from environment or use defaults
    project = os.getenv("WANDB_PROJECT", "dakota-rl-grammar")
    entity = os.getenv("WANDB_ENTITY", "christian-cooper-us")
    
    print(f"\nProject: {project}")
    print(f"Entity: {entity}")
    
    # Get recent runs
    print("\n" + "="*80)
    print("FETCHING RECENT RUNS")
    print("="*80)
    runs = get_recent_runs(project=project, entity=entity, limit=5)
    
    if not runs:
        print("No runs found!")
        return
    
    print(f"\nFound {len(runs)} recent run(s):")
    for i, run in enumerate(runs, 1):
        print(f"\n{i}. {run.name} (ID: {run.id})")
        print(f"   State: {run.state}")
        print(f"   Created: {run.created_at}")
        print(f"   URL: {run.url}")
    
    # Analyze the most recent run
    if runs:
        most_recent = runs[0]
        print("\n" + "="*80)
        print(f"ANALYZING MOST RECENT RUN: {most_recent.name}")
        print("="*80)
        
        analysis = analyze_run_structure(most_recent)
        
        # Export data
        print("\n" + "="*80)
        print("EXPORTING DATA")
        print("="*80)
        output_dir = Path("wandb_analysis") / most_recent.id
        export_run_data(most_recent, output_dir)
        
        # Save full analysis
        analysis_path = output_dir / "full_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nFull analysis saved to: {analysis_path}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nView run in browser: {most_recent.url}")
        print(f"Data exported to: {output_dir}")


if __name__ == "__main__":
    main()

