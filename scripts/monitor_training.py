#!/usr/bin/env python3
"""
Monitor active GRPO training runs for Dakota language model.

This script checks:
- Active W&B runs
- Training metrics (rewards, loss, etc.)
- System resources
- Process status
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("WARNING: psutil not installed. Install with: pip install psutil")
    psutil = None


def get_wandb_runs(project="grammar-gym", state="running"):
    """Get active W&B runs."""
    api = wandb.Api()
    try:
        runs = api.runs(f"{project}", filters={"state": state})
        return list(runs)
    except Exception as e:
        print(f"Error fetching W&B runs: {e}")
        return []


def format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def get_process_info():
    """Get information about Python processes."""
    if psutil is None:
        return []
    
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                runtime = time.time() - proc.info['create_time']
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                    'runtime': runtime
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return processes


def print_run_summary(run):
    """Print summary of a W&B run."""
    print(f"\n{'='*80}")
    print(f"Run: {run.name}")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")
    
    if run.state == "running":
        try:
            # W&B returns datetime strings in UTC
            if isinstance(run.created_at, str):
                created_str = run.created_at.replace('Z', '')
                created = datetime.fromisoformat(created_str)
            else:
                created = run.created_at
            # Get current UTC time for comparison
            now_utc = datetime.utcnow()
            if hasattr(created, 'replace'):
                created = created.replace(tzinfo=None)
            runtime = (now_utc - created).total_seconds()
            if runtime > 0:
                print(f"Runtime: {format_duration(runtime)}")
            else:
                print(f"Runtime: Just started")
        except Exception as e:
            print(f"Runtime: Could not calculate ({e})")
    
    # Get latest metrics
    if run.state == "running":
        try:
            # Try to get the latest step metrics
            history = run.scan_history(keys=[
                "reward/mean",
                "reward/std", 
                "loss/mean",
                "batch/solve_all",
                "batch/solve_none",
                "step"
            ])
            
            # Get the last non-null values
            last_metrics = {}
            for row in history:
                for key in ["reward/mean", "reward/std", "loss/mean", "batch/solve_all", "batch/solve_none", "step"]:
                    if key in row and row[key] is not None:
                        last_metrics[key] = row[key]
            
            if last_metrics:
                print(f"\nLatest Metrics:")
                print(f"  Step: {last_metrics.get('step', 'N/A')}")
                if 'reward/mean' in last_metrics:
                    print(f"  Reward Mean: {last_metrics['reward/mean']:.4f}")
                if 'reward/std' in last_metrics:
                    print(f"  Reward Std: {last_metrics['reward/std']:.4f}")
                if 'loss/mean' in last_metrics:
                    print(f"  Loss Mean: {last_metrics['loss/mean']:.4f}")
                if 'batch/solve_all' in last_metrics:
                    print(f"  Solve All: {last_metrics['batch/solve_all']}")
                if 'batch/solve_none' in last_metrics:
                    print(f"  Solve None: {last_metrics['batch/solve_none']}")
            else:
                print(f"\n  No metrics available yet (run may be initializing)")
        except Exception as e:
            print(f"  Could not fetch metrics: {e}")
    
    print(f"URL: {run.url}")


def main():
    """Main monitoring function."""
    print("="*80)
    print("Dakota GRPO Training Monitor")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check W&B runs
    print("\n" + "="*80)
    print("ACTIVE W&B RUNS")
    print("="*80)
    
    runs = get_wandb_runs(project="grammar-gym", state="running")
    
    if not runs:
        print("No active runs found.")
    else:
        print(f"Found {len(runs)} active run(s):")
        for run in runs:
            print_run_summary(run)
    
    # Check failed runs
    print("\n" + "="*80)
    print("RECENT FAILED RUNS")
    print("="*80)
    
    failed_runs = get_wandb_runs(project="grammar-gym", state="failed")
    if failed_runs:
        # Show only recent failures (last 24 hours)
        recent_failed = []
        for r in failed_runs:
            try:
                if isinstance(r.created_at, str):
                    created = datetime.fromisoformat(r.created_at.replace('Z', '+00:00'))
                else:
                    created = r.created_at
                if hasattr(created, 'replace'):
                    created = created.replace(tzinfo=None)
                if (datetime.now() - created).total_seconds() < 86400:
                    recent_failed.append(r)
            except:
                continue
        if recent_failed:
            print(f"Found {len(recent_failed)} recent failed run(s):")
            for run in recent_failed[:5]:  # Show up to 5
                print(f"\n  - {run.name}")
                print(f"    Created: {run.created_at}")
                print(f"    URL: {run.url}")
        else:
            print("No recent failures.")
    else:
        print("No failed runs found.")
    
    # Check system resources
    if psutil:
        print("\n" + "="*80)
        print("SYSTEM RESOURCES")
        print("="*80)
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(f"Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
        
        # Check Python processes
        processes = get_process_info()
        if processes:
            print(f"\nPython Processes ({len(processes)}):")
            print(f"{'PID':<8} {'CPU%':<8} {'Memory(MB)':<12} {'Runtime':<12}")
            print("-" * 50)
            for proc in sorted(processes, key=lambda x: x['memory_mb'], reverse=True)[:10]:
                print(f"{proc['pid']:<8} {proc['cpu_percent']:<8.1f} {proc['memory_mb']:<12.1f} {format_duration(proc['runtime']):<12}")
    
    print("\n" + "="*80)
    print("Monitoring complete. Run again to refresh.")
    print("="*80)


if __name__ == "__main__":
    main()

