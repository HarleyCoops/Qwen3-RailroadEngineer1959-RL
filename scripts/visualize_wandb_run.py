#!/usr/bin/env python3
"""
Visualize and analyze wandb run data - demonstrates what insights we can extract.

This script shows:
1. Training curves (loss, rewards, etc.)
2. Performance metrics over time
3. Statistical summaries
4. Key insights from the training run
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def load_run_data(run_id: str, project: str = "dakota-rl-grammar", entity: str = "christian-cooper-us"):
    """Load run data from wandb."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run


def analyze_training_metrics(run):
    """Analyze key training metrics."""
    print("\n" + "="*80)
    print("TRAINING METRICS ANALYSIS")
    print("="*80)
    
    history = run.history()
    if history.empty:
        print("No history data available")
        return None
    
    # Key metrics to analyze
    metrics = {
        'loss': ['loss/mean', 'loss/std', 'loss/min', 'loss/max'],
        'entropy': ['entropy/mean', 'entropy/std'],
        'performance': ['perf/throughput', 'perf/mfu', 'perf/peak_memory'],
        'optimization': ['optim/lr', 'optim/grad_norm'],
        'probabilities': ['inference_probs/mean', 'trainer_probs/mean'],
        'kl_divergence': ['mismatch_kl/mean', 'masked_mismatch_kl/mean']
    }
    
    analysis = {}
    
    for category, metric_keys in metrics.items():
        print(f"\n{category.upper()}:")
        analysis[category] = {}
        
        for key in metric_keys:
            if key in history.columns:
                values = history[key].dropna()
                if len(values) > 0:
                    analysis[category][key] = {
                        'initial': float(values.iloc[0]) if len(values) > 0 else None,
                        'final': float(values.iloc[-1]) if len(values) > 0 else None,
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'trend': 'increasing' if len(values) > 1 and values.iloc[-1] > values.iloc[0] else 'decreasing' if len(values) > 1 else 'stable'
                    }
                    
                    print(f"  {key}:")
                    print(f"    Initial: {analysis[category][key]['initial']:.6f}")
                    print(f"    Final: {analysis[category][key]['final']:.6f}")
                    print(f"    Mean: {analysis[category][key]['mean']:.6f}")
                    print(f"    Trend: {analysis[category][key]['trend']}")
    
    return analysis, history


def create_visualizations(history, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if history.empty:
        print("No data to visualize")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (15, 10)
    
    # 1. Loss over time
    if 'loss/mean' in history.columns:
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
        
        # Loss
        ax = axes[0, 0]
        steps = history['step'] if 'step' in history.columns else history.index
        ax.plot(steps, history['loss/mean'], label='Mean Loss', linewidth=2)
        if 'loss/std' in history.columns:
            ax.fill_between(steps, 
                          history['loss/mean'] - history['loss/std'],
                          history['loss/mean'] + history['loss/std'],
                          alpha=0.3, label='±1 Std')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[0, 1]
        if 'entropy/mean' in history.columns:
            ax.plot(steps, history['entropy/mean'], label='Mean Entropy', color='green', linewidth=2)
            if 'entropy/std' in history.columns:
                ax.fill_between(steps,
                              history['entropy/mean'] - history['entropy/std'],
                              history['entropy/mean'] + history['entropy/std'],
                              alpha=0.3, color='green')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Throughput
        ax = axes[1, 0]
        if 'perf/throughput' in history.columns:
            ax.plot(steps, history['perf/throughput'], label='Throughput', color='orange', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Tokens/sec')
        ax.set_title('Training Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Probabilities
        ax = axes[1, 1]
        if 'inference_probs/mean' in history.columns:
            ax.plot(steps, history['inference_probs/mean'], label='Inference Probs', linewidth=2)
        if 'trainer_probs/mean' in history.columns:
            ax.plot(steps, history['trainer_probs/mean'], label='Trainer Probs', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Probability')
        ax.set_title('Model Probabilities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plot_path = output_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training metrics plot to: {plot_path}")
        plt.close()
    
    # 2. Performance metrics
    perf_metrics = ['perf/mfu', 'perf/peak_memory']
    if any(m in history.columns for m in perf_metrics):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
        
        steps = history['step'] if 'step' in history.columns else history.index
        
        if 'perf/mfu' in history.columns:
            ax = axes[0]
            ax.plot(steps, history['perf/mfu'], label='MFU (%)', color='purple', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Model FLOPS Utilization (%)')
            ax.set_title('MFU Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        if 'perf/peak_memory' in history.columns:
            ax = axes[1]
            ax.plot(steps, history['perf/peak_memory'], label='Peak Memory', color='red', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Memory (GiB)')
            ax.set_title('Peak Memory Usage')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'performance_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance metrics plot to: {plot_path}")
        plt.close()


def generate_insights(run, analysis, history):
    """Generate key insights from the run."""
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Training completion
    if run.state == 'finished':
        insights.append("✓ Training completed successfully")
    elif run.state == 'crashed':
        insights.append("⚠ Training crashed")
    elif run.state == 'failed':
        insights.append("✗ Training failed")
    
    # Training duration
    if hasattr(run, 'summary') and '_runtime' in run.summary:
        runtime = run.summary['_runtime']
        hours = runtime / 3600
        insights.append(f"⏱ Training duration: {hours:.2f} hours ({runtime:.0f} seconds)")
    
    # Steps completed
    if 'step' in history.columns:
        max_step = history['step'].max()
        insights.append(f"📊 Steps completed: {max_step:.0f}")
    
    # Loss trends
    if analysis and 'loss' in analysis:
        if 'loss/mean' in analysis['loss']:
            loss_trend = analysis['loss']['loss/mean']['trend']
            initial_loss = analysis['loss']['loss/mean']['initial']
            final_loss = analysis['loss']['loss/mean']['final']
            if loss_trend == 'decreasing':
                insights.append(f"📉 Loss decreased from {initial_loss:.6f} to {final_loss:.6f}")
            elif loss_trend == 'increasing':
                insights.append(f"📈 Loss increased from {initial_loss:.6f} to {final_loss:.6f} (may indicate overfitting)")
    
    # Performance
    if analysis and 'performance' in analysis:
        if 'perf/throughput' in analysis['performance']:
            throughput = analysis['performance']['perf/throughput']['mean']
            insights.append(f"⚡ Average throughput: {throughput:.0f} tokens/sec")
        
        if 'perf/mfu' in analysis['performance']:
            mfu = analysis['performance']['perf/mfu']['mean']
            insights.append(f"🔧 Average MFU: {mfu:.2f}%")
    
    # Entropy (model confidence)
    if analysis and 'entropy' in analysis:
        if 'entropy/mean' in analysis['entropy']:
            entropy = analysis['entropy']['entropy/mean']['final']
            if entropy < 0.5:
                insights.append(f"🎯 Low entropy ({entropy:.4f}) - model is confident")
            else:
                insights.append(f"🤔 Higher entropy ({entropy:.4f}) - model is more uncertain")
    
    # Print insights
    for insight in insights:
        print(f"  {insight}")
    
    return insights


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze wandb run data")
    parser.add_argument("--run-id", type=str, default="7nikv4vp", help="Wandb run ID")
    parser.add_argument("--project", type=str, default="dakota-rl-grammar", help="Wandb project")
    parser.add_argument("--entity", type=str, default="christian-cooper-us", help="Wandb entity")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("="*80)
    print("WANDB RUN DETAILED ANALYSIS")
    print("="*80)
    print(f"Run ID: {args.run_id}")
    print(f"Project: {args.project}")
    print(f"Entity: {args.entity}")
    
    # Load run
    print("\nLoading run data from wandb...")
    run = load_run_data(args.run_id, args.project, args.entity)
    
    print(f"Run Name: {run.name}")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    
    # Analyze metrics
    analysis, history = analyze_training_metrics(run)
    
    # Generate insights
    insights = generate_insights(run, analysis, history)
    
    # Create visualizations
    if not args.no_plots:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        try:
            output_dir = Path("wandb_analysis") / args.run_id / "plots"
            create_visualizations(history, output_dir)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("(This might require matplotlib. Install with: pip install matplotlib)")
    
    # Save analysis
    output_dir = Path("wandb_analysis") / args.run_id
    analysis_path = output_dir / "detailed_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'run_id': args.run_id,
            'run_name': run.name,
            'state': run.state,
            'url': run.url,
            'analysis': analysis,
            'insights': insights
        }, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nView run in browser: {run.url}")
    print(f"Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()



