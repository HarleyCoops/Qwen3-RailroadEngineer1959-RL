#!/usr/bin/env python3
"""
Create comprehensive visualizations for Dakota RL training runs on Thinking Machines.
Generates beautiful plots specifically for Tinker-based runs (no Orchestrator/Trainer split).
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

# Set up beautiful plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palette
COLORS = {
    'reward': '#2E86AB',      # Blue
    'loss': '#A23B72',        # Purple
    'entropy': '#F18F01',     # Orange
    'kl': '#C73E1D',          # Red
    'throughput': '#6A994E',  # Green
    'morphology': '#06A77D',   # Teal
    'character': '#F77F00',    # Dark orange
    'composite': '#2E86AB',   # Blue
}


def get_history_series(history, candidates):
    """Return the first available metric series from history."""
    for key in candidates:
        if key in history.columns:
            return history[key], key
    return None, None


def load_run_data(run_id: str, project: str = "dakota-rl-grammar", entity: str = "christian-cooper-us"):
    """Load run data from wandb."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run


def create_reward_progression_plot(run, output_dir: Path):
    """Create beautiful reward progression visualization for Tinker run."""
    history = run.history()
    reward_series, reward_key = get_history_series(
        history,
        [
            "env/all/ledger/composite_predicted", # Primary target for Tinker runs
            "reward/mean",
            "env/all/reward/total",
        ],
    )
    if history.empty or reward_series is None:
        print(f"No reward data available. Keys found: {history.columns.tolist()}")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Reward Progression - Dakota RL Training (Thinking Machines)', fontsize=18, fontweight='bold', y=0.98)
    
    steps = history['_step'] if '_step' in history.columns else history.index
    
    # Main reward curve
    ax = axes[0]
    ax.plot(steps, reward_series, 
            color=COLORS['reward'], linewidth=2.5, label='Composite Reward', zorder=3)
    
    # Add shaded improvement regions
    initial_reward = reward_series.iloc[0]
    final_reward = reward_series.iloc[-1]
    # Check for NaN and handle gracefully
    if pd.isna(initial_reward): initial_reward = 0.0
    if pd.isna(final_reward): final_reward = 0.0
        
    improvement = final_reward - initial_reward
    improvement_pct = ((improvement / initial_reward) * 100) if initial_reward != 0 else 0
    
    ax.axhspan(initial_reward, final_reward, alpha=0.1, color=COLORS['reward'], zorder=1)
    ax.fill_between(steps, initial_reward, reward_series, 
                    alpha=0.2, color=COLORS['reward'], zorder=2)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Reward: {initial_reward:.3f} → {final_reward:.3f} ({improvement_pct:.1f}% improvement)', 
                 fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, steps.max()])
    
    # Component breakdown
    ax2 = axes[1]
    component_metrics = [
        ('Morphological Accuracy', ['env/all/ledger/affix_norm', 'env/all/ledger/affix_raw']),
        ('Character Preservation', ['env/all/ledger/char_overlap_norm', 'env/all/ledger/char_overlap_raw']),
        ('Overall Composite', ['env/all/ledger/composite_predicted', 'env/all/ledger/composite_with_penalty']),
    ]
    
    for label, metric_keys in component_metrics:
        series, _ = get_history_series(history, metric_keys)
        if series is not None:
            color_map = {
                'Morphological Accuracy': COLORS['morphology'],
                'Character Preservation': COLORS['character'],
                'Overall Composite': COLORS['composite'],
            }
            ax2.plot(steps, series, 
                    label=label, linewidth=2.5, color=color_map[label], alpha=0.8)
    
    ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward Value', fontsize=12, fontweight='bold')
    ax2.set_title('Reward Component Breakdown', fontsize=13, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.set_xlim([0, steps.max()])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = output_dir / 'reward_progression.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved reward progression plot: {plot_path}")
    plt.close()


def create_training_metrics_plot(run, output_dir: Path):
    """Create comprehensive training metrics visualization for Tinker run."""
    history = run.history()
    if history.empty:
        print("No training data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics - Dakota RL (Thinking Machines)', fontsize=18, fontweight='bold', y=0.98)
    
    steps = history['_step'] if '_step' in history.columns else history.index
    
    # Loss plot
    ax = axes[0, 0]
    # Check specifically for keys that might be present in a Tinker run
    loss_key = 'loss' # Tinker generic
    if 'loss/mean' in history.columns: loss_key = 'loss/mean'
    elif 'train/loss' in history.columns: loss_key = 'train/loss'
    
    if loss_key in history.columns:
        ax.plot(steps, history[loss_key], 
               color=COLORS['loss'], linewidth=2, label='Loss', zorder=3)
        ax.set_yscale('log')
    else:
        print(f"Warning: No loss metric found. Candidates: {history.columns.tolist()[:10]}...")

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontweight='bold')
    ax.set_title('Policy Loss Over Time', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    # Entropy plot (or Length Penalty as proxy for "reasoning cost")
    ax = axes[0, 1]
    entropy_key = 'entropy'
    if 'entropy/mean' in history.columns: entropy_key = 'entropy/mean'
    
    if entropy_key in history.columns:
        ax.plot(steps, history[entropy_key], 
               color=COLORS['entropy'], linewidth=2, label='Mean Entropy', zorder=3)
        ax.set_title('Model Entropy (Confidence)', fontsize=13, pad=10)
        ax.set_ylabel('Entropy', fontweight='bold')
    elif 'env/all/ledger/length_penalty_norm' in history.columns:
         # Fallback to length penalty visualization if entropy isn't explicitly logged
        ax.plot(steps, history['env/all/ledger/length_penalty_norm'], 
               color=COLORS['entropy'], linewidth=2, label='Length Penalty Norm', zorder=3)
        ax.set_title('Length Penalty (Verbosity Control)', fontsize=13, pad=10)
        ax.set_ylabel('Penalty Norm (1.0 = No Penalty)', fontweight='bold')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    # KL Divergence plot
    ax = axes[1, 0]
    kl_keys = ['kl', 'kl_divergence', 'mismatch_kl/mean', 'train/kl']
    found_kl = False
    for key in kl_keys:
        if key in history.columns:
            ax.plot(steps, history[key], 
                   label='KL Divergence', linewidth=2, color=COLORS['kl'], alpha=0.8)
            found_kl = True
            break
            
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('KL Divergence', fontweight='bold')
    ax.set_title('Policy Divergence (KL)', fontsize=13, pad=10)
    if found_kl:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    # Length / Token usage
    ax = axes[1, 1]
    if 'env/all/ac_tokens_per_turn' in history.columns:
        ax.plot(steps, history['env/all/ac_tokens_per_turn'], 
               label='Avg Tokens/Turn', linewidth=2, color='#4A90E2', alpha=0.8)
    
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Token Count', fontweight='bold')
    ax.set_title('Response Length (Conciseness)', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved training metrics plot: {plot_path}")
    plt.close()


def create_summary_dashboard(run, output_dir: Path):
    """Create a summary dashboard with key metrics for Tinker run."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dakota RL Training (Thinking Machines) - Comprehensive Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    history = run.history()
    steps = history['_step'] if '_step' in history.columns else history.index
    
    # 1. Reward progression (large, top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    reward_series, _ = get_history_series(
        history,
        ["env/all/ledger/composite_predicted", "reward/mean", "env/all/reward/total"],
    )
    if reward_series is not None:
        ax1.plot(steps, reward_series, 
                color=COLORS['reward'], linewidth=3, label='Overall Reward')
        initial = reward_series.iloc[0]
        final = reward_series.iloc[-1]
        if pd.isna(initial): initial = 0.0
        if pd.isna(final): final = 0.0
            
        ax1.fill_between(steps, initial, reward_series, 
                        alpha=0.2, color=COLORS['reward'])
        improvement_pct = ((final-initial)/initial*100) if initial != 0 else 0
        ax1.set_title(f'Reward: {initial:.3f} → {final:.3f} ({improvement_pct:.1f}% improvement)', 
                     fontsize=12, fontweight='bold')
    ax1.set_xlabel('Step', fontweight='bold')
    ax1.set_ylabel('Reward', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Key stats box (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = []
    
    if run.summary:
        if '_runtime' in run.summary:
            runtime = run.summary['_runtime']
            stats_text.append(f"Duration: {runtime/3600:.2f}h")
        if '_step' in run.summary:
            stats_text.append(f"Steps: {run.summary['_step']}")
    
    if reward_series is not None:
        stats_text.append(f"Final Reward: {reward_series.iloc[-1]:.3f}")
    
    affix_series, _ = get_history_series(history, ["env/all/ledger/affix_norm"])
    char_series, _ = get_history_series(history, ["env/all/ledger/char_overlap_norm"])
    
    if affix_series is not None:
        stats_text.append(f"Morphology: {affix_series.iloc[-1]:.3f}")
    if char_series is not None:
        stats_text.append(f"Character: {char_series.iloc[-1]:.3f}")
    
    ax2.text(0.1, 0.5, '\n'.join(stats_text), 
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Component comparison (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if affix_series is not None and char_series is not None:
        components = ['Morphology', 'Character', 'Composite']
        composite_val = reward_series.iloc[-1] if reward_series is not None else 0
        values = [
            affix_series.iloc[-1],
            char_series.iloc[-1],
            composite_val,
        ]
        colors_bar = [COLORS['morphology'], COLORS['character'], COLORS['composite']]
        bars = ax3.bar(components, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Final Value', fontweight='bold')
        ax3.set_title('Component Performance', fontsize=11, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if pd.isna(height): height = 0
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Length Penalty Trend (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'env/all/ledger/length_penalty_norm' in history.columns:
        ax4.plot(steps, history['env/all/ledger/length_penalty_norm'], 
                color='purple', linewidth=2, label='Penalty Norm')
        ax4.set_title('Length Penalty Trend', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Token Usage (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'env/all/ac_tokens_per_turn' in history.columns:
        ax5.plot(steps, history['env/all/ac_tokens_per_turn'], 
                color='#4A90E2', linewidth=2)
        ax5.set_title('Tokens per Turn', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Component Trends (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    if affix_series is not None:
        ax6.plot(steps, affix_series, 
                label='Morphology', color=COLORS['morphology'], linewidth=2)
    if char_series is not None:
        ax6.plot(steps, char_series, 
                label='Character', color=COLORS['character'], linewidth=2)
    ax6.set_xlabel('Step', fontweight='bold')
    ax6.set_ylabel('Reward', fontweight='bold')
    ax6.set_title('Component Trends Detail', fontsize=11, fontweight='bold')
    ax6.set_ylim([0, 1.05])
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    plot_path = output_dir / 'comprehensive_dashboard.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved comprehensive dashboard: {plot_path}")
    plt.close()


def main():
    """Main function to create all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create beautiful visualizations for Tinker RL training runs")
    parser.add_argument("--run-id", type=str, required=True, help="Tinker WandB run ID (e.g., i55d4x26)")
    parser.add_argument("--project", type=str, default="dakota-rl-grammar", help="Wandb project")
    parser.add_argument("--entity", type=str, default="christian-cooper-us", help="Wandb entity")
    parser.add_argument("--output-dir", type=str, default="wandb_visualizations/tinker", help="Output directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING TINKER RL VISUALIZATIONS")
    print("="*80)
    print(f"Run ID: {args.run_id}")
    print(f"Project: {args.project}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load run
    print("Loading run from wandb...")
    try:
        run = load_run_data(args.run_id, args.project, args.entity)
        print(f"[OK] Loaded run: {run.name}")
    except Exception as e:
        print(f"ERROR: Failed to load run: {e}")
        return 1
    
    print("\nGenerating visualizations...")
    
    # Create all plots
    create_reward_progression_plot(run, output_dir)
    create_training_metrics_plot(run, output_dir)
    create_summary_dashboard(run, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}")
    print(f"\nView run in browser: {run.url}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

