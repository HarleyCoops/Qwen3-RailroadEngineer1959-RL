#!/usr/bin/env python3
"""
Create comprehensive visualizations for Dakota RL training runs.
Generates beautiful plots for both trainer and orchestrator runs.
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


def create_reward_progression_plot(orchestrator_run, output_dir: Path):
    """Create beautiful reward progression visualization."""
    history = orchestrator_run.history()
    reward_series, reward_key = get_history_series(
        history,
        [
            "reward/mean",
            "reward/total",
            "env/all/reward/total",
        ],
    )
    if history.empty or reward_series is None:
        print("No reward data available")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Reward Progression - Dakota RL Training', fontsize=18, fontweight='bold', y=0.98)
    
    steps = history['_step'] if '_step' in history.columns else history.index
    
    # Main reward curve
    ax = axes[0]
    ax.plot(steps, reward_series, 
            color=COLORS['reward'], linewidth=2.5, label='Overall Reward', zorder=3)
    
    # Add milestone markers
    milestones = [
        (49, 0.177, '25%', '#90EE90'),
        (71, 0.234, '50%', '#FFD700'),
        (109, 0.292, '75%', '#FFA500'),
        (160, 0.326, '90%', '#FF6347'),
    ]
    
    for step, reward, label, color in milestones:
        if step <= steps.max():
            ax.scatter([step], [reward], s=150, color=color, 
                      edgecolors='black', linewidth=1.5, zorder=5, marker='*')
            ax.annotate(label, xy=(step, reward), xytext=(10, 10),
                       textcoords='offset points', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Add shaded improvement regions
    initial_reward = reward_series.iloc[0]
    final_reward = reward_series.iloc[-1]
    improvement = final_reward - initial_reward
    
    ax.axhspan(initial_reward, final_reward, alpha=0.1, color=COLORS['reward'], zorder=1)
    ax.fill_between(steps, initial_reward, reward_series, 
                    alpha=0.2, color=COLORS['reward'], zorder=2)
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Reward: {initial_reward:.3f} → {final_reward:.3f} ({improvement/final_reward*100:.1f}% improvement)', 
                 fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, steps.max()])
    
    # Component breakdown
    ax2 = axes[1]
    component_metrics = [
        ('Morphological Accuracy', ['metrics/affix_reward', 'env/all/ledger/affix_raw', 'env/all/ledger/affix_norm']),
        ('Character Preservation', ['metrics/char_overlap_reward', 'env/all/ledger/char_overlap_raw', 'env/all/ledger/char_overlap_norm']),
        ('Overall Composite', ['reward/mean', 'reward/total', 'env/all/reward/total']),
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


def create_training_metrics_plot(trainer_run, output_dir: Path):
    """Create comprehensive training metrics visualization."""
    history = trainer_run.history()
    if history.empty:
        print("No training data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics - Dakota RL Trainer', fontsize=18, fontweight='bold', y=0.98)

    steps = history['_step'] if '_step' in history.columns else history.index

    def _first_present(keys):
        for k in keys:
            if k in history.columns:
                return k
        return None

    # Loss plot
    ax = axes[0, 0]
    loss_key = _first_present(['loss/mean', 'train/loss', 'optim/loss'])
    if loss_key:
        ax.plot(steps, history[loss_key],
               color=COLORS['loss'], linewidth=2, label=loss_key, zorder=3)
        std_key = _first_present(['loss/std', 'train/loss_std'])
        if std_key:
            ax.fill_between(steps,
                          history[loss_key] - history[std_key],
                          history[loss_key] + history[std_key],
                          alpha=0.3, color=COLORS['loss'], zorder=2, label='±1 Std')
        ax.set_yscale('log')
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontweight='bold')
    ax.set_title('Policy Loss Over Time', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)

    # Entropy plot
    ax = axes[0, 1]
    entropy_key = _first_present(['entropy/mean', 'optim/entropy'])
    if entropy_key:
        ax.plot(steps, history[entropy_key],
               color=COLORS['entropy'], linewidth=2, label=entropy_key, zorder=3)
        std_key = _first_present(['entropy/std'])
        if std_key:
            ax.fill_between(steps,
                          history[entropy_key] - history[std_key],
                          history[entropy_key] + history[std_key],
                          alpha=0.3, color=COLORS['entropy'], zorder=2)
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Entropy', fontweight='bold')
    ax.set_title('Model Entropy (Confidence)', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)

    # KL Divergence plot
    ax = axes[1, 0]
    kl_metrics = {
        'Masked KL': _first_present(['masked_mismatch_kl/mean', 'optim/kl_sample_train_v1']),
        'Overall KL': _first_present(['mismatch_kl/mean', 'optim/kl_sample_train_v2']),
        'Unmasked KL': _first_present(['unmasked_mismatch_kl/mean']),
    }
    colors_kl = [COLORS['kl'], '#8B0000', '#FF6B6B']
    for i, (label, metric_key) in enumerate(kl_metrics.items()):
        if metric_key:
            ax.plot(steps, history[metric_key], 
                   label=label, linewidth=2, color=colors_kl[i], alpha=0.8)
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('KL Divergence', fontweight='bold')
    ax.set_title('Policy Divergence (KL)', fontsize=13, pad=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    # Inference probabilities
    ax = axes[1, 1]
    if 'inference_probs/mean' in history.columns:
        ax.plot(steps, history['inference_probs/mean'], 
               label='Inference Probs', linewidth=2, color='#4A90E2', alpha=0.8)
    if 'trainer_probs/mean' in history.columns:
        ax.plot(steps, history['trainer_probs/mean'], 
               label='Trainer Probs', linewidth=2, color='#7B68EE', alpha=0.8)
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Model Probabilities', fontsize=13, pad=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved training metrics plot: {plot_path}")
    plt.close()


def create_performance_plot(trainer_run, output_dir: Path):
    """Create performance metrics visualization."""
    history = trainer_run.history()
    if history.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Metrics', fontsize=16, fontweight='bold')
    
    steps = history['_step'] if '_step' in history.columns else history.index
    
    # Throughput
    ax = axes[0]
    if 'perf/throughput' in history.columns:
        ax.plot(steps, history['perf/throughput'], 
               color=COLORS['throughput'], linewidth=2.5, label='Throughput')
        avg_throughput = history['perf/throughput'].mean()
        ax.axhline(y=avg_throughput, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Avg: {avg_throughput:.0f} tok/s')
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Tokens/sec', fontweight='bold')
    ax.set_title('Training Throughput', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    # MFU
    ax = axes[1]
    if 'perf/mfu' in history.columns:
        ax.plot(steps, history['perf/mfu'], 
               color='purple', linewidth=2.5, label='MFU')
        avg_mfu = history['perf/mfu'].mean()
        ax.axhline(y=avg_mfu, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Avg: {avg_mfu:.2f}%')
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Model FLOPS Utilization (%)', fontweight='bold')
    ax.set_title('GPU Efficiency (MFU)', fontsize=13, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)
    
    plt.tight_layout()
    plot_path = output_dir / 'performance_metrics.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved performance metrics plot: {plot_path}")
    plt.close()


def create_summary_dashboard(orchestrator_run, trainer_run, output_dir: Path):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dakota RL Training - Comprehensive Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    orch_history = orchestrator_run.history()
    train_history = trainer_run.history()
    
    orch_steps = orch_history['_step'] if '_step' in orch_history.columns else orch_history.index
    train_steps = train_history['_step'] if '_step' in train_history.columns else train_history.index
    
    # 1. Reward progression (large, top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    orch_reward_series, _ = get_history_series(
        orch_history,
        ["reward/mean", "reward/total", "env/all/reward/total"],
    )
    if orch_reward_series is not None:
        ax1.plot(orch_steps, orch_reward_series, 
                color=COLORS['reward'], linewidth=3, label='Overall Reward')
        initial = orch_reward_series.iloc[0]
        final = orch_reward_series.iloc[-1]
        ax1.fill_between(orch_steps, initial, orch_reward_series, 
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
    
    if orchestrator_run.summary:
        if '_runtime' in orchestrator_run.summary:
            runtime = orchestrator_run.summary['_runtime']
            stats_text.append(f"Duration: {runtime/3600:.2f}h")
        if '_step' in orchestrator_run.summary:
            stats_text.append(f"Steps: {orchestrator_run.summary['_step']}")
    
    if orch_reward_series is not None:
        stats_text.append(f"Final Reward: {orch_reward_series.iloc[-1]:.3f}")
    
    affix_series, _ = get_history_series(
        orch_history,
        ["metrics/affix_reward", "env/all/ledger/affix_raw", "env/all/ledger/affix_norm"],
    )
    char_series, _ = get_history_series(
        orch_history,
        ["metrics/char_overlap_reward", "env/all/ledger/char_overlap_raw", "env/all/ledger/char_overlap_norm"],
    )
    if affix_series is not None:
        stats_text.append(f"Morphology: {affix_series.iloc[-1]:.3f}")
    if char_series is not None:
        stats_text.append(f"Character: {char_series.iloc[-1]:.3f}")
    
    if 'perf/throughput' in train_history.columns:
        stats_text.append(f"Throughput: {train_history['perf/throughput'].mean():.0f} tok/s")
    
    ax2.text(0.1, 0.5, '\n'.join(stats_text), 
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Component comparison (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if affix_series is not None and char_series is not None:
        components = ['Morphology', 'Character', 'Composite']
        composite_val = orch_reward_series.iloc[-1] if orch_reward_series is not None else 0
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
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Loss (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'loss/mean' in train_history.columns:
        ax4.plot(train_steps, train_history['loss/mean'], 
                color=COLORS['loss'], linewidth=2)
        ax4.set_yscale('log')
    ax4.set_xlabel('Step', fontweight='bold')
    ax4.set_ylabel('Loss', fontweight='bold')
    ax4.set_title('Training Loss', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Entropy (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'entropy/mean' in train_history.columns:
        ax5.plot(train_steps, train_history['entropy/mean'], 
                color=COLORS['entropy'], linewidth=2)
    ax5.set_xlabel('Step', fontweight='bold')
    ax5.set_ylabel('Entropy', fontweight='bold')
    ax5.set_title('Model Entropy', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. KL Divergence (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    if 'mismatch_kl/mean' in train_history.columns:
        ax6.plot(train_steps, train_history['mismatch_kl/mean'], 
                color=COLORS['kl'], linewidth=2, label='Overall KL')
        if 'masked_mismatch_kl/mean' in train_history.columns:
            ax6.plot(train_steps, train_history['masked_mismatch_kl/mean'], 
                    color='#8B0000', linewidth=2, label='Masked KL', alpha=0.7)
        ax6.set_yscale('log')
    ax6.set_xlabel('Step', fontweight='bold')
    ax6.set_ylabel('KL Divergence', fontweight='bold')
    ax6.set_title('Policy Divergence', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    # 7. Throughput (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    if 'perf/throughput' in train_history.columns:
        ax7.plot(train_steps, train_history['perf/throughput'], 
               color=COLORS['throughput'], linewidth=2)
    ax7.set_xlabel('Step', fontweight='bold')
    ax7.set_ylabel('Tokens/sec', fontweight='bold')
    ax7.set_title('Throughput', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Component trends (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    morph_series, _ = get_history_series(
        orch_history,
        ["metrics/affix_reward", "env/all/ledger/affix_raw", "env/all/ledger/affix_norm"],
    )
    char_series, _ = get_history_series(
        orch_history,
        ["metrics/char_overlap_reward", "env/all/ledger/char_overlap_raw", "env/all/ledger/char_overlap_norm"],
    )
    if morph_series is not None:
        ax8.plot(orch_steps, morph_series, 
                label='Morphology', color=COLORS['morphology'], linewidth=2)
    if char_series is not None:
        ax8.plot(orch_steps, char_series, 
                label='Character', color=COLORS['character'], linewidth=2)
    ax8.set_xlabel('Step', fontweight='bold')
    ax8.set_ylabel('Reward', fontweight='bold')
    ax8.set_title('Component Trends', fontsize=11, fontweight='bold')
    ax8.set_ylim([0, 1.05])
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=8)
    
    plot_path = output_dir / 'comprehensive_dashboard.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved comprehensive dashboard: {plot_path}")
    plt.close()


def main():
    """Main function to create all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create beautiful visualizations for RL training runs")
    parser.add_argument("--trainer-id", type=str, default="7nikv4vp", help="Trainer run ID")
    parser.add_argument("--orchestrator-id", type=str, default="29hn8w98", help="Orchestrator run ID")
    parser.add_argument("--project", type=str, default="dakota-rl-grammar", help="Wandb project")
    parser.add_argument("--entity", type=str, default="christian-cooper-us", help="Wandb entity")
    parser.add_argument("--output-dir", type=str, default="wandb_visualizations", help="Output directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING RL TRAINING VISUALIZATIONS")
    print("="*80)
    print(f"Trainer Run: {args.trainer_id}")
    print(f"Orchestrator Run: {args.orchestrator_id}")
    print(f"Project: {args.project}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load runs
    print("Loading runs from wandb...")
    try:
        trainer_run = load_run_data(args.trainer_id, args.project, args.entity)
        orchestrator_run = load_run_data(args.orchestrator_id, args.project, args.entity)
        print(f"[OK] Loaded trainer run: {trainer_run.name}")
        print(f"[OK] Loaded orchestrator run: {orchestrator_run.name}")
    except Exception as e:
        print(f"ERROR: Failed to load runs: {e}")
        return 1
    
    print("\nGenerating visualizations...")
    
    # Create all plots
    create_reward_progression_plot(orchestrator_run, output_dir)
    create_training_metrics_plot(trainer_run, output_dir)
    create_performance_plot(trainer_run, output_dir)
    create_summary_dashboard(orchestrator_run, trainer_run, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}")
    print(f"\nView runs in browser:")
    print(f"  Trainer: {trainer_run.url}")
    print(f"  Orchestrator: {orchestrator_run.url}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

