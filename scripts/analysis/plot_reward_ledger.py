"""
Plot reward ledger reconciliation visualization.

Shows how components combine to form the final reward.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


def plot_reward_ledger(
    csv_path: str = "wandb_analysis/reward_ledger.csv",
    out_path: str = "wandb_analysis/reward_ledger.png"
):
    """
    Generate reward ledger reconciliation plot.
    
    Shows component contributions, penalties, multipliers, and final reward.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    df = df.sort_values("step")
    
    # Recomposition terms (weighted components)
    comp_exact = df["w_exact"] * df["exact_match_norm"]
    comp_char = df["w_char"] * df["char_overlap_norm"]
    comp_pattern = df["w_pattern"] * df["pattern_norm"]
    comp_affix = df["w_affix"] * df["affix_norm"]
    pre = df["composite_pre"]
    length_mult = df["length_penalty_norm"]  # This is a multiplier, not a penalty
    dm = df["difficulty_multiplier"]
    pred = df["composite_predicted"]
    rew = df["reward_scalar"]
    
    # Calculate length penalty (as a penalty, not multiplier)
    # length_penalty_norm is actually a multiplier (1.0 = no penalty)
    # So penalty = 1.0 - multiplier
    length_penalty = 1.0 - df["length_penalty_norm"]
    
    plt.figure(figsize=(14, 8))
    
    # Plot weighted components
    plt.plot(df["step"], comp_exact, label="w_exact·exact_match_norm", alpha=0.7, linewidth=1.5)
    plt.plot(df["step"], comp_char, label="w_char·char_overlap_norm", alpha=0.7, linewidth=1.5)
    plt.plot(df["step"], comp_pattern, label="w_pattern·pattern_norm", alpha=0.7, linewidth=1.5)
    plt.plot(df["step"], comp_affix, label="w_affix·affix_norm", alpha=0.7, linewidth=1.5)
    
    # Plot composites
    plt.plot(df["step"], pre, label="composite_pre", linewidth=2, linestyle="--")
    plt.plot(df["step"], length_mult, label="length_penalty_multiplier", alpha=0.6, linewidth=1.5)
    plt.plot(df["step"], dm, label="difficulty_multiplier", alpha=0.6, linewidth=1.5)
    
    # Plot final predictions
    plt.plot(df["step"], pred, label="composite_predicted", linewidth=2.5, linestyle=":", color="purple")
    plt.plot(df["step"], rew, label="reward_scalar", linewidth=3, color="red", marker="o", markersize=2)
    
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.title("Reward Ledger Reconciliation (per step means)", fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontweight="bold")
    plt.ylabel("Value", fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved reward ledger plot to: {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot reward ledger reconciliation")
    parser.add_argument("--csv", type=str, default="wandb_analysis/reward_ledger.csv",
                       help="Path to reward ledger CSV")
    parser.add_argument("--out", type=str, default="wandb_analysis/reward_ledger.png",
                       help="Output path for plot")
    
    args = parser.parse_args()
    plot_reward_ledger(args.csv, args.out)

