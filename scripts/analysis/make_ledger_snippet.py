"""
Generate markdown table snippet from reward ledger CSV.

Creates a head/tail table for README inclusion.
"""

import pandas as pd
from pathlib import Path


def make_ledger_snippet(
    csv_path: str = "wandb_analysis/reward_ledger.csv",
    out_path: str = "wandb_analysis/reward_ledger_head_tail.md",
    head_rows: int = 3,
    tail_rows: int = 3
):
    """
    Generate markdown table with first and last rows from ledger CSV.
    """
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    # Select key columns for display
    cols = [
        "step",
        "char_overlap_norm",
        "affix_norm",
        "exact_match_norm",
        "w_char",
        "w_affix",
        "w_exact",
        "difficulty_multiplier",
        "length_penalty_norm",
        "composite_predicted",
        "reward_scalar",
        "composite_diff",
    ]
    
    # Filter to columns that exist
    cols = [c for c in cols if c in df.columns]
    
    # Get head and tail
    head = df[cols].head(head_rows)
    tail = df[cols].tail(tail_rows)
    snippet = pd.concat([head, tail])
    
    # Generate markdown
    md = snippet.to_markdown(index=False, floatfmt=".4f")
    
    # Write to file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md)
    print(f"Saved markdown snippet to: {out_path}")
    print("\n" + md)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reward ledger markdown snippet")
    parser.add_argument("--csv", type=str, default="wandb_analysis/reward_ledger.csv",
                       help="Path to reward ledger CSV")
    parser.add_argument("--out", type=str, default="wandb_analysis/reward_ledger_head_tail.md",
                       help="Output path for markdown")
    parser.add_argument("--head", type=int, default=3, help="Number of head rows")
    parser.add_argument("--tail", type=int, default=3, help="Number of tail rows")
    
    args = parser.parse_args()
    make_ledger_snippet(args.csv, args.out, args.head, args.tail)

