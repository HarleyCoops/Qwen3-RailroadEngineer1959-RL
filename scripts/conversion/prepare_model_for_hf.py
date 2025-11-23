"""
Helper script to prepare model for HuggingFace upload.

This script helps locate model files and optionally converts checkpoints
to HuggingFace format if needed.
"""

import os
import sys
from pathlib import Path
from typing import Optional

def find_latest_weights_dir(base_dir: str = "dakota_rl_training/outputs") -> Optional[Path]:
    """Find the latest weights directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return None
    
    # Look for weights directories
    weights_dirs = []
    
    # Check for direct weights/ directory
    weights_path = base_path / "weights"
    if weights_path.exists():
        # Look for step_X subdirectories
        for step_dir in sorted(weights_path.glob("step_*"), reverse=True):
            if step_dir.is_dir():
                weights_dirs.append(step_dir)
    
    # Also check for output directories that might contain weights
    for output_dir in base_path.iterdir():
        if output_dir.is_dir():
            output_weights = output_dir / "weights"
            if output_weights.exists():
                for step_dir in sorted(output_weights.glob("step_*"), reverse=True):
                    if step_dir.is_dir():
                        weights_dirs.append(step_dir)
    
    if weights_dirs:
        # Return the latest (highest step number)
        latest = max(weights_dirs, key=lambda p: int(p.name.split("_")[-1]))
        return latest
    
    return None


def check_model_files(model_dir: Path) -> dict:
    """Check what model files exist in directory."""
    results = {
        "has_config": False,
        "has_tokenizer": False,
        "has_weights": False,
        "weight_files": [],
        "missing_files": [],
    }
    
    # Check config.json
    if (model_dir / "config.json").exists():
        results["has_config"] = True
    else:
        results["missing_files"].append("config.json")
    
    # Check tokenizer files
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ]
    has_any_tokenizer = False
    for tf in tokenizer_files:
        if (model_dir / tf).exists():
            has_any_tokenizer = True
            break
    results["has_tokenizer"] = has_any_tokenizer
    if not has_any_tokenizer:
        results["missing_files"].append("tokenizer files")
    
    # Check weight files
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    for pattern in weight_patterns:
        for weight_file in model_dir.glob(pattern):
            if "model" in weight_file.name.lower():
                results["has_weights"] = True
                results["weight_files"].append(weight_file.name)
    
    if not results["has_weights"]:
        results["missing_files"].append("model weight files")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare and check model files for HuggingFace upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: If model files are on a remote instance, download them first:
  python scripts/conversion/download_model_from_instance.py --instance-ip <ip> --remote-path <path>
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory to check (if not provided, will search for latest)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="dakota_rl_training/outputs",
        help="Base directory to search for model files"
    )
    
    args = parser.parse_args()
    
    # Find or use provided model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"ERROR: Model directory not found: {args.model_dir}")
            print("\n" + "="*70)
            print("MODEL FILES NOT FOUND LOCALLY")
            print("="*70)
            print("Your model files are likely on the remote Prime Intellect instance.")
            print("\nTo download them, run:")
            print("  python scripts/conversion/download_model_from_instance.py \\")
            print("    --instance-ip <your-instance-ip> \\")
            print("    --remote-path ~/dakota_rl_training/outputs/<run_name>/weights/step_400")
            print("\nOr provide the correct local path with --model-dir")
            print("="*70)
            return 1
    else:
        print(f"Searching for latest weights directory in {args.base_dir}...")
        model_dir = find_latest_weights_dir(args.base_dir)
        if not model_dir:
            print(f"ERROR: No weights directory found in {args.base_dir}")
            print("\n" + "="*70)
            print("MODEL FILES NOT FOUND LOCALLY")
            print("="*70)
            print("Your model files are likely on the remote Prime Intellect instance.")
            print("\nTo download them, run:")
            print("  python scripts/conversion/download_model_from_instance.py \\")
            print("    --instance-ip <your-instance-ip> \\")
            print("    --remote-path ~/dakota_rl_training/outputs/<run_name>/weights/step_400")
            print("\nCommon remote paths:")
            print("  ~/dakota_rl_training/outputs/ledger_test_400/weights/step_400")
            print("  ~/dakota_rl_training/outputs/grpo_30b/weights/step_400")
            print("="*70)
            return 1
        print(f"Found: {model_dir}")
    
    # Check files
    print(f"\nChecking model files in: {model_dir}")
    print("="*70)
    
    results = check_model_files(model_dir)
    
    print(f"Config: {'' if results['has_config'] else ''}")
    print(f"Tokenizer: {'' if results['has_tokenizer'] else ''}")
    print(f"Weights: {'' if results['has_weights'] else ''}")
    
    if results["weight_files"]:
        print(f"\nWeight files found:")
        for wf in results["weight_files"]:
            size = (model_dir / wf).stat().st_size / 1024 / 1024
            print(f"  - {wf} ({size:.1f} MB)")
    
    if results["missing_files"]:
        print(f"\n Missing files:")
        for mf in results["missing_files"]:
            print(f"  - {mf}")
        print("\nYou may need to:")
        print("  1. Convert checkpoint using extract_hf_from_ckpt.py")
        print("  2. Or copy files from base model repository")
        return 1
    
    print("\n All required files found!")
    print(f"\nReady to upload! Run:")
    print(f"  python scripts/conversion/upload_model_to_hf.py --model-dir \"{model_dir}\"")
    
    return 0


if __name__ == "__main__":
    exit(main())

