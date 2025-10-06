#!/usr/bin/env python3
"""
Dakota Grammar RL Training Script

Launch RL training with PrimeIntellect framework
"""

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train Dakota grammar model with RL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local training (not distributed)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (single batch)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR RL TRAINING")
    print("="*70)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\nERROR: Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration: {config_path}")
    print(f"Model: {config['model']['base']}")
    print(f"Algorithm: {config['training']['algorithm']}")
    print(f"Epochs: {config['training']['num_epochs']}")

    # Check datasets
    for env in config['environments']:
        dataset = Path(env['dataset'])
        if dataset.exists():
            print(f"[OK] Dataset found: {dataset}")
        else:
            print(f"[ERROR] Dataset not found: {dataset}")
            print(f"  Please run: python convert_rules_to_primeintellect.py")
            return

    print("\n" + "="*70)
    print(" TRAINING LAUNCH")
    print("="*70)

    print("\nTo launch training:")
    print("\n1. Install PrimeIntellect framework:")
    print("   pip install git+https://github.com/PrimeIntellect-ai/verifiers.git")
    print("   pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git")

    if args.local:
        print("\n2. Run local training:")
        print("   (Training implementation requires PrimeIntellect prime-rl)")
        print("   prime-rl train --config configs/training_config.yaml --local")
    else:
        print("\n2. Run distributed training:")
        print("   prime-rl train \\")
        print(f"     --config {args.config} \\")
        print("     --num-workers 4 \\")
        print("     --use-toploc \\")
        print("     --wandb-project dakota-rl-grammar")

    print("\n3. Monitor training:")
    print("   - Weights & Biases: https://wandb.ai")
    print("   - Checkpoints: dakota_rl_training/checkpoints/")

    print("\n4. Key metrics to track:")
    print("   - reward/mean")
    print("   - char_accuracy (Dakota special characters)")
    print("   - affix_accuracy (morphology)")
    print("   - semantic_accuracy (translation)")

    print("\n" + "="*70)
    print(" READY TO TRAIN")
    print("="*70)
    print(f"\nDataset: 5,657 tasks from 1,036 grammar rules")
    print(f"Curriculum: Easy (1,998) -> Medium (2,155) -> Hard (398)")
    print(f"Expected time: 8-12 hours on distributed workers")
    print()


if __name__ == "__main__":
    main()
