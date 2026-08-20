"""
Training Time Estimation Calculator for Dakota RL Training

This script estimates training time based on:
- Dataset size (number of tasks)
- Model size (Qwen2.5-7B-Instruct)
- Batch size and gradient accumulation
- Number of epochs
- Curriculum stages
- Hardware configuration (distributed vs single GPU)
"""

import json
from pathlib import Path
from typing import Dict, List


def count_tasks_in_jsonl(file_path: Path) -> int:
    """Count tasks in a JSONL file."""
    if not file_path.exists():
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def estimate_training_time(
    dataset_size: int,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    effective_batch_size: int = None,
    tasks_per_hour: int = None,
    model_size: str = "7B",
    ppo_epochs: int = 4  # RL-specific: multiple PPO updates per batch
) -> Dict[str, float]:
    """
    Estimate training time for RL training.
    
    Args:
        dataset_size: Number of tasks in dataset
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of epochs
        effective_batch_size: Override calculated effective batch size
        tasks_per_hour: Override calculated tasks per hour (for different hardware)
        model_size: Model size string (e.g., "7B")
    
    Returns:
        Dictionary with time estimates in hours
    """
    
    # Calculate effective batch size
    if effective_batch_size is None:
        effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Estimate tasks per hour based on model size and hardware
    # RL training is MUCH slower than supervised learning due to:
    # - Rollout generation (inference for each task)
    # - Reward computation (verifier calls)
    # - PPO updates (multiple epochs per batch)
    # - TOPLOC verification overhead
    
    if tasks_per_hour is None:
        # Realistic estimates for RL training (7B model):
        # - Single GPU: ~50-150 tasks/hour (slow due to RL overhead)
        # - Multi-GPU distributed (4 workers): ~200-600 tasks/hour
        # - Cloud distributed (PrimeIntellect): ~300-800 tasks/hour
        
        if model_size == "7B":
            # Conservative estimate for distributed RL training
            # This accounts for: inference, reward computation, PPO updates, verification
            base_tasks_per_hour = 400  # Distributed with 4 workers (realistic for RL)
        else:
            base_tasks_per_hour = 200
        
        tasks_per_hour = base_tasks_per_hour
    
    # Calculate steps per epoch
    steps_per_epoch = dataset_size / effective_batch_size
    
    # RL training is slower than supervised learning due to:
    # - Rollout generation (model inference)
    # - Reward computation
    # - PPO updates (multiple epochs per batch)
    # - Verification overhead (if TOPLOC enabled)
    
    # Time per step (in hours)
    # Includes: forward pass, reward computation, backward pass, optimization
    # Note: PPO multiplies this by ppo_epochs (default 4)
    time_per_step = 1.0 / tasks_per_hour  # hours per task
    
    # Steps per epoch
    steps_per_epoch = dataset_size / effective_batch_size
    
    # Time per epoch (accounting for PPO epochs)
    # Each batch requires ppo_epochs updates
    time_per_epoch = steps_per_epoch * time_per_step * ppo_epochs
    
    # Total time for all epochs
    total_time = time_per_epoch * num_epochs
    
    # Add overhead: checkpointing, evaluation, logging
    overhead_factor = 1.15  # 15% overhead
    
    total_time_with_overhead = total_time * overhead_factor
    
    return {
        "dataset_size": dataset_size,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "tasks_per_hour": tasks_per_hour,
        "time_per_epoch_hours": time_per_epoch,
        "total_time_hours": total_time,
        "total_time_with_overhead_hours": total_time_with_overhead,
        "total_time_with_overhead_minutes": total_time_with_overhead * 60,
    }


def estimate_curriculum_training(
    easy_tasks: int,
    medium_tasks: int,
    hard_tasks: int,
    easy_epochs: int = 1,
    medium_epochs: int = 1,
    hard_epochs: int = 1,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    model_size: str = "7B",
    ppo_epochs: int = 4
) -> Dict[str, any]:
    """Estimate time for curriculum learning across multiple stages."""
    
    # Estimate each stage
    easy_estimate = estimate_training_time(
        easy_tasks, batch_size, gradient_accumulation_steps, easy_epochs, model_size=model_size, ppo_epochs=ppo_epochs
    )
    medium_estimate = estimate_training_time(
        medium_tasks, batch_size, gradient_accumulation_steps, medium_epochs, model_size=model_size, ppo_epochs=ppo_epochs
    )
    hard_estimate = estimate_training_time(
        hard_tasks, batch_size, gradient_accumulation_steps, hard_epochs, model_size=model_size, ppo_epochs=ppo_epochs
    )
    
    total_time = (
        easy_estimate["total_time_with_overhead_hours"] +
        medium_estimate["total_time_with_overhead_hours"] +
        hard_estimate["total_time_with_overhead_hours"]
    )
    
    return {
        "easy_stage": easy_estimate,
        "medium_stage": medium_estimate,
        "hard_stage": hard_estimate,
        "total_time_hours": total_time,
        "total_time_minutes": total_time * 60,
        "breakdown": {
            "easy": f"{easy_estimate['total_time_with_overhead_hours']:.2f} hours ({easy_estimate['total_time_with_overhead_minutes']:.0f} min)",
            "medium": f"{medium_estimate['total_time_with_overhead_hours']:.2f} hours ({medium_estimate['total_time_with_overhead_minutes']:.0f} min)",
            "hard": f"{hard_estimate['total_time_with_overhead_hours']:.2f} hours ({hard_estimate['total_time_with_overhead_minutes']:.0f} min)",
        }
    }


def load_config(config_path: Path) -> Dict:
    """Load training configuration."""
    # This is a simplified version - in practice you'd parse YAML/TOML
    return {}


def main():
    """Main estimation function."""
    print("=" * 70)
    print(" Dakota RL Training Time Estimation")
    print("=" * 70)
    print()
    
    # Dataset sizes from your config
    dataset_dir = Path("dakota_rl_training/datasets")
    
    easy_file = dataset_dir / "grammar_tasks_easy.jsonl"
    medium_file = dataset_dir / "grammar_tasks_medium.jsonl"
    hard_file = dataset_dir / "grammar_tasks_hard.jsonl"
    complete_file = dataset_dir / "grammar_tasks_complete.jsonl"
    
    # Count tasks
    easy_tasks = count_tasks_in_jsonl(easy_file)
    medium_tasks = count_tasks_in_jsonl(medium_file)
    hard_tasks = count_tasks_in_jsonl(hard_file)
    total_tasks = count_tasks_in_jsonl(complete_file)
    
    print(f"Dataset Sizes:")
    print(f"  Easy tasks: {easy_tasks:,}")
    print(f"  Medium tasks: {medium_tasks:,}")
    print(f"  Hard tasks: {hard_tasks:,}")
    print(f"  Total tasks: {total_tasks:,}")
    print()
    
    # Training configuration from train.toml
    batch_size = 16  # batch_size from config
    gradient_accumulation_steps = 4  # gradient_accumulation_steps from config
    easy_epochs = 1
    medium_epochs = 1
    hard_epochs = 1
    
    print("Training Configuration:")
    print(f"  Model: Qwen2.5-7B-Instruct")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  PPO epochs: 4 (RL-specific: multiple updates per batch)")
    print()
    
    # Estimate for different scenarios
    print("=" * 70)
    print(" TIME ESTIMATES (Distributed Training - 4 Workers)")
    print("=" * 70)
    print()
    
    # Scenario 1: Distributed training (PrimeIntellect cloud)
    print("Scenario 1: PrimeIntellect Distributed Training (4 workers)")
    print("-" * 70)
    estimate = estimate_curriculum_training(
        easy_tasks, medium_tasks, hard_tasks,
        easy_epochs, medium_epochs, hard_epochs,
        batch_size, gradient_accumulation_steps,
        model_size="7B", ppo_epochs=4
    )
    
    print(f"Easy Stage ({easy_tasks:,} tasks, {easy_epochs} epoch):")
    print(f"  {estimate['breakdown']['easy']}")
    print(f"Medium Stage ({medium_tasks:,} tasks, {medium_epochs} epoch):")
    print(f"  {estimate['breakdown']['medium']}")
    print(f"Hard Stage ({hard_tasks:,} tasks, {hard_epochs} epoch):")
    print(f"  {estimate['breakdown']['hard']}")
    print()
    print(f"TOTAL TRAINING TIME: {estimate['total_time_hours']:.2f} hours ({estimate['total_time_minutes']:.0f} minutes)")
    print(f"  Range: {estimate['total_time_hours'] * 0.8:.1f} - {estimate['total_time_hours'] * 1.4:.1f} hours")
    print()
    
    # Scenario 2: Single GPU (slower)
    print("Scenario 2: Single GPU Training (Local)")
    print("-" * 70)
    # Override with single GPU throughput
    single_gpu_tasks_per_hour = 100  # Much slower for RL
    
    single_gpu_easy = estimate_training_time(
        easy_tasks, batch_size=4, gradient_accumulation_steps=4, num_epochs=easy_epochs,
        tasks_per_hour=single_gpu_tasks_per_hour, model_size="7B"
    )
    single_gpu_medium = estimate_training_time(
        medium_tasks, batch_size=4, gradient_accumulation_steps=4, num_epochs=medium_epochs,
        tasks_per_hour=single_gpu_tasks_per_hour, model_size="7B"
    )
    single_gpu_hard = estimate_training_time(
        hard_tasks, batch_size=4, gradient_accumulation_steps=4, num_epochs=hard_epochs,
        tasks_per_hour=single_gpu_tasks_per_hour, model_size="7B"
    )
    
    single_gpu_total = (
        single_gpu_easy['total_time_with_overhead_hours'] +
        single_gpu_medium['total_time_with_overhead_hours'] +
        single_gpu_hard['total_time_with_overhead_hours']
    )
    
    print(f"  Estimated tasks/hour: ~{single_gpu_tasks_per_hour}")
    print(f"  Total time: ~{single_gpu_total:.1f} hours ({single_gpu_total * 60:.0f} minutes)")
    print(f"  Range: {single_gpu_total * 0.7:.1f} - {single_gpu_total * 1.5:.1f} hours")
    print()
    
    # Per-stage breakdown
    print("=" * 70)
    print(" DETAILED PER-STAGE BREAKDOWN")
    print("=" * 70)
    print()
    
    for stage_name, tasks, epochs in [
        ("Easy", easy_tasks, easy_epochs),
        ("Medium", medium_tasks, medium_epochs),
        ("Hard", hard_tasks, hard_epochs),
    ]:
        stage_est = estimate_training_time(
            tasks, batch_size, gradient_accumulation_steps, epochs, model_size="7B", ppo_epochs=4
        )
        print(f"{stage_name} Stage:")
        print(f"  Tasks: {tasks:,}")
        print(f"  Steps per epoch: {stage_est['steps_per_epoch']:.0f}")
        print(f"  Time per epoch: {stage_est['time_per_epoch_hours']:.2f} hours")
        print(f"  Total ({epochs} epoch): {stage_est['total_time_with_overhead_hours']:.2f} hours")
        print()
    
    # Factors affecting time
    print("=" * 70)
    print(" FACTORS AFFECTING TRAINING TIME")
    print("=" * 70)
    print()
    print("1. Hardware:")
    print("   - GPU type (A100/H100 vs V100)")
    print("   - Number of workers (distributed)")
    print("   - Network bandwidth (for distributed)")
    print()
    print("2. Model Configuration:")
    print("   - Sequence length (longer = slower)")
    print("   - LoRA rank (higher = slightly slower)")
    print("   - Batch size (larger = fewer steps but slower per step)")
    print()
    print("3. RL-Specific:")
    print("   - Rollout generation (inference overhead)")
    print("   - Reward computation (verifier calls)")
    print("   - PPO epochs (default: 4)")
    print("   - TOPLOC verification (adds ~10-20% overhead)")
    print()
    print("4. Dataset:")
    print("   - Task complexity (harder tasks = slower inference)")
    print("   - Average sequence length")
    print("   - Multi-turn tasks (slower than single-turn)")
    print()
    
    # Cost estimation (rough)
    print("=" * 70)
    print(" COST ESTIMATION (PrimeIntellect)")
    print("=" * 70)
    print()
    print("Note: Actual costs depend on PrimeIntellect pricing")
    print(f"Estimated compute time: {estimate['total_time_hours']:.2f} hours")
    print(f"  (Range: {estimate['total_time_hours'] * 0.8:.1f} - {estimate['total_time_hours'] * 1.4:.1f} hours)")
    print()
    print("Typical RL training costs:")
    print("  - $1-5/hour per GPU (varies by instance type)")
    print("  - With 4 workers: ~$4-20/hour total")
    print(f"  - Estimated total: ${estimate['total_time_hours'] * 2:.0f} - ${estimate['total_time_hours'] * 10:.0f}")
    print()
    
    print("=" * 70)
    print(" RECOMMENDATION")
    print("=" * 70)
    print()
    print("For accurate time estimation:")
    print("1. Run a small test run (50-100 tasks) to measure actual throughput")
    print("2. Monitor first epoch to get real tasks/hour")
    print("3. Adjust estimates based on observed performance")
    print()
    print("Expected range for your configuration:")
    print(f"  Minimum: {estimate['total_time_hours'] * 0.7:.1f} hours (fast hardware, optimal settings)")
    print(f"  Typical: {estimate['total_time_hours']:.1f} hours")
    print(f"  Maximum: {estimate['total_time_hours'] * 1.5:.1f} hours (slower hardware, verification overhead)")
    print()


if __name__ == "__main__":
    main()

