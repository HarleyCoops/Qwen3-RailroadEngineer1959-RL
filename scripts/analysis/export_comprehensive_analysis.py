#!/usr/bin/env python3
"""
Comprehensive W&B Data Export for Dakota RL Analysis

Exports:
1. Reward curves: Overall and per-component (char/morph/sem) over training steps
2. Accuracy by difficulty: Final performance on easy/medium/hard/advanced tasks
3. Sample efficiency: How quickly did rewards improve?
4. KL divergence: How much did policy drift from base model?
5. Curriculum timing: Steps spent in each difficulty stage
6. Loss curves: From the trainer run
7. Example generations: Actual model outputs at checkpoints

Usage:
    python scripts/analysis/export_comprehensive_analysis.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed")
    print("Install with: pip install wandb")
    sys.exit(1)

# Default run IDs
DEFAULT_ORCHESTRATOR_RUN = "29hn8w98"
DEFAULT_TRAINER_RUN = "7nikv4vp"
PROJECT = "dakota-rl-grammar"
ENTITY = "christian-cooper-us"


def fetch_wandb_run(run_id: str, entity: str = ENTITY, project: str = PROJECT) -> Optional[Dict]:
    """Fetch run data from W&B API."""
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    
    if not api_key:
        print(f"  WARNING: No WANDB_API_KEY found, cannot fetch {run_id} from API")
        return None
    
    try:
        wandb.login(key=api_key)
        api = wandb.Api()
        run_path = f"{entity}/{project}/{run_id}"
        print(f"  Fetching {run_path} from W&B...")
        run = api.run(run_path)
        
        # Get all available history
        history = run.history(samples=50000)
        
        # Get files (for example generations)
        files = []
        try:
            for file in run.files():
                files.append({
                    "name": file.name,
                    "size": file.size,
                    "url": file.url,
                })
        except:
            pass
        
        return {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": dict(run.summary),
            "config": dict(run.config),
            "history": history,
            "history_columns": history.columns.tolist() if not history.empty else [],
            "files": files,
        }
    except Exception as e:
        print(f"  ERROR fetching {run_id}: {e}")
        return None


def extract_reward_components(history: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Extract reward component curves from history."""
    components = {
        "overall": [],
        "character": [],
        "morphology": [],
        "semantic": [],
    }
    
    # Find reward columns
    reward_cols = [col for col in history.columns if 'reward' in col.lower()]
    
    # Overall reward - prioritize reward/mean over metrics/pattern_reward
    overall_cols = []
    if 'reward/mean' in reward_cols:
        overall_cols = ['reward/mean']
    elif any('total' in col.lower() or 'overall' in col.lower() for col in reward_cols):
        overall_cols = [col for col in reward_cols if 'total' in col.lower() or 'overall' in col.lower()]
    elif reward_cols:
        # Skip pattern_reward if it's all zeros, prefer other reward metrics
        non_pattern = [col for col in reward_cols if 'pattern' not in col.lower()]
        if non_pattern:
            overall_cols = [non_pattern[0]]
        else:
            overall_cols = [reward_cols[0]]
    
    # Character reward (char, character, orthography)
    char_cols = [col for col in reward_cols if any(term in col.lower() for term in ['char', 'character', 'orthography', 'ortho'])]
    
    # Morphology reward (morph, affix)
    morph_cols = [col for col in reward_cols if any(term in col.lower() for term in ['morph', 'affix'])]
    
    # Semantic reward (semantic, meaning, semantic)
    sem_cols = [col for col in reward_cols if any(term in col.lower() for term in ['semantic', 'meaning', 'sem'])]
    
    # Build component DataFrames
    result = {}
    if overall_cols:
        result["overall"] = history[["_step"] + overall_cols].copy()
    if char_cols:
        result["character"] = history[["_step"] + char_cols].copy()
    if morph_cols:
        result["morphology"] = history[["_step"] + morph_cols].copy()
    if sem_cols:
        result["semantic"] = history[["_step"] + sem_cols].copy()
    
    return result


def analyze_accuracy_by_difficulty(history: pd.DataFrame) -> Dict[str, Any]:
    """Analyze accuracy metrics by difficulty level."""
    analysis = {}
    
    # Find difficulty-related columns
    difficulty_cols = {}
    for col in history.columns:
        col_lower = col.lower()
        if 'easy' in col_lower:
            difficulty_cols.setdefault('easy', []).append(col)
        elif 'medium' in col_lower:
            difficulty_cols.setdefault('medium', []).append(col)
        elif 'hard' in col_lower:
            difficulty_cols.setdefault('hard', []).append(col)
        elif 'advanced' in col_lower:
            difficulty_cols.setdefault('advanced', []).append(col)
    
    # Also look for accuracy/success metrics
    accuracy_cols = [col for col in history.columns if 'accuracy' in col.lower() or 'success' in col.lower() or 'correct' in col.lower()]
    
    for difficulty in ['easy', 'medium', 'hard', 'advanced']:
        if difficulty in difficulty_cols:
            cols = difficulty_cols[difficulty]
            # Filter to accuracy-related
            acc_cols = [col for col in cols if any(term in col.lower() for term in ['accuracy', 'success', 'correct', 'reward'])]
            if acc_cols:
                final_values = {}
                for col in acc_cols:
                    values = history[col].dropna()
                    if len(values) > 0:
                        final_values[col] = {
                            "final": float(values.iloc[-1]),
                            "mean": float(values.mean()),
                            "max": float(values.max()),
                        }
                if final_values:
                    analysis[difficulty] = final_values
    
    return analysis


def calculate_sample_efficiency(history: pd.DataFrame, reward_col: str = None) -> Dict[str, Any]:
    """Calculate how quickly rewards improved."""
    if reward_col is None:
        # Find best reward column (prefer reward/mean)
        reward_cols = [col for col in history.columns if 'reward' in col.lower()]
        if not reward_cols:
            return {"error": "No reward column found"}
        # Prefer reward/mean, then non-pattern rewards
        if 'reward/mean' in reward_cols:
            reward_col = 'reward/mean'
        else:
            non_pattern = [col for col in reward_cols if 'pattern' not in col.lower()]
            reward_col = non_pattern[0] if non_pattern else reward_cols[0]
    
    if reward_col not in history.columns:
        return {"error": f"Reward column {reward_col} not found"}
    
    values = history[reward_col].dropna()
    if len(values) < 10:
        return {"error": "Insufficient data"}
    
    # Calculate milestones
    initial_value = values.iloc[0]
    final_value = values.iloc[-1]
    max_value = values.max()
    
    # Find steps to reach certain percentages of improvement
    improvement = final_value - initial_value
    milestones = {}
    
    if improvement > 0:
        for pct in [0.25, 0.5, 0.75, 0.9]:
            target = initial_value + improvement * pct
            steps_to_reach = None
            for idx, val in enumerate(values):
                if val >= target:
                    steps_to_reach = history.iloc[idx]["_step"] if "_step" in history.columns else idx
                    break
            if steps_to_reach is not None:
                milestones[f"{int(pct*100)}%_improvement"] = {
                    "step": int(steps_to_reach),
                    "value": float(target),
                }
    
    # Calculate learning rate (improvement per step)
    steps = history["_step"].values if "_step" in history.columns else np.arange(len(values))
    if len(steps) > 1:
        learning_rate = improvement / (steps[-1] - steps[0]) if steps[-1] != steps[0] else 0
    else:
        learning_rate = 0
    
    return {
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "max_value": float(max_value),
        "total_improvement": float(improvement),
        "improvement_pct": float(improvement / abs(initial_value) * 100) if initial_value != 0 else 0,
        "learning_rate": float(learning_rate),
        "milestones": milestones,
        "total_steps": len(values),
    }


def extract_kl_divergence(history: pd.DataFrame) -> Dict[str, Any]:
    """Extract KL divergence metrics."""
    kl_cols = [col for col in history.columns if 'kl' in col.lower() or 'divergence' in col.lower()]
    
    if not kl_cols:
        return {"error": "No KL divergence columns found"}
    
    analysis = {}
    for col in kl_cols:
        values = history[col].dropna()
        if len(values) > 0:
            analysis[col] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "final": float(values.iloc[-1]),
                "trend": "increasing" if len(values) > 10 and values.iloc[-1] > values.iloc[0] else "decreasing",
            }
    
    return analysis


def analyze_curriculum_timing(history: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """Analyze steps spent in each difficulty stage."""
    # Look for difficulty-related columns or config
    analysis = {
        "stages": {},
        "transitions": [],
    }
    
    # Check config for curriculum settings
    if "curriculum" in str(config).lower() or "difficulty" in str(config).lower():
        # Try to extract curriculum info from config
        for key, value in config.items():
            if 'difficulty' in str(key).lower() or 'curriculum' in str(key).lower():
                analysis["config"][key] = value
    
    # Look for difficulty indicators in history
    difficulty_cols = [col for col in history.columns if 'difficulty' in col.lower() or 'stage' in col.lower()]
    
    if difficulty_cols:
        for col in difficulty_cols:
            values = history[col].dropna().unique()
            for val in values:
                stage_name = str(val)
                stage_data = history[history[col] == val]
                if len(stage_data) > 0:
                    steps = stage_data["_step"].values if "_step" in stage_data.columns else np.arange(len(stage_data))
                    analysis["stages"][stage_name] = {
                        "start_step": int(steps[0]),
                        "end_step": int(steps[-1]),
                        "duration": int(steps[-1] - steps[0]),
                        "samples": len(stage_data),
                    }
    
    return analysis


def extract_loss_curves(history: pd.DataFrame) -> pd.DataFrame:
    """Extract loss curves from trainer history."""
    loss_cols = [col for col in history.columns if 'loss' in col.lower()]
    
    if not loss_cols:
        return pd.DataFrame()
    
    return history[["_step"] + loss_cols].copy()


def extract_example_generations(run_data: Dict) -> List[Dict]:
    """Extract example generations from run files or history."""
    examples = []
    
    # Check history for generation-related columns
    if "history" in run_data and not run_data["history"].empty:
        history = run_data["history"]
        gen_cols = [col for col in history.columns if any(term in col.lower() for term in ['generation', 'output', 'example', 'sample', 'response'])]
        
        if gen_cols:
            # Get examples at different steps
            steps_to_sample = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]
            for step_idx in steps_to_sample:
                if step_idx < len(history):
                    row = history.iloc[step_idx]
                    example = {
                        "step": int(row.get("_step", step_idx)),
                        "generations": {}
                    }
                    for col in gen_cols:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            example["generations"][col] = str(val)
                    if example["generations"]:
                        examples.append(example)
    
    # Check files for generation artifacts
    if "files" in run_data:
        gen_files = [f for f in run_data["files"] if any(term in f["name"].lower() for term in ['generation', 'example', 'output', 'sample'])]
        for file_info in gen_files:
            examples.append({
                "source": "file",
                "filename": file_info["name"],
                "url": file_info.get("url", ""),
            })
    
    return examples


def export_comprehensive_analysis(
    orchestrator_run_id: str = DEFAULT_ORCHESTRATOR_RUN,
    trainer_run_id: str = DEFAULT_TRAINER_RUN,
    output_dir: str = "wandb_analysis"
) -> Dict[str, Any]:
    """Export all requested analyses."""
    print("="*80)
    print("Comprehensive W&B Analysis Export")
    print("="*80)
    print(f"Orchestrator Run: {orchestrator_run_id}")
    print(f"Trainer Run: {trainer_run_id}")
    print()
    
    # Fetch data
    print("Fetching data from W&B...")
    orchestrator_data = fetch_wandb_run(orchestrator_run_id)
    trainer_data = fetch_wandb_run(trainer_run_id)
    
    if not orchestrator_data:
        print("ERROR: Could not fetch orchestrator data")
        return None
    if not trainer_data:
        print("ERROR: Could not fetch trainer data")
        return None
    
    print(f"[OK] Fetched orchestrator data ({len(orchestrator_data['history'])} steps)")
    print(f"[OK] Fetched trainer data ({len(trainer_data['history'])} steps)")
    print()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "exported_at": datetime.now().isoformat(),
        "orchestrator_run": orchestrator_run_id,
        "trainer_run": trainer_run_id,
    }
    
    # 1. Reward curves
    print("1. Extracting reward curves...")
    reward_components = extract_reward_components(orchestrator_data["history"])
    results["reward_curves"] = {}
    for component, df in reward_components.items():
        csv_path = output_path / f"reward_curve_{component}.csv"
        df.to_csv(csv_path, index=False)
        results["reward_curves"][component] = {
            "file": str(csv_path),
            "columns": df.columns.tolist(),
            "steps": len(df),
        }
        print(f"   [OK] {component} reward curve: {csv_path}")
    
    # 2. Accuracy by difficulty
    print("\n2. Analyzing accuracy by difficulty...")
    accuracy_analysis = analyze_accuracy_by_difficulty(orchestrator_data["history"])
    results["accuracy_by_difficulty"] = accuracy_analysis
    if accuracy_analysis:
        json_path = output_path / "accuracy_by_difficulty.json"
        with open(json_path, 'w') as f:
            json.dump(accuracy_analysis, f, indent=2)
        print(f"   [OK] Accuracy analysis: {json_path}")
    else:
        print("   [WARNING] No difficulty-specific accuracy metrics found")
    
    # 3. Sample efficiency
    print("\n3. Calculating sample efficiency...")
    sample_efficiency = calculate_sample_efficiency(orchestrator_data["history"])
    results["sample_efficiency"] = sample_efficiency
    if "error" not in sample_efficiency:
        json_path = output_path / "sample_efficiency.json"
        with open(json_path, 'w') as f:
            json.dump(sample_efficiency, f, indent=2)
        print(f"   [OK] Sample efficiency: {json_path}")
        print(f"   Improvement: {sample_efficiency.get('improvement_pct', 0):.2f}%")
        print(f"   Learning rate: {sample_efficiency.get('learning_rate', 0):.6f} per step")
    
    # 4. KL divergence
    print("\n4. Extracting KL divergence...")
    kl_analysis = extract_kl_divergence(trainer_data["history"])
    results["kl_divergence"] = kl_analysis
    if "error" not in kl_analysis:
        json_path = output_path / "kl_divergence.json"
        with open(json_path, 'w') as f:
            json.dump(kl_analysis, f, indent=2)
        csv_path = output_path / "kl_divergence_curve.csv"
        kl_cols = [col for col in trainer_data["history"].columns if 'kl' in col.lower()]
        if kl_cols:
            trainer_data["history"][["_step"] + kl_cols].to_csv(csv_path, index=False)
        print(f"   [OK] KL divergence: {json_path}")
    else:
        print(f"   [WARNING] {kl_analysis.get('error', 'No KL data found')}")
    
    # 5. Curriculum timing
    print("\n5. Analyzing curriculum timing...")
    curriculum_analysis = analyze_curriculum_timing(orchestrator_data["history"], orchestrator_data.get("config", {}))
    results["curriculum_timing"] = curriculum_analysis
    json_path = output_path / "curriculum_timing.json"
    with open(json_path, 'w') as f:
        json.dump(curriculum_analysis, f, indent=2)
    print(f"   [OK] Curriculum timing: {json_path}")
    
    # 6. Loss curves
    print("\n6. Extracting loss curves...")
    loss_curves = extract_loss_curves(trainer_data["history"])
    if not loss_curves.empty:
        csv_path = output_path / "loss_curves.csv"
        loss_curves.to_csv(csv_path, index=False)
        results["loss_curves"] = {
            "file": str(csv_path),
            "columns": loss_curves.columns.tolist(),
            "steps": len(loss_curves),
        }
        print(f"   [OK] Loss curves: {csv_path}")
    else:
        print("   [WARNING] No loss columns found")
    
    # 7. Example generations
    print("\n7. Extracting example generations...")
    examples = extract_example_generations(orchestrator_data)
    if examples:
        results["example_generations"] = examples
        json_path = output_path / "example_generations.json"
        with open(json_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"   [OK] Example generations: {json_path} ({len(examples)} examples)")
    else:
        print("   [WARNING] No example generations found")
    
    # Save comprehensive summary
    print("\n" + "="*80)
    print("Saving comprehensive summary...")
    summary_path = output_path / "comprehensive_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[OK] Summary saved: {summary_path}")
    
    print("\n" + "="*80)
    print("Export Complete!")
    print("="*80)
    print(f"\nAll files saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export comprehensive W&B analysis data"
    )
    parser.add_argument(
        "--orchestrator-run",
        type=str,
        default=DEFAULT_ORCHESTRATOR_RUN,
        help=f"Orchestrator run ID (default: {DEFAULT_ORCHESTRATOR_RUN})"
    )
    parser.add_argument(
        "--trainer-run",
        type=str,
        default=DEFAULT_TRAINER_RUN,
        help=f"Trainer run ID (default: {DEFAULT_TRAINER_RUN})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wandb_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    try:
        export_comprehensive_analysis(
            orchestrator_run_id=args.orchestrator_run,
            trainer_run_id=args.trainer_run,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\nExport cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

