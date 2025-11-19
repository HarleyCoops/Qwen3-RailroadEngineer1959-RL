#!/usr/bin/env python3
"""Launch Dakota RL training on Tinker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root and environments directory to sys.path to allow imports
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT_DIR / "environments"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.append(str(ENV_DIR))

# Add dakota_rl_training to path to allow importing verifiers
RL_TRAINING_DIR = ROOT_DIR / "dakota_rl_training"
if str(RL_TRAINING_DIR) not in sys.path:
    sys.path.append(str(RL_TRAINING_DIR))

from dotenv import load_dotenv
import weave
from tinker_cookbook.rl import train

from dakota_rl_training.tinker_integration import (
    DakotaGrammarDatasetBuilder,
    export_reward_ledger,
)

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Dakota grammar RL policy on Tinker.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model to fine-tune.")
    parser.add_argument("--log-path", default="dakota_rl_training/outputs/tinker_run", help="Directory for logs/checkpoints.")
    parser.add_argument("--dataset-path", default=None, help="Optional override JSONL dataset.")
    parser.add_argument("--eval-path", default=None, help="Optional eval split JSONL.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of env groups per batch.")
    parser.add_argument("--group-size", type=int, default=16, help="Rollouts per GRPO group.")
    parser.add_argument("--max-examples", type=int, default=-1, help="Limit number of training examples.")
    parser.add_argument("--eval-examples", type=int, default=-1, help="Limit number of eval examples.")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval split when eval_path not provided.")
    parser.add_argument("--system-prompt", default=None, help="Override default Dakota system prompt.")
    parser.add_argument("--difficulty-filter", nargs="*", default=None, help="Filter dataset difficulties (case-insensitive).")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Filter dataset task types.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffling seed.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max sampled tokens per completion.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--learning-rate", type=float, default=4e-5, help="Optimizer learning rate.")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank for Tinker training client.")
    parser.add_argument("--loss-fn", choices=["importance_sampling", "ppo"], default="importance_sampling", help="RL loss function.")
    parser.add_argument("--num-substeps", type=int, default=1, help="Optimizer steps per training iteration.")
    parser.add_argument("--kl-penalty-coef", type=float, default=0.0, help="KL penalty coefficient.")
    parser.add_argument("--kl-discount-factor", type=float, default=0.0, help="KL penalty discount.")
    parser.add_argument("--eval-every", type=int, default=20, help="How often to run eval batches.")
    parser.add_argument("--save-every", type=int, default=20, help="How often to checkpoint weights.")
    parser.add_argument("--num-groups-to-log", type=int, default=4, help="Trajectory groups to pretty-print/logtree.")
    parser.add_argument("--wandb-project", default="thinking-machines-qwen3-30b", help="Weights & Biases project.")
    parser.add_argument("--wandb-name", default=None, help="Weights & Biases run name.")
    parser.add_argument("--base-url", default=None, help="Custom Tinker API base URL.")
    parser.add_argument("--ledger-csv", default="wandb_analysis/reward_ledger_tinker.csv", help="Output CSV for reward ledger.")
    parser.add_argument("--async-groups", type=int, default=0, help="Enable async mode with this many groups per batch.")
    parser.add_argument("--async-max-steps", type=int, default=4, help="Maximum policy lag when async mode enabled.")
    parser.add_argument("--stream-groups-per-batch", type=int, default=0, help="Streaming minibatch total groups.")
    parser.add_argument("--stream-num-minibatches", type=int, default=2, help="Streaming minibatch subdivision.")
    return parser


def build_dataset_builder(args: argparse.Namespace) -> DakotaGrammarDatasetBuilder:
    return DakotaGrammarDatasetBuilder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        group_size=args.group_size,
        dataset_path=args.dataset_path,
        eval_path=args.eval_path,
        max_examples=args.max_examples,
        eval_examples=args.eval_examples,
        eval_fraction=args.eval_fraction,
        system_prompt=args.system_prompt,
        difficulty_filter=args.difficulty_filter,
        task_filter=args.task_filter,
        seed=args.seed,
    )


def build_config(args: argparse.Namespace) -> train.Config:
    dataset_builder = build_dataset_builder(args)
    async_config = None
    if args.async_groups > 0:
        async_config = train.AsyncConfig(
            max_steps_off_policy=args.async_max_steps,
            groups_per_batch=args.async_groups,
        )
    stream_config = None
    if args.stream_groups_per_batch > 0:
        stream_config = train.StreamMinibatchConfig(
            groups_per_batch=args.stream_groups_per_batch,
            num_minibatches=args.stream_num_minibatches,
        )

    return train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        loss_fn=args.loss_fn,
        num_substeps=args.num_substeps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        log_path=args.log_path,
        base_url=args.base_url,
        eval_every=args.eval_every,
        save_every=args.save_every,
        remove_constant_reward_groups=True,
        async_config=async_config,
        stream_minibatch_config=stream_config,
        num_groups_to_log=args.num_groups_to_log,
    )


@weave.op()
async def run_training(cfg: train.Config) -> None:
    await train.main(cfg)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    load_dotenv()

    cfg = build_config(args)
    Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Tinker RL training run at %s", cfg.log_path)

    # Initialize Weave
    if cfg.wandb_project:
        weave.init(cfg.wandb_project)

    asyncio.run(run_training(cfg))
    logger.info("Training complete.")

    metrics_path = Path(cfg.log_path) / "metrics.jsonl"
    ledger_csv = Path(args.ledger_csv)
    if metrics_path.exists():
        export_reward_ledger(metrics_path, ledger_csv)
        logger.info("Reward ledger exported to %s", ledger_csv)
    else:
        logger.warning("metrics.jsonl not found; skipping ledger export.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
