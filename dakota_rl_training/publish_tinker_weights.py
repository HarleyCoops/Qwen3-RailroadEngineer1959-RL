#!/usr/bin/env python3
"""Publish Dakota RL checkpoints via the Tinker CLI and mirror to W&B."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dakota_rl_training.tinker_integration import (
    build_metadata,
    download_checkpoint_archive,
    publish_checkpoint,
    select_checkpoint,
    write_metadata,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Tinker checkpoints and export archives.")
    parser.add_argument("--log-path", required=True, help="Training log directory containing checkpoints.jsonl")
    parser.add_argument("--checkpoint-name", default="final", help="Checkpoint name to publish (use 'latest' for most recent entry).")
    parser.add_argument("--output-dir", default="wandb_analysis/published", help="Directory to store downloaded archives + metadata.")
    parser.add_argument("--artifact-name", default="dakota-tinker-weights", help="Friendly name for published weights.")
    parser.add_argument("--wandb-project", default=None, help="Optional W&B project to upload artifact.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/username.")
    parser.add_argument("--wandb-run-name", default="publish-weights", help="Name for the W&B publish run.")
    parser.add_argument("--wandb-run-url", default=None, help="Existing W&B run URL to embed in metadata.")
    parser.add_argument("--skip-publish", action="store_true", help="Skip calling `tinker checkpoint publish` (metadata only).")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading checkpoint archive.")
    return parser.parse_args()


def maybe_upload_wandb(
    project: str | None,
    entity: str | None,
    run_name: str,
    artifact_name: str,
    archive_path: Path | None,
    metadata_path: Path,
) -> None:
    if not project:
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping artifact upload.")
        return

    run = wandb.init(project=project, entity=entity, job_type="publish", name=run_name)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={"source": "tinker", "metadata_path": str(metadata_path)},
    )
    if archive_path:
        artifact.add_file(str(archive_path))
    artifact.add_file(str(metadata_path))
    run.log_artifact(artifact)
    run.finish()


def resolve_checkpoint(log_path: Path, name: str | None) -> dict:
    if name == "latest":
        return select_checkpoint(log_path)
    return select_checkpoint(log_path, name=name)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    checkpoint = resolve_checkpoint(log_path, args.checkpoint_name)
    sampler_path = checkpoint.get("sampler_path")
    if not sampler_path:
        raise ValueError(f"Checkpoint entry missing sampler_path: {checkpoint}")

    logger.info("Selected checkpoint %s at %s", checkpoint.get("name"), sampler_path)
    if not args.skip_publish:
        logger.info("Publishing checkpoint via Tinker CLI...")
        publish_checkpoint(sampler_path)
    else:
        logger.info("Skipping publish step.")

    archive_path: Path | None = None
    if not args.skip_download:
        archive_name = f"{args.artifact_name}.tar.gz"
        archive_path = download_checkpoint_archive(sampler_path, output_dir / archive_name)
        logger.info("Downloaded archive to %s", archive_path)
    else:
        logger.info("Skipping archive download.")

    metadata = build_metadata(checkpoint, log_path, wandb_run_url=args.wandb_run_url)
    metadata_path = write_metadata(metadata, output_dir / f"{args.artifact_name}_metadata.json")
    logger.info("Wrote metadata to %s", metadata_path)

    maybe_upload_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        artifact_name=args.artifact_name,
        archive_path=archive_path,
        metadata_path=metadata_path,
    )
    logger.info("Publish flow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
