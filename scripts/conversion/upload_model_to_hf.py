#!/usr/bin/env python3
"""
Upload Dakota RL model weights and model card to HuggingFace.
Supports large safetensors files and updates README metadata.
"""

import argparse
import os
from pathlib import Path
import logging
from huggingface_hub import HfApi, create_repo, metadata_update

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="Target HF repo (e.g., HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890)")
    parser.add_argument("--weights-dir", default="tmp_publish", help="Directory containing adapter_model.safetensors and config")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace API token")
    parser.add_argument("--private", action="store_true", help="Make repo private if creating new")
    parser.add_argument("--commit-message", default="Upload Tinker RL fine-tuned weights", help="Commit message")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.token:
        logger.error("No HF_TOKEN found. Please set HF_TOKEN env var or pass --token")
        return 1

    api = HfApi(token=args.token)
    weights_dir = Path(args.weights_dir)
    
    if not weights_dir.exists():
        logger.error(f"Weights directory not found: {weights_dir}")
        return 1

    # Ensure critical files exist
    required_files = ["adapter_model.safetensors", "adapter_config.json"]
    for f in required_files:
        if not (weights_dir / f).exists():
            logger.error(f"Missing required file: {f}")
            return 1

    logger.info(f"Target Repo: {args.repo_id}")
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        logger.info("Repository verified/created.")
    except Exception as e:
        logger.error(f"Error creating/verifying repo: {e}")
        return 1

    # Upload weights and preview image
    logger.info("Uploading weights and assets...")
    try:
        api.upload_folder(
            folder_path=str(weights_dir),
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            ignore_patterns=["checkpoint_complete", "*.tar.gz"]  # Skip internal markers
        )
        logger.info("Weights uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload weights: {e}")
        return 1

    # Update metadata for preview image if grammar.jpg exists in upload
    if (weights_dir / "grammar.jpg").exists():
        logger.info("Updating model card metadata for preview image...")
        try:
            metadata_update(
                repo_id=args.repo_id,
                metadata={"preview_image": "grammar.jpg"},
                token=args.token,
                commit_message="Update preview image metadata"
            )
            logger.info("Metadata updated successfully.")
        except Exception as e:
            logger.warning(f"Failed to update metadata (non-critical): {e}")

    logger.info("Upload complete.")
    print(f"\nView model at: https://huggingface.co/{args.repo_id}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
