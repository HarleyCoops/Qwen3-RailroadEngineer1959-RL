"""
Upload complete model to HuggingFace Model Hub.

This script uploads:
- Model weights (from checkpoint or weights directory)
- Model config
- Tokenizer files
- Model card (README.md)
- Any additional files needed for inference

Supports both:
1. Direct upload from weights directory (HF-compatible format)
2. Conversion from checkpoint format using extract_hf_from_ckpt.py
"""

import os
import json
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_hf_token(token: Optional[str] = None) -> str:
    """Get HuggingFace token from various sources."""
    load_dotenv()
    
    # Priority: explicit token > env var > cached token > interactive login
    if token:
        return token
    
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    if env_token:
        logger.info("Found HF_TOKEN in environment")
        return env_token
    
    # Try to get cached token from huggingface-cli login
    try:
        from huggingface_hub.utils import HfFolder
        cached_token = HfFolder.get_token()
    except (ImportError, AttributeError):
        # HfFolder might not be available in all versions
        # Try alternative method
        try:
            api_temp = HfApi()
            # This will use cached token if available
            cached_token = None
        except Exception:
            cached_token = None
    
    if cached_token:
        logger.info("Using cached token from huggingface-cli login")
        return cached_token
    
    logger.info("No token found. Attempting interactive login...")
    try:
        login()
        # After login, try to get token
        try:
            from huggingface_hub.utils import HfFolder
            token = HfFolder.get_token()
        except (ImportError, AttributeError):
            # If HfFolder not available, token should be cached by login()
            # Try to validate by creating API instance
            api_test = HfApi()
            api_test.whoami()  # This will use cached token
            token = "cached"  # Placeholder - API will use cached token
        if not token or token == "cached":
            # Token is cached, API will use it automatically
            logger.info("[OK] Login successful - token cached")
            return None  # Return None to indicate use cached token
        logger.info("[OK] Login successful")
        return token
    except Exception as e:
        logger.error(f"Login failed: {e}")
        logger.error("\n" + "="*70)
        logger.error("AUTHENTICATION REQUIRED")
        logger.error("="*70)
        logger.error("Please do ONE of the following:")
        logger.error("")
        logger.error("Option 1: Use huggingface-cli login (RECOMMENDED)")
        logger.error("  Run: huggingface-cli login")
        logger.error("  Then run this script again")
        logger.error("")
        logger.error("Option 2: Set valid token in .env file")
        logger.error("  Add this line to your .env file:")
        logger.error("  HF_TOKEN=your_token_here")
        logger.error("  Get token from: https://huggingface.co/settings/tokens")
        logger.error("="*70)
        raise


def find_model_files(model_dir: Path) -> dict:
    """Find all model files in directory."""
    files = {
        "config": None,
        "tokenizer_config": None,
        "tokenizer_files": [],
        "model_files": [],
        "generation_config": None,
    }
    
    # Find config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        files["config"] = config_path
    
    # Find generation_config.json
    gen_config_path = model_dir / "generation_config.json"
    if gen_config_path.exists():
        files["generation_config"] = gen_config_path
    
    # Find tokenizer files
    tokenizer_config = model_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        files["tokenizer_config"] = tokenizer_config
    
    # Find all tokenizer-related files
    tokenizer_patterns = [
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]
    for pattern in tokenizer_patterns:
        tokenizer_file = model_dir / pattern
        if tokenizer_file.exists():
            files["tokenizer_files"].append(tokenizer_file)
    
    # Find model weight files
    for ext in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        for model_file in model_dir.glob(ext):
            if "model" in model_file.name.lower() or "pytorch_model" in model_file.name.lower():
                files["model_files"].append(model_file)
    
    # Also check for model.safetensors.index.json
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        files["model_files"].append(index_file)
    
    return files


def upload_model_to_hf(
    model_dir: str,
    repo_id: str = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890",
    model_card_path: Optional[str] = "MODEL_CARD.md",
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> bool:
    """
    Upload complete model to HuggingFace Model Hub.
    
    Args:
        model_dir: Directory containing model files (weights, config, tokenizer)
        repo_id: HuggingFace model repo ID (e.g., "username/model-name")
        model_card_path: Path to MODEL_CARD.md file (optional)
        token: HuggingFace token (if not provided, will use login or env var)
        private: Whether to make the repo private
        commit_message: Custom commit message (optional)
    
    Returns:
        True if successful
    """
    # Get authentication token
    hf_token = get_hf_token(token)
    
    # Validate token (use None if token is cached)
    try:
        if hf_token:
            api = HfApi(token=hf_token)
        else:
            # Use cached token
            api = HfApi()
        user_info = api.whoami()
        logger.info(f"Authenticated as: {user_info.get('name', 'unknown')}")
        # Use the token for uploads
        token = hf_token
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise
    
    # Validate model directory
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")
    
    # Find model files
    logger.info(f"Scanning model directory: {model_path}")
    model_files = find_model_files(model_path)
    
    # Validate essential files exist
    if not model_files["config"]:
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not model_files["model_files"]:
        logger.warning("No model weight files found! Make sure weights are in the directory.")
        logger.warning("Expected files: model.safetensors, pytorch_model.bin, or similar")
    
    # Log found files
    logger.info("Found files:")
    if model_files["config"]:
        logger.info(f"  ✓ config.json")
    if model_files["generation_config"]:
        logger.info(f"  ✓ generation_config.json")
    if model_files["tokenizer_config"]:
        logger.info(f"  ✓ tokenizer_config.json")
    logger.info(f"  ✓ Tokenizer files: {len(model_files['tokenizer_files'])}")
    logger.info(f"  ✓ Model weight files: {len(model_files['model_files'])}")
    
    # Check/create repo
    logger.info(f"Checking HuggingFace model repo: {repo_id}")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"[OK] Repository exists: https://huggingface.co/{repo_id}")
    except Exception:
        logger.info(f"Repository not found. Creating new repo: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=False,
            )
            logger.info(f"[OK] Repository created: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            raise
    
    # Prepare commit message
    if not commit_message:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        commit_message = f"Upload model files ({timestamp} UTC)"
    
    # Upload files
    uploaded_count = 0
    
    # Upload config.json
    if model_files["config"]:
        logger.info(f"Uploading config.json...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_files["config"]),
                path_in_repo="config.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )
            logger.info("[OK] config.json uploaded")
            uploaded_count += 1
        except Exception as e:
            logger.error(f"Failed to upload config.json: {e}")
            raise
    
    # Upload generation_config.json
    if model_files["generation_config"]:
        logger.info(f"Uploading generation_config.json...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_files["generation_config"]),
                path_in_repo="generation_config.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )
            logger.info("[OK] generation_config.json uploaded")
            uploaded_count += 1
        except Exception as e:
            logger.warning(f"Failed to upload generation_config.json: {e}")
    
    # Upload tokenizer files
    if model_files["tokenizer_config"]:
        logger.info(f"Uploading tokenizer_config.json...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_files["tokenizer_config"]),
                path_in_repo="tokenizer_config.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )
            logger.info("[OK] tokenizer_config.json uploaded")
            uploaded_count += 1
        except Exception as e:
            logger.error(f"Failed to upload tokenizer_config.json: {e}")
            raise
    
    for tokenizer_file in model_files["tokenizer_files"]:
        logger.info(f"Uploading {tokenizer_file.name}...")
        try:
            api.upload_file(
                path_or_fileobj=str(tokenizer_file),
                path_in_repo=tokenizer_file.name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )
            logger.info(f"[OK] {tokenizer_file.name} uploaded")
            uploaded_count += 1
        except Exception as e:
            logger.error(f"Failed to upload {tokenizer_file.name}: {e}")
            raise
    
    # Upload model weight files
    for model_file in model_files["model_files"]:
        logger.info(f"Uploading {model_file.name} ({model_file.stat().st_size / 1024 / 1024:.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=model_file.name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )
            logger.info(f"[OK] {model_file.name} uploaded")
            uploaded_count += 1
        except Exception as e:
            logger.error(f"Failed to upload {model_file.name}: {e}")
            raise
    
    # Upload model card (README.md)
    if model_card_path:
        card_path = Path(model_card_path)
        if card_path.exists():
            logger.info(f"Uploading model card (README.md)...")
            try:
                with open(card_path, 'r', encoding='utf-8') as f:
                    model_card_content = f.read()
                
                api.upload_file(
                    path_or_fileobj=model_card_content.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                    commit_message=commit_message,
                )
                logger.info("[OK] Model card uploaded")
                uploaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to upload model card: {e}")
        else:
            logger.warning(f"Model card not found: {model_card_path}")
    
    logger.info("="*70)
    logger.info("[SUCCESS] Model successfully uploaded to HuggingFace!")
    logger.info(f"   Repository: https://huggingface.co/{repo_id}")
    logger.info(f"   Files uploaded: {uploaded_count}")
    logger.info("="*70)
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload complete model to HuggingFace Model Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from weights directory
  python upload_model_to_hf.py --model-dir "outputs/weights/step_400"
  
  # Upload to specific repo
  python upload_model_to_hf.py --model-dir "outputs/weights/step_400" --repo-id "username/model-name"
  
  # Upload without model card
  python upload_model_to_hf.py --model-dir "outputs/weights/step_400" --no-model-card
  
  # Upload as private repo
  python upload_model_to_hf.py --model-dir "outputs/weights/step_400" --private
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model files (weights, config, tokenizer)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890",
        help="HuggingFace model repo ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--model-card-path",
        type=str,
        default="MODEL_CARD.md",
        help="Path to MODEL_CARD.md file (default: MODEL_CARD.md)"
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip uploading model card"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message"
    )
    
    args = parser.parse_args()
    
    try:
        upload_model_to_hf(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            model_card_path=None if args.no_model_card else args.model_card_path,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
        )
        
        print("\n" + "="*70)
        print("UPLOAD SUMMARY")
        print("="*70)
        print(f"Repository: {args.repo_id}")
        print(f"URL: https://huggingface.co/{args.repo_id}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

