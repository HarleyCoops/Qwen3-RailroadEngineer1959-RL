"""
Upload MODEL_CARD.md to HuggingFace Model Hub.

This script uploads the model card (README.md) to the HuggingFace model repository.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def upload_model_card_to_hf(
    model_card_path: str = "MODEL_CARD.md",
    repo_id: str = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890",
    token: Optional[str] = None,
) -> bool:
    """
    Upload model card to HuggingFace Model Hub.
    
    Args:
        model_card_path: Path to MODEL_CARD.md file
        repo_id: HuggingFace model repo ID (e.g., "username/model-name")
        token: HuggingFace token (if not provided, will use login or env var)
    
    Returns:
        True if successful
    """
    load_dotenv()
    
    # Get token from various sources
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    
    # Try to get cached token from huggingface-cli login
    try:
        from huggingface_hub.utils import HfFolder
        cached_token = HfFolder.get_token()
    except (ImportError, AttributeError):
        # HfFolder might not be available in all versions
        cached_token = None
    
    # Validate tokens and choose the best one
    # If token passed as argument, use it (skip validation for now)
    if not token:
        if env_token:
            # Validate env token first
            logger.info("Found HF_TOKEN in environment. Validating...")
            try:
                api_test = HfApi(token=env_token)
                api_test.whoami()
                token = env_token
                logger.info("[OK] Environment token is valid")
            except Exception:
                logger.warning("[WARNING] HF_TOKEN in environment is invalid or expired")
                logger.warning("Will try cached token from huggingface-cli login...")
                if cached_token:
                    token = cached_token
                    logger.info("[OK] Using cached token from huggingface-cli login")
                else:
                    token = None
        elif cached_token:
            token = cached_token
            logger.info("[OK] Using cached token from huggingface-cli login")
    
    # If still no valid token, try interactive login
    if not token:
        logger.info("No valid token found. Attempting interactive login...")
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
                token = None  # Token is cached, API will use it automatically
            if not token:
                # Token is cached, API will use it automatically
                logger.info("[OK] Login successful - token cached")
                return None  # Return None to indicate use cached token
            logger.info("[OK] Login successful")
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
            logger.error("")
            logger.error("Option 3: Pass token as argument")
            logger.error("  python upload_model_card.py --token your_token_here")
            logger.error("")
            logger.error("NOTE: If HF_TOKEN is set but invalid, remove it from .env")
            logger.error("      or unset it: $env:HF_TOKEN=$null")
            logger.error("="*70)
            raise
    
    # Final verification
    try:
        api_temp = HfApi(token=token)
        user_info = api_temp.whoami()
        logger.info(f"Authenticated as: {user_info.get('name', 'unknown')}")
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        logger.error("Your token may be invalid or expired.")
        logger.error("Get a new token from: https://huggingface.co/settings/tokens")
        raise
    
    # Validate model card exists
    card_path = Path(model_card_path)
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {model_card_path}")
    
    # Read model card content
    logger.info(f"Reading model card from {model_card_path}...")
    with open(card_path, 'r', encoding='utf-8') as f:
        model_card_content = f.read()
    
    # Initialize API (use None if token is cached)
    if token:
        api = HfApi(token=token)
    else:
        # Use cached token
        api = HfApi()
    
    # Check if repo exists, create only if needed
    logger.info(f"Checking HuggingFace model repo: {repo_id}")
    try:
        # Try to get repo info - if this works, repo exists
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"[OK] Repository exists: https://huggingface.co/{repo_id}")
    except Exception:
        # Repo doesn't exist, try to create it
        logger.info(f"Repository not found. Creating new repo: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=False,
            )
            logger.info(f"[OK] Repository created: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            logger.error("Make sure you have permission to create repos under this namespace.")
            raise
    
    # Upload model card as README.md
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    commit_message = f"Update model card ({timestamp} UTC)"
    
    logger.info("Uploading model card (README.md)...")
    try:
        api.upload_file(
            path_or_fileobj=model_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
        logger.info("[OK] Model card uploaded successfully")
    except Exception as e:
        logger.error(f"Failed to upload model card: {e}")
        raise

    # Upload visualizations if they exist
    viz_dir = Path("wandb_visualizations")
    if viz_dir.exists():
        logger.info(f"Found visualizations directory: {viz_dir}")
        for img_file in viz_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                logger.info(f"Uploading {img_file.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(img_file),
                        path_in_repo=f"visualizations/{img_file.name}",
                        repo_id=repo_id,
                        repo_type="model",
                        token=token,
                        commit_message=f"Upload visualization {img_file.name}",
                    )
                    logger.info(f"[OK] {img_file.name} uploaded")
                except Exception as e:
                    logger.error(f"Failed to upload {img_file.name}: {e}")
    
    logger.info("="*70)
    logger.info("[SUCCESS] Model card successfully uploaded to HuggingFace!")
    logger.info(f"   Repository: https://huggingface.co/{repo_id}")
    logger.info("="*70)
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload MODEL_CARD.md to HuggingFace Model Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with default repo
  python upload_model_card.py
  
  # Upload to specific repo
  python upload_model_card.py --repo-id "username/model-name"
  
  # Use custom model card path
  python upload_model_card.py --model-card-path "custom_card.md"
        """
    )
    
    parser.add_argument(
        "--model-card-path",
        type=str,
        default="MODEL_CARD.md",
        help="Path to MODEL_CARD.md file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890",
        help="HuggingFace model repo ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    try:
        upload_model_card_to_hf(
            model_card_path=args.model_card_path,
            repo_id=args.repo_id,
            token=args.token,
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

