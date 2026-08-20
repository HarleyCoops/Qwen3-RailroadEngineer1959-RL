#!/usr/bin/env python3
"""
Automatically push files to HuggingFace Space.

Usage:
    python scripts/push_space.py
    
Or specify Space name:
    python scripts/push_space.py --space-name "HarleyCooper/Dakota-Grammar-Demo"
"""

import argparse
import os
from pathlib import Path
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.errors import RepositoryNotFoundError

# Default Space name (update this to your actual Space name)
DEFAULT_SPACE_NAME = "HarleyCooper/Dakota-.6B"

# Files to push from huggingface_space/
SPACE_FILES = [
    "app.py",
    "requirements.txt",
    "README.md"
]


def push_to_space(
    space_name: str,
    source_dir: Path = Path("huggingface_space"),
    token: Optional[str] = None
) -> bool:
    """
    Push files to HuggingFace Space.
    
    Args:
        space_name: Space name (e.g., "HarleyCooper/Dakota-Grammar-Demo")
        source_dir: Directory containing Space files
        token: Optional HF token (uses login if not provided)
    
    Returns:
        True if successful, False otherwise
    """
    # Get token
    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("No HF token found. Attempting to login...")
            try:
                login()
                # After login, try to get token from cache
                from huggingface_hub.utils import HfFolder
                token = HfFolder.get_token()
                if not token:
                    print("Could not retrieve token after login. Using API without explicit token.")
            except Exception as e:
                print(f"Login failed: {e}")
                print("Please set HF_TOKEN environment variable or run: huggingface-cli login")
                return False
    
    # Initialize API
    api = HfApi(token=token)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return False
    
    print(f"Pushing files to Space: {space_name}")
    print(f"Source directory: {source_dir}")
    print("=" * 70)
    
    # Check if Space exists, create if it doesn't
    try:
        api.repo_info(repo_id=space_name, repo_type="space", token=token)
        print(f"Space exists: {space_name}")
    except RepositoryNotFoundError:
        print(f"Space not found: {space_name}")
        print("Creating Space...")
        try:
            # Read README to get SDK type
            readme_path = source_dir / "README.md"
            sdk = "gradio"  # Default
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'sdk: gradio' in content.lower():
                        sdk = "gradio"
                    elif 'sdk: streamlit' in content.lower():
                        sdk = "streamlit"
            
            api.create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk=sdk,
                token=token,
                exist_ok=False
            )
            print(f"Created Space: {space_name}")
        except Exception as e:
            print(f"Failed to create Space: {e}")
            print(f"\nPlease create the Space manually:")
            print(f"   1. Go to https://huggingface.co/spaces")
            print(f"   2. Click 'Create new Space'")
            print(f"   3. Name it: {space_name.split('/')[-1]}")
            print(f"   4. Select SDK: {sdk}")
            print(f"   5. Then run this script again")
            return False
    
    # Upload all files and verify commits were created
    uploaded = []
    failed = []
    actually_committed = []
    
    # Get the latest commit before uploading to compare
    try:
        recent_commits = api.list_repo_commits(
            repo_id=space_name,
            repo_type="space",
            token=token,
        )
        commit_before = recent_commits[0].commit_id if recent_commits else None
    except Exception:
        commit_before = None
    
    # Upload each file
    for filename in SPACE_FILES:
        file_path = source_dir / filename
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            failed.append(filename)
            continue
        
        try:
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=space_name,
                repo_type="space",
                token=token,
                commit_message=f"Update {filename}",
            )
            uploaded.append(filename)
            
            # Check if a new commit was created
            try:
                recent_commits = api.list_repo_commits(
                    repo_id=space_name,
                    repo_type="space",
                    token=token,
                )
                commit_after = recent_commits[0].commit_id if recent_commits else None
                if commit_after != commit_before:
                    print(f"  [COMMITTED] {filename}")
                    actually_committed.append(filename)
                    commit_before = commit_after  # Update for next file
                else:
                    print(f"  [SKIPPED] No changes to {filename}")
            except Exception:
                # Can't verify, assume it worked
                print(f"  [UPLOADED] {filename}")
                actually_committed.append(filename)
                
        except RepositoryNotFoundError as e:
            error_msg = str(e)
            print(f"  [ERROR] Space not found: {error_msg}")
            failed.append(filename)
        except Exception as e:
            error_str = str(e)
            if "no files have been modified" in error_str.lower() or "skipping" in error_str.lower():
                print(f"  [SKIPPED] No changes to {filename}")
                uploaded.append(filename)  # File is up to date
            else:
                print(f"  [ERROR] Failed: {e}")
                failed.append(filename)
    
    # Summary
    print("\n" + "=" * 70)
    print("Upload Summary:")
    print(f"   Files processed: {len(uploaded)}")
    print(f"   Actually committed: {len(actually_committed)}")
    print(f"   Failed: {len(failed)}")
    
    if actually_committed:
        print(f"\n[SUCCESS] Committed {len(actually_committed)} file(s): {', '.join(actually_committed)}")
        print(f"   Space will rebuild: https://huggingface.co/spaces/{space_name}")
    
    if uploaded and not actually_committed:
        print(f"\n[INFO] All files are already up to date (no changes to commit)")
    
    if failed:
        print(f"\n[ERROR] Failed to upload {len(failed)} file(s): {', '.join(failed)}")
        return False
    
    return True


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Push files to HuggingFace Space"
    )
    parser.add_argument(
        "--space-name",
        type=str,
        default=DEFAULT_SPACE_NAME,
        help=f"Space name (default: {DEFAULT_SPACE_NAME})"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="huggingface_space",
        help="Source directory (default: huggingface_space)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, uses login if not provided)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path
    source_dir = Path(args.source_dir)
    
    # Push files
    success = push_to_space(
        space_name=args.space_name,
        source_dir=source_dir,
        token=args.token
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())


