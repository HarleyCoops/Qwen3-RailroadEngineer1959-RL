"""
Publish Dakota Grammar Environment to PrimeIntellect Environment Hub
Adapted from Stoney Nakoda publishing patterns.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Check if required tools are installed."""
    # Check for uv-installed prime in .local/bin (common on Windows)
    import platform
    if platform.system() == "Windows":
        user_home = Path.home()
        uv_bin_path = user_home / ".local" / "bin"
        if uv_bin_path.exists() and str(uv_bin_path) not in os.environ.get("PATH", ""):
            # Add to PATH for this session
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{uv_bin_path}{os.pathsep}{current_path}"
            logger.info(f"Added {uv_bin_path} to PATH for this session")
    
    # Try local prime first
    try:
        result = subprocess.run(
            ["prime", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("PrimeIntellect CLI found (local)")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try WSL prime
    try:
        result = subprocess.run(
            ["wsl", "prime", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("PrimeIntellect CLI found (WSL)")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.warning("PrimeIntellect CLI not found. Install with: uv tool install prime")
    logger.warning("If installed via uv, add C:\\Users\\<user>\\.local\\bin to your PATH")
    return False


def get_prime_command() -> List[str]:
    """Get the correct prime command (local or WSL)."""
    # Try local first
    try:
        subprocess.run(["prime", "--version"], capture_output=True, timeout=2)
        return ["prime"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fall back to WSL
    return ["wsl", "prime"]


def build_package(package_dir: Path) -> bool:
    """Build the Python package."""
    logger.info(f"Building package from {package_dir}")
    
    # Try python3 -m build first (most common in WSL)
    try:
        result = subprocess.run(
            ["python3", "-m", "build"],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Package built successfully with python3 -m build")
        return True
    except subprocess.CalledProcessError as e:
        logger.debug(f"python3 -m build failed: {e.stderr}")
        pass
    except FileNotFoundError:
        pass
    
    # Try python -m build
    try:
        result = subprocess.run(
            ["python", "-m", "build"],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Package built successfully with python -m build")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try uv build (if uv is available)
    try:
        result = subprocess.run(
            ["uv", "build"],
            cwd=package_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Package built successfully with uv build")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    logger.error("Build failed: Need either 'python -m build' or 'uv build'. Install with:")
    logger.error("  pip install build")
    logger.error("  or")
    logger.error("  uv pip install build")
    return False


def publish_environment(
    package_dir: Path,
    owner: str,
    env_name: Optional[str] = None,
    version: str = "0.1.0",
    dry_run: bool = False
) -> bool:
    """
    Publish environment to PrimeIntellect Environment Hub.
    
    Args:
        package_dir: Path to package directory (contains pyproject.toml)
        owner: Owner/organization name on PrimeIntellect
        env_name: Environment name (defaults to package name)
        version: Version to publish
        dry_run: If True, print command without executing
    """
    load_dotenv()
    
    if not check_prerequisites():
        return False
    
    package_dir = Path(package_dir).resolve()
    if not package_dir.exists():
        logger.error(f"Package directory not found: {package_dir}")
        return False
    
    if not (package_dir / "pyproject.toml").exists():
        logger.error(f"pyproject.toml not found in {package_dir}")
        return False
    
    if env_name is None:
        env_name = "Dakota1890"  # Default to Dakota1890
    
    # Build package first
    if not build_package(package_dir):
        return False
    
    # Get the correct prime command (local or WSL)
    prime_cmd = get_prime_command()
    
    # According to PrimeIntellect docs, 'prime env push' should be run
    # from inside the package directory and reads metadata from pyproject.toml
    # No need to pass --owner, --name, or --version flags
    
    # Change to package directory and run push
    # For WSL, convert Windows path to WSL path if needed
    work_dir = str(package_dir)
    if prime_cmd[0] == "wsl" and work_dir.startswith("C:"):
        work_dir = work_dir.replace("C:\\", "/mnt/c/").replace("\\", "/")
    
    # Push command - must be run from package directory
    cmd = prime_cmd + ["env", "push"]
    
    # Optional flags
    if dry_run:
        logger.info(f"Dry run - would execute: {' '.join(cmd)}")
        logger.info(f"  From directory: {work_dir}")
        return True
    
    logger.info(f"Pushing {env_name} v{version} to PrimeIntellect...")
    logger.info(f"Running from: {work_dir}")
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Environment pushed successfully!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Push failed: {e}")
        if e.stderr:
            logger.error(e.stderr)
        if e.stdout:
            logger.error(e.stdout)
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Publish Dakota Grammar Environment to PrimeIntellect"
    )
    parser.add_argument(
        "--package-dir",
        type=str,
        default="environments/dakota_grammar_translation",
        help="Path to package directory"
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=os.getenv("PRIMEINTELLECT_OWNER", "cm758faf6000zwilxepmmi7xp"),
        help="PrimeIntellect owner/organization (defaults to your user ID)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Dakota1890",
        help="Environment name (defaults to Dakota1890)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.1.0",
        help="Version to publish"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )
    
    args = parser.parse_args()
    
    success = publish_environment(
        package_dir=Path(args.package_dir),
        owner=args.owner,
        env_name=args.name,
        version=args.version,
        dry_run=args.dry_run
    )
    
    if success:
        logger.info("Publishing complete!")
        logger.info(f"Environment: {args.owner}/{args.name or 'Dakota1890'}")
        logger.info(f"View at: https://primeintellect.ai/env/{args.owner}/{args.name or 'Dakota1890'}")
        logger.info(f"\nTo use in training:")
        logger.info(f"  environment: {args.owner}/{args.name or 'Dakota1890'}")
    else:
        logger.error("Publishing failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

