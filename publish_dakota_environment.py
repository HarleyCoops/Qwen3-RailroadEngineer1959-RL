"""
Publish Dakota Grammar Environment to PrimeIntellect Environment Hub
Adapted from Stoney Nakoda publishing patterns.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Check if required tools are installed."""
    try:
        result = subprocess.run(
            ["prime", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("PrimeIntellect CLI found")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.warning("PrimeIntellect CLI not found. Install with: pip install prime-intellect-cli")
    return False


def build_package(package_dir: Path) -> bool:
    """Build the Python package."""
    logger.info(f"Building package from {package_dir}")
    try:
        subprocess.run(
            ["python", "-m", "build"],
            cwd=package_dir,
            check=True
        )
        logger.info("Package built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("python -m build not available. Install with: pip install build")
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
        # Try to extract from pyproject.toml
        import tomli
        try:
            with open(package_dir / "pyproject.toml", "rb") as f:
                pyproject = tomli.load(f)
                env_name = pyproject.get("project", {}).get("name", "dakota-grammar-translation")
        except Exception:
            env_name = "dakota-grammar-translation"
    
    # Build package first
    if not build_package(package_dir):
        return False
    
    # Publish command
    cmd = [
        "prime", "env", "publish",
        str(package_dir),
        "--owner", owner,
        "--name", env_name,
        "--version", version
    ]
    
    if dry_run:
        logger.info(f"Dry run - would execute: {' '.join(cmd)}")
        return True
    
    logger.info(f"Publishing {env_name} v{version} to PrimeIntellect...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Environment published successfully!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Publish failed: {e}")
        logger.error(e.stderr)
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
        default=os.getenv("PRIMEINTELLECT_OWNER", "HarleyCoops"),
        help="PrimeIntellect owner/organization"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Environment name (defaults to package name)"
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
        logger.info("✓ Publishing complete!")
        logger.info(f"Environment: {args.owner}/{args.name or 'dakota-grammar-translation'}")
        logger.info(f"Install with: pip install {args.name or 'dakota-grammar-translation'}")
    else:
        logger.error("✗ Publishing failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

