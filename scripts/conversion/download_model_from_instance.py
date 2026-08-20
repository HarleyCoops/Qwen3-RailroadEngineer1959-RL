"""
Download model weights and files from Prime Intellect training instance.

This script downloads model files (weights, config, tokenizer) from your
remote training instance so you can upload them to Hugging Face.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

def run_scp(
    source: str,
    dest: str,
    ssh_key: Optional[str] = None,
    ssh_port: Optional[int] = None,
    user: str = "root"
) -> bool:
    """Run SCP command to download files."""
    scp_cmd = ["scp"]
    
    # Add SSH key if provided
    if ssh_key:
        scp_cmd.extend(["-i", ssh_key])
    
    # Add port if provided
    if ssh_port:
        scp_cmd.extend(["-P", str(ssh_port)])
    
    # Add source and destination
    scp_cmd.extend([source, dest])
    
    print(f"Running: {' '.join(scp_cmd)}")
    try:
        result = subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
        print(f" Successfully downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f" SCP failed: {e.stderr}")
        return False


def download_model_files(
    instance_ip: str,
    remote_path: str,
    local_path: str,
    ssh_key: Optional[str] = None,
    ssh_port: Optional[int] = None,
    user: str = "root"
) -> bool:
    """
    Download model files from remote instance.
    
    Args:
        instance_ip: IP address or hostname of the instance
        remote_path: Remote path to model files (e.g., ~/dakota_rl_training/outputs/ledger_test_400/weights/step_400)
        local_path: Local directory to save files
        ssh_key: Path to SSH private key (optional)
        ssh_port: SSH port (optional, default 22)
        user: SSH user (default: root)
    """
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct remote source path
    remote_source = f"{user}@{instance_ip}:{remote_path}"
    
    # Use -r flag for recursive download
    scp_cmd = ["scp", "-r"]
    
    if ssh_key:
        scp_cmd.extend(["-i", ssh_key])
    
    if ssh_port:
        scp_cmd.extend(["-P", str(ssh_port)])
    
    scp_cmd.extend([remote_source, str(local_dir)])
    
    print(f"Downloading model files from instance...")
    print(f"  Remote: {remote_source}")
    print(f"  Local: {local_dir}")
    print(f"\nRunning: {' '.join(scp_cmd)}")
    
    try:
        result = subprocess.run(scp_cmd, check=True)
        print(f"\n Successfully downloaded model files to {local_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Download failed. Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check SSH key path is correct")
        print("2. Verify instance IP/hostname")
        print("3. Ensure remote path exists")
        print("4. Check SSH connection: ssh -i <key> <user>@<ip>")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download model files from Prime Intellect training instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from instance (interactive)
  python download_model_from_instance.py --instance-ip 185.216.20.236 --remote-path "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400"
  
  # Download with SSH key
  python download_model_from_instance.py --instance-ip 185.216.20.236 --remote-path "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400" --ssh-key "C:\\Users\\chris\\.ssh\\prime_rl_key"
  
  # Download checkpoints instead of weights
  python download_model_from_instance.py --instance-ip 185.216.20.236 --remote-path "~/dakota_rl_training/outputs/ledger_test_400/checkpoints/step_400/trainer" --local-path "local_checkpoints"
        """
    )
    
    parser.add_argument(
        "--instance-ip",
        type=str,
        required=True,
        help="IP address or hostname of Prime Intellect instance"
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        required=True,
        help="Remote path to model files (e.g., ~/dakota_rl_training/outputs/ledger_test_400/weights/step_400)"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="downloaded_model",
        help="Local directory to save files (default: downloaded_model)"
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        default=None,
        help="Path to SSH private key (optional, will try default locations)"
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=None,
        help="SSH port (default: 22)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="root",
        help="SSH user (default: root)"
    )
    
    args = parser.parse_args()
    
    # Try to find SSH key if not provided
    ssh_key = args.ssh_key
    if not ssh_key:
        # Check common locations
        possible_keys = [
            Path.home() / ".ssh" / "prime_rl_key",
            Path.home() / ".ssh" / "id_rsa",
            Path("C:/Users/chris/.ssh/prime_rl_key"),
        ]
        for key_path in possible_keys:
            if key_path.exists():
                ssh_key = str(key_path)
                print(f"Found SSH key: {ssh_key}")
                break
    
    if not ssh_key or not Path(ssh_key).exists():
        print(" Warning: No SSH key found. You may need to provide --ssh-key")
        print("  Or ensure SSH key is in ~/.ssh/prime_rl_key")
    
    # Download files
    success = download_model_files(
        instance_ip=args.instance_ip,
        remote_path=args.remote_path,
        local_path=args.local_path,
        ssh_key=ssh_key,
        ssh_port=args.ssh_port,
        user=args.user
    )
    
    if success:
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Files saved to: {Path(args.local_path).absolute()}")
        print(f"\nNext step: Upload to Hugging Face")
        print(f"  python scripts/conversion/upload_model_to_hf.py --model-dir \"{args.local_path}\"")
        print("="*70)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())

