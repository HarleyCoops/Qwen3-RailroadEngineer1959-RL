import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_tokenizer.py <target_directory>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    print(f"Downloading tokenizer files from {base_model} to {target_dir}...")

    try:
        snapshot_download(
            repo_id=base_model,
            local_dir=target_dir,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json"
            ],
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


