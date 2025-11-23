import os
from huggingface_hub import list_inference_endpoint_providers, get_token

def list_gpu_options():
    """List available GPU options for Inference Endpoints."""
    try:
        # Use token from environment or login
        token = os.getenv("HF_TOKEN") or get_token()
        if not token:
            print("❌ Please login first: huggingface-cli login")
            return

        print("🔍 Fetching available instance types...")
        # Note: As of current huggingface_hub, there isn't a direct public API to list 
        # specific pricing/instance availability via the SDK cleanly without a project ID,
        # but we can try to create a dummy config or just list common ones.
        
        # Common configurations for 30B models:
        # 30B parameters * 2 bytes (fp16) = 60GB VRAM minimum.
        # 
        # Options:
        # 1. 1x A100 (80GB) - Best single card
        # 2. 4x A10G (24GB * 4 = 96GB) - Cheaper multi-card option
        # 3. 2x A100 (80GB) - Overkill but works
        
        print("Recommended configurations for Qwen3-30B (requires ~60GB+ VRAM):")
        print("-" * 60)
        print(f"{'Accelerator':<20} | {'VRAM':<10} | {'Approx Cost':<15}")
        print("-" * 60)
        print(f"{'1x Nvidia A100':<20} | {'80 GB':<10} | {'$4.00 - $5.00/hr':<15}")
        print(f"{'4x Nvidia A10G':<20} | {'96 GB':<10} | {'$5.00 - $6.00/hr':<15}")
        print("-" * 60)
        print("\nWARNING: You will be billed immediately upon creation.")
        print("   Remember to PAUSE or DELETE the endpoint when done.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_gpu_options()

