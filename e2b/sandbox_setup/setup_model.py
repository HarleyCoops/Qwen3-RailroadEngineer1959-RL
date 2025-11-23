"""
Model Setup Script for E2B Sandbox

This script runs inside the E2B sandbox to:
1. Download the Qwen3-VL Thinking model
2. Initialize the model and tokenizer
3. Verify the setup
4. Cache the model for future use

This script is executed by e2b_connector.py during the setup phase.
"""

import os
import sys
from pathlib import Path


def setup_model():
    """
    Download and initialize the Qwen3-VL Thinking model.
    """
    print("=" * 70)
    print("Qwen3-VL Thinking Model Setup")
    print("=" * 70)
    print()
    
    try:
        # Import required libraries
        print("[1/5] Importing libraries...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(" Libraries imported successfully")
        
        # Set model configuration
        print("\n[2/5] Configuring model parameters...")
        model_name = "Qwen/Qwen3-VL-7B-Instruct"  # Default model
        cache_dir = "/root/.cache/huggingface"
        
        # Create cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f" Cache directory: {cache_dir}")
        
        # Check for API key (optional, for private models)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            print(" Hugging Face token detected")
        else:
            print("ℹ No Hugging Face token (using public model)")
        
        # Download and load tokenizer
        print("\n[3/5] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True
        )
        print(f" Tokenizer loaded: {model_name}")
        
        # Download and load model
        print("\n[4/5] Downloading model (this may take several minutes)...")
        print("Note: The model size is substantial, please be patient...")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load model with appropriate settings
        model_kwargs = {
            "cache_dir": cache_dir,
            "token": hf_token,
            "trust_remote_code": True,
        }
        
        # Add device-specific settings
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            print("WARNING: Running on CPU. Performance will be significantly slower.")
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        print(f" Model loaded: {model_name}")
        
        # Verify model setup
        print("\n[5/5] Verifying model setup...")
        test_text = "Hello, I am Qwen3-VL Thinking."
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(" Model verification successful")
        
        # Save model info
        info_file = Path("/root/model_info.txt")
        with open(info_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Cache: {cache_dir}\n")
            f.write(f"Status: Ready\n")
        
        print("\n" + "=" * 70)
        print("Model Setup Complete!")
        print("=" * 70)
        print()
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Cache: {cache_dir}")
        print()
        print("The model is now ready for inference.")
        print()
        
        return 0
        
    except ImportError as e:
        print(f"\n Import Error: {e}")
        print("\nMake sure all required packages are installed:")
        print("  pip install -r /root/requirements.txt")
        return 1
        
    except Exception as e:
        print(f"\n Setup Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def check_gpu():
    """
    Check GPU availability and info.
    """
    try:
        import torch
        print("\nGPU Information:")
        print("-" * 70)
        if torch.cuda.is_available():
            print(f" CUDA available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        else:
            print("ℹ CUDA not available - using CPU")
        print()
    except Exception as e:
        print(f"Could not check GPU: {e}")


if __name__ == "__main__":
    print("\nStarting model setup process...")
    print()
    
    # Check GPU first
    check_gpu()
    
    # Run setup
    exit_code = setup_model()
    
    sys.exit(exit_code)


