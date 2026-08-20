"""
E2B Deployment Example for Qwen3-VL Thinking

This script demonstrates how to use the E2BConnector to:
1. Start a sandbox environment
2. Set up the Qwen3-VL Thinking model
3. Run inference
4. Clean up resources

Usage:
    python e2b/main.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2b.e2b_connector import E2BConnector


def main():
    """
    Main function demonstrating E2B deployment workflow.
    """
    print("=" * 70)
    print("E2B Deployment Example for Qwen3-VL Thinking")
    print("=" * 70)
    print()
    
    # Check for API key
    api_key = os.getenv('E2B_API_KEY')
    if not api_key:
        print("ERROR: E2B_API_KEY environment variable not set!")
        print()
        print("Please set your E2B API key:")
        print("  1. Copy .env.template to .env")
        print("  2. Add your E2B API key to the .env file")
        print("  3. Run this script again")
        print()
        print("Get your API key from: https://e2b.dev/dashboard")
        sys.exit(1)
    
    # Use context manager for automatic cleanup
    try:
        with E2BConnector() as connector:
            print("\n[Step 1/4] Sandbox started")
            print("-" * 70)
            
            # Upload setup files
            print("\n[Step 2/4] Uploading setup files...")
            print("-" * 70)
            connector.upload_setup_files()
            
            # Install dependencies
            print("\n[Step 3/4] Installing dependencies...")
            print("-" * 70)
            connector.install_dependencies()
            
            # Setup model
            print("\n[Step 4/4] Setting up Qwen3-VL Thinking model...")
            print("-" * 70)
            print("NOTE: This step may take several minutes for first-time setup")
            connector.setup_model()
            
            # Run sample inference
            print("\n" + "=" * 70)
            print("Running Sample Inference")
            print("=" * 70)
            
            prompt = "Describe what you see in this image."
            print(f"\nPrompt: {prompt}")
            
            # Example without image
            print("\nRunning text-only inference...")
            result = connector.run_inference(prompt)
            print("\nInference Result:")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Response: {result.get('response', 'No response')}")
            
            # Example with image (if available)
            sample_image = "Public/Dictionary.jpeg"
            if os.path.exists(sample_image):
                print(f"\nRunning inference with image: {sample_image}")
                result = connector.run_inference(
                    prompt="Describe this dictionary page in detail.",
                    image_path=sample_image
                )
                print("\nInference Result:")
                print(f"  Status: {result.get('status', 'unknown')}")
                print(f"  Response: {result.get('response', 'No response')}")
            
            print("\n" + "=" * 70)
            print("Deployment Example Completed Successfully!")
            print("=" * 70)
            print()
            print("Next steps:")
            print("  - Customize setup_model.py for your specific model requirements")
            print("  - Modify the inference script in e2b_connector.py")
            print("  - Add error handling for production use")
            print("  - Implement proper model loading and inference logic")
            print()
            
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_connection():
    """
    Quick test to verify E2B connection without full setup.
    """
    print("Testing E2B connection...")
    try:
        connector = E2BConnector()
        connector.start_sandbox()
        print(" Connection successful!")
        connector.close()
        return True
    except Exception as e:
        print(f" Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Uncomment to run quick connection test first
    # if test_connection():
    #     print("\nProceeding with full deployment...")
    # else:
    #     sys.exit(1)
    
    main()

