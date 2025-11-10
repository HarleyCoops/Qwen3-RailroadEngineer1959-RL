#!/usr/bin/env python3
"""
Example usage of the HF Inference standalone script.
"""

from hf_inference_standalone import DakotaInferenceClient

def main():
    """Example usage of the Dakota Inference Client."""
    
    print("🚀 Dakota Grammar RL Inference Example")
    print("=" * 70)
    
    # Initialize client (will use your HF login)
    print("\n1. Initializing client...")
    try:
        client = DakotaInferenceClient()
        print(f"✅ Client initialized ({client.mode} mode)")
        print(f"   Model: {client.model_id}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nPlease ensure you're logged in:")
        print("  huggingface-cli login")
        return
    
    # Example prompts
    examples = [
        "Translate to Dakota: Hello",
        "Translate to English: Háu",
        "Complete: Wićaŋyaŋpi kta čha",
        "Add the affix -pi to: wićaŋyaŋ"
    ]
    
    print("\n2. Running example prompts...")
    print("=" * 70)
    
    for i, prompt in enumerate(examples, 1):
        print(f"\nExample {i}: {prompt}")
        print("-" * 70)
        
        result = client.generate(
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.3,
            top_p=0.9
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Response: {result['response']}")
    
    print("\n" + "=" * 70)
    print("✅ Examples completed!")
    print("\nTo use interactively, run:")
    print("  python hf_inference_standalone.py --interactive")

if __name__ == "__main__":
    main()

