"""
Test inference with Qwen3-VL-235B-A22B-Thinking via OpenRouter.
This script tests both text and image analysis capabilities.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the client
from implementation.examples.openrouter_integration import Qwen3VLClient

def test_text_inference():
    """Test basic text reasoning."""
    print("=" * 60)
    print("TEST 1: Text Reasoning")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        return False
    
    print(f"✓ API Key loaded: {api_key[:20]}...")
    
    try:
        client = Qwen3VLClient(api_key)
        print("✓ Client initialized")
        
        print("\nSending query: 'Explain what a vision-language model is in one sentence.'")
        response = client.chat(
            "Explain what a vision-language model is in one sentence.",
            thinking_budget=1000,
            temperature=0.7,
            max_tokens=500
        )
        
        print("\n" + "-" * 60)
        print("RESPONSE:")
        print("-" * 60)
        print(response["text"])
        print("-" * 60)
        print(f"Reasoning tokens used: {response.get('reasoning_tokens', 'N/A')}")
        print(f"Total tokens: {response.get('usage', {}).get('total_tokens', 'N/A')}")
        
        if response.get("reasoning"):
            print("\n" + "-" * 60)
            print("REASONING PROCESS (first 300 chars):")
            print("-" * 60)
            print(response["reasoning"][:300] + "...")
        
        print("\n✅ Text inference test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Text inference test FAILED: {str(e)}\n")
        
        # Try to get more details from the response
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print("API Error Details:")
            print(e.response.text)
        
        import traceback
        traceback.print_exc()
        return False


def test_image_inference():
    """Test image analysis with the Blackfeet dictionary."""
    print("=" * 60)
    print("TEST 2: Image Analysis (Blackfeet Dictionary)")
    print("=" * 60)
    
    image_path = Path("Public/Dictionary.jpeg")
    
    if not image_path.exists():
        print(f"❌ Image not found at {image_path}")
        print("Please ensure the dictionary image exists in the Public/ folder.")
        return False
    
    print(f"✓ Image found: {image_path}")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    try:
        client = Qwen3VLClient(api_key)
        
        print("\nAnalyzing dictionary image...")
        print("Query: 'What language is this dictionary for? List 3 words you can see.'")
        
        response = client.analyze_image(
            image_path,
            "What language is this dictionary for? List 3 words you can see with their translations if visible.",
            thinking_budget="medium",
            temperature=0.6,
            max_tokens=800
        )
        
        print("\n" + "-" * 60)
        print("ANALYSIS:")
        print("-" * 60)
        print(response["text"])
        print("-" * 60)
        print(f"Reasoning tokens used: {response.get('reasoning_tokens', 'N/A')}")
        
        if response.get("reasoning"):
            print("\n" + "-" * 60)
            print("REASONING PROCESS (first 300 chars):")
            print("-" * 60)
            print(response["reasoning"][:300] + "...")
        
        print("\n✅ Image inference test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Image inference test FAILED: {str(e)}\n")
        
        # Try to get more details from the response
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print("API Error Details:")
            print(e.response.text)
        
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("  Qwen3-VL-235B-A22B-Thinking Inference Test")
    print("*" * 60)
    print("\n")
    
    # Test 1: Text inference
    text_passed = test_text_inference()
    
    # Test 2: Image inference
    image_passed = test_image_inference()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Text Inference:  {'✅ PASSED' if text_passed else '❌ FAILED'}")
    print(f"Image Inference: {'✅ PASSED' if image_passed else '❌ FAILED'}")
    print("=" * 60)
    
    if text_passed and image_passed:
        print("\n🎉 All tests passed! You're ready to use the model.")
        print("\nNext steps:")
        print("  1. Try modifying the prompts in this script")
        print("  2. Experiment with different thinking_budget values")
        print("  3. Analyze your own images")
        print("  4. Explore the examples in implementation/examples/")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("\nCommon issues:")
        print("  - Invalid API key (check your .env file)")
        print("  - Network connectivity")
        print("  - Missing dependencies (run: pip install -r requirements.txt)")
        print("  - Rate limits (wait a moment and try again)")


if __name__ == "__main__":
    main()
