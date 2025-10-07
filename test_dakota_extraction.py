"""
Test Blackfoot Character Extraction

This script tests whether the VLM correctly captures Blackfoot special characters
like ć, š, ŋ, ḣ, ṡ from historical grammar/dictionary images.
"""

import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from implementation.examples.openrouter_integration import Qwen3VLClient
from blackfeet_extraction.core.blackfoot_extraction_prompt import build_blackfoot_extraction_prompt


def test_blackfoot_extraction(image_path: Path, thinking_budget: int = 6000):
    """
    Test extraction on a Blackfoot grammar page.

    Args:
        image_path: Path to image file
        thinking_budget: Reasoning tokens (higher = better accuracy)
    """
    print("\n" + "="*80)
    print(" BLACKFOOT CHARACTER EXTRACTION TEST")
    print("="*80)
    print(f"\nImage: {image_path.name}")
    print(f"Thinking budget: {thinking_budget} tokens")
    print()

    # Initialize client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        print("Set it in your .env file or export it:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        return

    client = Qwen3VLClient(api_key)

    # Build extraction prompt
    page_context = "This is page from Blackfoot Grammar and Dictionary. Focus on preserving special characters: ć, š, ŋ, ḣ, ṡ, ó, á"
    prompt = build_blackfoot_extraction_prompt(page_context)

    print("🔍 Analyzing image with Qwen3-VL Thinking model...")
    print("   (This uses extended reasoning to ensure character accuracy)")
    print()

    # Extract with high thinking budget
    result = client.analyze_image(
        image_path=image_path,
        prompt=prompt,
        thinking_budget=thinking_budget
    )

    # Display results
    print("\n" + "="*80)
    print(" EXTRACTION RESULTS")
    print("="*80)

    print(f"\n📊 Reasoning tokens used: {result['reasoning_tokens']}")
    print(f"📝 Response tokens: {result['response_tokens']}")
    print()

    # Try to parse as JSON
    try:
        extracted_data = json.loads(result['text'])
        print("✅ Successfully parsed JSON output\n")

        # Check for special characters
        print("🔤 SPECIAL CHARACTERS DETECTED:")
        print("-" * 80)

        if 'interlinear_entries' in extracted_data:
            all_special_chars = set()
            for entry in extracted_data['interlinear_entries']:
                if 'special_characters_found' in entry:
                    all_special_chars.update(entry['special_characters_found'])

            if all_special_chars:
                print(f"Found: {', '.join(sorted(all_special_chars))}")
                print()
            else:
                print("No special characters listed (check extraction)")
                print()

        # Display sample entries
        if 'interlinear_entries' in extracted_data and extracted_data['interlinear_entries']:
            print("📖 SAMPLE EXTRACTED ENTRIES:")
            print("-" * 80)

            for i, entry in enumerate(extracted_data['interlinear_entries'][:3], 1):
                print(f"\nEntry {i}:")
                print(f"  Blackfoot: {entry.get('blackfoot_text', 'N/A')}")
                print(f"  Glosses:   {' '.join(entry.get('word_glosses', []))}")
                print(f"  English:   {entry.get('english_translation', 'N/A')}")
                print(f"  Confidence: {entry.get('confidence', 'N/A')}")

                # Highlight special characters
                special = entry.get('special_characters_found', [])
                if special:
                    print(f"  Special chars in this entry: {', '.join(special)}")

        # Display vocabulary items
        if 'vocabulary_items' in extracted_data and extracted_data['vocabulary_items']:
            print("\n\n📚 VOCABULARY EXTRACTED:")
            print("-" * 80)

            for item in extracted_data['vocabulary_items'][:5]:
                word = item.get('blackfoot_word', 'N/A')
                gloss = item.get('gloss', 'N/A')
                chars = item.get('special_chars', [])

                char_display = f" [{', '.join(chars)}]" if chars else ""
                print(f"  {word:<20} = {gloss}{char_display}")

        # Save full results
        output_path = Path("data/test_extraction_blackfoot.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        print(f"\n\n💾 Full extraction saved to: {output_path}")

    except json.JSONDecodeError as e:
        print("⚠️  Warning: Could not parse as JSON")
        print(f"Error: {e}")
        print("\nRaw response:")
        print("-" * 80)
        print(result['text'])

    # Display reasoning trace if available
    if result.get('reasoning'):
        print("\n\n🧠 REASONING TRACE (First 500 chars):")
        print("-" * 80)
        print(result['reasoning'][:500])
        if len(result['reasoning']) > 500:
            print(f"\n... (truncated, {len(result['reasoning'])} total chars)")

        # Save reasoning trace
        reasoning_path = Path("data/test_extraction_blackfoot_reasoning.txt")
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            f.write(result['reasoning'])
        print(f"\n💾 Full reasoning trace saved to: {reasoning_path}")

    print("\n" + "="*80)
    print(" TEST COMPLETE")
    print("="*80)
    print()

    # Character validation summary
    print("🎯 CHARACTER VALIDATION:")
    print("-" * 80)
    print("Check the extracted text above for these Blackfoot characters:")
    print("  ✓ ć (c-acute)     - in Wićášta, mićú")
    print("  ✓ š (s-caron)     - in Wićášta, wašte")
    print("  ✓ ŋ (eng)         - in éiŋhiŋtku, toŋaŋa")
    print("  ✓ ó (o-acute)     - in Wióni")
    print("  ✓ á, é, í, ú      - various pitch accents")
    print("  ✓ ḣ, ṡ            - dotted consonants (if present)")
    print()
    print("If these characters appear correctly above, the VLM extraction is working! ✅")
    print()


if __name__ == "__main__":
    # Test on first Blackfoot page
    test_image = Path("data/processed_images/grammardictionar00riggrich_0089.jpg")

    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        print("\nAvailable images:")
        for img in Path("data/processed_images").glob("*.jpg"):
            print(f"  {img}")
        sys.exit(1)

    test_blackfoot_extraction(test_image, thinking_budget=6000)
