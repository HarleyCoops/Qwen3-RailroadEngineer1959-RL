#!/usr/bin/env python3
"""
Dakota Grammar Extraction Pipeline
Extracts grammar rules from pages 1-88 for RL training
"""

import base64
import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from blackfeet_extraction.tools.image_converter import ImageConverter

# Load environment
load_dotenv()


def build_grammar_extraction_prompt():
    """Build specialized prompt for grammar rule extraction."""
    return """You are extracting Dakota language grammar rules from a historical linguistic text.

This page contains grammatical analysis and linguistic rules for the Dakota language.

Your task: Extract ALL grammatical information as structured data.

Focus on:
1. Grammar rules (verb conjugation, noun declension, particle usage)
2. Example sentences with interlinear glosses
3. Linguistic patterns and transformations
4. Morphological analysis
5. Phonological rules

For interlinear examples, preserve this format:
- Line 1: Dakota text (preserve ć, š, ŋ, ḣ, ṡ, and all diacritics)
- Line 2: Word-by-word English glosses
- Line 3: Full English translation

CRITICAL: Preserve ALL Dakota special characters exactly:
- Acute accents: á, é, í, ó, ú (pitch markers)
- Caron diacritics: č, š, ž
- Eng: ŋ (ng sound)
- Dotted consonants: ḣ, ṡ
- Syllable breaks: hyphens (e.g., wa-šté)
- Glottal stop: ʼ

Return JSON with this structure:
{
  "page_number": <int>,
  "grammar_rules": [
    {
      "rule_id": "grammar_p<page>_r<rule_num>",
      "rule_type": "<morphology|syntax|phonology|particles|conjugation|declension>",
      "rule_description": "<clear description of the grammatical rule>",
      "dakota_pattern": "<Dakota language pattern or structure>",
      "english_explanation": "<explanation in English>",
      "examples": [
        {
          "dakota": "<Dakota text with all diacritics>",
          "gloss": "<word-by-word gloss if provided>",
          "english": "<English translation>",
          "notes": "<any linguistic notes>"
        }
      ],
      "constraints": "<any constraints or conditions>",
      "confidence": <0.0-1.0>
    }
  ],
  "interlinear_texts": [
    {
      "text_id": "interlinear_p<page>_t<text_num>",
      "dakota_lines": ["<line 1>", "<line 2>", ...],
      "gloss_lines": ["<gloss 1>", "<gloss 2>", ...],
      "english_translation": "<full translation>",
      "linguistic_notes": "<any annotations>",
      "confidence": <0.0-1.0>
    }
  ],
  "linguistic_terms": [
    {
      "term": "<linguistic term>",
      "definition": "<definition or explanation>",
      "dakota_examples": ["<example 1>", "<example 2>"]
    }
  ],
  "page_notes": "<any additional context or notes>",
  "special_characters_found": ["list of Dakota special characters used"],
  "extraction_quality": "<excellent|good|fair|poor>",
  "extraction_notes": "<any extraction challenges or ambiguities>"
}

Rules:
- Extract EVERY grammar rule, no matter how small
- Preserve ALL Dakota text exactly as written
- Maintain interlinear alignment
- Include ALL examples
- Note any unclear or damaged text
- Set confidence based on text clarity

Output ONLY valid JSON, no additional text."""


def extract_grammar_page_with_claude(image_path: Path, page_number: int) -> dict:
    """Extract grammar rules from a single page using Claude."""

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Read and encode image
    print(f"  Reading {image_path.name}...")
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Build prompt
    prompt = build_grammar_extraction_prompt()

    print("  Sending to Claude Sonnet 4.5...")

    # Extract with Claude
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )

    # Parse response
    response_text = response.content[0].text

    # Extract JSON from response
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        extraction = json.loads(response_text)
        extraction["page_number"] = page_number
        return extraction
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON: {e}")
        print(f"  Response preview: {response_text[:500]}")
        return {
            "page_number": page_number,
            "grammar_rules": [],
            "interlinear_texts": [],
            "linguistic_terms": [],
            "error": str(e),
            "raw_response": response_text
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Dakota grammar rules from pages 1-88"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="1-88",
        help="Page range to extract (e.g., 1-10, 1-88)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test on page 10 only"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR EXTRACTION")
    print(" Pages 1-88: Grammar Rules and Linguistic Analysis")
    print("="*70)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("Add to .env file: ANTHROPIC_API_KEY=your_key_here")
        return

    # Setup directories
    jp2_dir = Path("dictionary/grammardictionar00riggrich_jp2")
    processed_dir = Path("data/processed_images")
    output_dir = Path("data/grammar_extracted")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine page range
    if args.test:
        start_page, end_page = 10, 10
        print("\nTEST MODE: Processing page 10 only")
    else:
        if "-" in args.pages:
            start_page, end_page = map(int, args.pages.split("-"))
        else:
            start_page = end_page = int(args.pages)

    num_pages = end_page - start_page + 1
    print(f"\nProcessing pages {start_page}-{end_page} ({num_pages} pages)")
    print(f"Estimated cost: ${num_pages * 0.25:.2f}")
    print(f"Estimated time: {num_pages * 2} minutes")

    if not args.test and not args.yes and num_pages > 10:
        confirm = input("\nContinue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return

    # Convert JP2 to JPEG if needed
    print("\nStep 1: Converting images...")
    converter = ImageConverter(
        input_dir=str(jp2_dir),
        output_dir=str(processed_dir),
        quality=95
    )

    for page_num in range(start_page, end_page + 1):
        jp2_file = jp2_dir / f"grammardictionar00riggrich_{page_num:04d}.jp2"
        if jp2_file.exists():
            converter.convert_jp2_to_jpeg(jp2_file)

    # Extract grammar rules
    print("\nStep 2: Extracting grammar rules...")

    all_extractions = []

    for page_num in range(start_page, end_page + 1):
        print(f"\nPage {page_num}/{end_page}")

        image_path = processed_dir / f"grammardictionar00riggrich_{page_num:04d}.jpg"

        if not image_path.exists():
            print("  WARNING: Image not found, skipping")
            continue

        try:
            extraction = extract_grammar_page_with_claude(image_path, page_num)

            # Save individual page
            output_file = output_dir / f"grammar_page_{page_num:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction, f, indent=2, ensure_ascii=False)

            print(f"  Extracted {len(extraction.get('grammar_rules', []))} rules")
            print(f"  Found {len(extraction.get('interlinear_texts', []))} interlinear texts")
            print(f"  Saved to {output_file}")

            all_extractions.append(extraction)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save combined output
    combined_file = output_dir / f"grammar_combined_{start_page}-{end_page}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            "page_range": f"{start_page}-{end_page}",
            "total_pages": len(all_extractions),
            "pages": all_extractions
        }, f, indent=2, ensure_ascii=False)

    # Statistics
    total_rules = sum(len(e.get('grammar_rules', [])) for e in all_extractions)
    total_examples = sum(
        sum(len(r.get('examples', [])) for r in e.get('grammar_rules', []))
        for e in all_extractions
    )
    total_interlinear = sum(len(e.get('interlinear_texts', [])) for e in all_extractions)

    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nPages processed: {len(all_extractions)}")
    print(f"Grammar rules: {total_rules}")
    print(f"Example sentences: {total_examples}")
    print(f"Interlinear texts: {total_interlinear}")
    print(f"\nOutput directory: {output_dir}/")
    print(f"Combined file: {combined_file}")
    print("\nNext step:")
    print("  python organize_grammar_for_rl.py --input data/grammar_extracted/")
    print()


if __name__ == "__main__":
    main()
