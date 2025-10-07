"""
Advanced Dictionary Page Processor with Linguistic Schema

This version uses the specialized Dakota dictionary schema and
extraction prompt for maximum precision.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.examples.openrouter_integration import Qwen3VLClient
from blackfeet_extraction.schemas.dictionary_schema import (
    DictionaryEntry,
    validate_entry,
    expand_pos,
)
from blackfeet_extraction.core.extraction_prompt import build_extraction_prompt


class AdvancedPageProcessor:
    """Process dictionary pages with full linguistic structure extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/extracted",
        reasoning_dir: str = "data/reasoning_traces",
    ):
        """Initialize processor."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")

        self.client = Qwen3VLClient(self.api_key)
        self.output_dir = Path(output_dir)
        self.reasoning_dir = Path(reasoning_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(
        self,
        image_path: Path,
        page_number: int,
        thinking_budget: int = 6000,  # Higher for complex extraction
        page_context: str = "",
    ) -> Dict[str, Any]:
        """
        Extract structured dictionary entries from a page.

        Args:
            image_path: Path to page image
            page_number: Page number
            thinking_budget: Reasoning tokens (higher = more careful)
            page_context: Optional context about this page

        Returns:
            Extraction with DictionaryEntry objects
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*70}")
        print(f"Processing Page {page_number}: {image_path.name}")
        print(f"{'='*70}")

        # Build specialized prompt
        prompt = build_extraction_prompt(page_context)

        print(f"Analyzing with thinking budget: {thinking_budget} tokens...")
        print("Using Dakota dictionary specialized prompt...")

        # Call Qwen3-VL with high thinking budget for accuracy
        response = self.client.analyze_image(
            image_path,
            prompt,
            thinking_budget=thinking_budget,
            temperature=0.2,  # Low for consistency
            max_tokens=16000,  # Allow for many entries
        )

        # Parse response
        extraction = self._parse_response(
            response["text"],
            page_number,
            image_path.name,
        )

        # Add metadata
        extraction["metadata"] = {
            "page_number": page_number,
            "image_path": str(image_path),
            "processed_at": datetime.now().isoformat(),
            "reasoning_tokens": response.get("reasoning_tokens"),
            "total_tokens": response.get("usage", {}).get("total_tokens"),
            "thinking_budget": thinking_budget,
        }

        # Validate entries
        self._validate_extraction(extraction)

        # Save reasoning trace
        if response.get("reasoning"):
            self._save_reasoning_trace(
                page_number,
                response["reasoning"],
                extraction,
            )

        # Save extraction
        self._save_extraction(page_number, extraction)

        print(f"✓ Extracted {len(extraction.get('entries', []))} entries")
        print(f"✓ Reasoning tokens: {response.get('reasoning_tokens', 'N/A')}")

        return extraction

    def _parse_response(
        self,
        response_text: str,
        page_number: int,
        source_image: str,
    ) -> Dict[str, Any]:
        """Parse model response into structured format."""
        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            raw_data = json.loads(json_text)

            # Convert to DictionaryEntry objects
            entries = []
            for i, entry_data in enumerate(raw_data.get("entries", [])):
                # Generate entry ID
                entry_data["entry_id"] = f"page_{page_number:03d}_entry_{i+1:03d}"
                entry_data["page_number"] = page_number
                entry_data["source_image"] = source_image

                # Expand POS if abbreviated
                if entry_data.get("part_of_speech"):
                    entry_data["pos_full"] = expand_pos(
                        entry_data["part_of_speech"]
                    )

                # Create DictionaryEntry object
                try:
                    entry = DictionaryEntry(**entry_data)
                    entries.append(entry.to_dict())
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not create entry {i+1}: {e}")
                    # Keep raw data
                    entries.append(entry_data)

            return {
                "page_metadata": raw_data.get("page_metadata", {}),
                "entries": entries,
                "raw_response": response_text,
            }

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print(f"Raw response:\n{response_text[:500]}...")
            return {
                "page_metadata": {"error": str(e)},
                "entries": [],
                "raw_response": response_text,
            }

    def _validate_extraction(self, extraction: Dict[str, Any]) -> None:
        """Validate extracted entries and print warnings."""
        print(f"\n{'─'*70}")
        print("VALIDATION")
        print(f"{'─'*70}")

        total = len(extraction.get("entries", []))
        valid_count = 0
        warnings = []

        for entry_dict in extraction.get("entries", []):
            # Convert back to DictionaryEntry for validation
            try:
                entry = DictionaryEntry(**entry_dict)
                is_valid, issues = validate_entry(entry)

                if is_valid:
                    valid_count += 1
                else:
                    for issue in issues:
                        warnings.append(f"{entry.headword}: {issue}")

            except Exception as e:
                warnings.append(f"Entry validation error: {e}")

        print(f"Total entries: {total}")
        print(f"Valid entries: {valid_count}")
        print(f"Warnings: {len(warnings)}")

        if warnings:
            print("\nValidation Issues:")
            for warning in warnings[:10]:  # Show first 10
                print(f"  ⚠️  {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

    def _save_extraction(self, page_number: int, extraction: Dict[str, Any]) -> None:
        """Save extraction to disk."""
        output_path = self.output_dir / f"page_{page_number:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to: {output_path}")

    def _save_reasoning_trace(
        self,
        page_number: int,
        reasoning: str,
        extraction: Dict[str, Any],
    ) -> None:
        """Save reasoning trace."""
        trace_path = self.reasoning_dir / f"page_{page_number:03d}_reasoning.json"
        trace = {
            "page_number": page_number,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "num_entries": len(extraction.get("entries", [])),
            "page_metadata": extraction.get("page_metadata", {}),
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved reasoning to: {trace_path}")

    def display_sample_entries(self, extraction: Dict[str, Any], num: int = 3) -> None:
        """Display sample entries for review."""
        entries = extraction.get("entries", [])
        if not entries:
            print("\nNo entries to display")
            return

        print(f"\n{'═'*70}")
        print(f"SAMPLE ENTRIES (showing {min(num, len(entries))} of {len(entries)})")
        print(f"{'═'*70}")

        for entry_dict in entries[:num]:
            print(f"\n{entry_dict.get('headword', 'N/A')}")
            print(f"  POS: {entry_dict.get('part_of_speech', 'N/A')}")
            if entry_dict.get("derived_from"):
                print(f"  From: {entry_dict['derived_from']}")
            print(f"  Definition: {entry_dict.get('definition_primary', 'N/A')}")
            if entry_dict.get("inflected_forms"):
                print(f"  Forms: {', '.join(entry_dict['inflected_forms'])}")
            print(f"  Confidence: {entry_dict.get('confidence', 0):.2f}")
            print(f"  Column: {entry_dict.get('column', '?')}")


def main():
    """Example usage."""
    from blackfeet_extraction.tools.image_converter import ImageConverter

    # Step 1: Convert first JP2 file
    print("Step 1: Converting JP2 to JPEG...")
    converter = ImageConverter(
        input_dir="dictionary/grammardictionar00riggrich_jp2",
        output_dir="data/processed_images",
    )

    jp2_files = sorted(
        Path("dictionary/grammardictionar00riggrich_jp2").glob("*.jp2")
    )
    if not jp2_files:
        print("No JP2 files found")
        return

    # Convert first page
    first_image = converter.convert_jp2_to_jpeg(jp2_files[0])

    # Step 2: Extract with advanced processor
    print("\nStep 2: Extracting dictionary entries...")
    processor = AdvancedPageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    extraction = processor.extract_page(
        image_path=first_image,
        page_number=1,
        thinking_budget=6000,
        page_context="First page of dictionary, may contain title/header material",
    )

    # Step 3: Display results
    processor.display_sample_entries(extraction, num=5)


if __name__ == "__main__":
    main()
