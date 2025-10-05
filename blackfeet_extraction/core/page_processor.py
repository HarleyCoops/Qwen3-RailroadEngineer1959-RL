"""
Blackfeet Dictionary Page Processor

This module processes individual dictionary pages using Qwen3-VL-235B-A22B-Thinking
to extract structured linguistic data with reasoning traces.

Inspired by the Stoney Nakoda language preservation project by @harleycoops.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from implementation.examples.openrouter_integration import Qwen3VLClient


class PageProcessor:
    """Process individual dictionary pages to extract structured linguistic data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/extracted",
        reasoning_dir: str = "data/reasoning_traces",
    ):
        """
        Initialize the page processor.

        Args:
            api_key: OpenRouter API key (defaults to env var)
            output_dir: Directory for structured extraction output
            reasoning_dir: Directory for reasoning traces
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")

        self.client = Qwen3VLClient(self.api_key)
        self.output_dir = Path(output_dir)
        self.reasoning_dir = Path(reasoning_dir)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(
        self,
        image_path: Path,
        page_number: int,
        thinking_budget: int = 4096,
    ) -> Dict[str, Any]:
        """
        Extract structured data from a single dictionary page.

        Args:
            image_path: Path to dictionary page image
            page_number: Page number for tracking
            thinking_budget: Reasoning token budget (higher = more thorough)

        Returns:
            Structured extraction with entries, metadata, and reasoning
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\n{'='*60}")
        print(f"Processing Page {page_number}: {image_path.name}")
        print(f"{'='*60}")

        # Craft extraction prompt
        prompt = self._build_extraction_prompt()

        # Send to Qwen3-VL with high thinking budget
        print(f"Analyzing page with thinking budget: {thinking_budget} tokens...")
        response = self.client.analyze_image(
            image_path,
            prompt,
            thinking_budget=thinking_budget,
            temperature=0.3,  # Lower for consistency
            max_tokens=8192,  # Allow detailed output
        )

        # Parse response
        extraction = self._parse_response(response["text"], page_number)

        # Add metadata
        extraction["metadata"] = {
            "page_number": page_number,
            "image_path": str(image_path),
            "processed_at": datetime.now().isoformat(),
            "reasoning_tokens": response.get("reasoning_tokens"),
            "total_tokens": response.get("usage", {}).get("total_tokens"),
        }

        # Save reasoning trace
        if response.get("reasoning"):
            self._save_reasoning_trace(page_number, response["reasoning"], extraction)

        # Save extraction
        self._save_extraction(page_number, extraction)

        print(f"✓ Extracted {len(extraction.get('entries', []))} entries")
        print(f"✓ Reasoning tokens used: {response.get('reasoning_tokens', 'N/A')}")

        return extraction

    def _build_extraction_prompt(self) -> str:
        """Build the extraction prompt for the model."""
        return """Analyze this historical Blackfeet language dictionary page from 1890.

**Your Task:**
Extract ALL dictionary entries in structured format. For each entry, identify:

1. **Blackfeet Word** (headword in Blackfeet language)
2. **English Translation** (meaning in English)
3. **Part of Speech** (noun, verb, adjective, etc., if indicated)
4. **Grammatical Info** (any notes about grammar, conjugation, etc.)
5. **Example Sentences** (if present)
6. **Etymology or Notes** (any additional information)

**Important Guidelines:**
- Preserve original spelling exactly as printed
- Handle diacritics and special characters carefully
- Identify the structure: is this single-column or multi-column?
- Track your confidence for each extraction
- Note any damaged, unclear, or ambiguous text
- Preserve the sequential order of entries

**Output Format:**
Return a JSON object with this structure:

```json
{
  "layout": "single-column" or "multi-column",
  "entries": [
    {
      "entry_id": "unique_id",
      "blackfeet": "the Blackfeet word",
      "english": "the English translation",
      "pos": "part of speech or null",
      "grammatical_notes": "any grammar info or null",
      "examples": ["example sentences"],
      "etymology": "etymological notes or null",
      "confidence": 0.0-1.0,
      "notes": "any extraction concerns or null"
    }
  ],
  "page_notes": "overall observations about this page"
}
```

**Think carefully about:**
- Typography conventions (how are headwords vs definitions formatted?)
- Abbreviations used (n. for noun, v. for verb, etc.)
- How to distinguish between different entries
- Whether there are subsections or related entries

Provide ONLY the JSON output, no other text."""

    def _parse_response(self, response_text: str, page_number: int) -> Dict[str, Any]:
        """
        Parse the model's response into structured data.

        Args:
            response_text: Raw response from model
            page_number: Page number for error tracking

        Returns:
            Parsed extraction dictionary
        """
        try:
            # Try to extract JSON from response
            # Look for JSON code block or raw JSON
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

            extraction = json.loads(json_text)

            # Add entry IDs if missing
            for i, entry in enumerate(extraction.get("entries", [])):
                if "entry_id" not in entry:
                    entry["entry_id"] = f"page_{page_number:03d}_entry_{i+1:03d}"

            return extraction

        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Could not parse JSON response: {e}")
            print(f"Raw response:\n{response_text[:500]}...")

            # Return fallback structure
            return {
                "layout": "unknown",
                "entries": [],
                "page_notes": f"JSON parsing failed: {e}",
                "raw_response": response_text,
            }

    def _save_extraction(self, page_number: int, extraction: Dict[str, Any]) -> None:
        """Save extraction to disk."""
        output_path = self.output_dir / f"page_{page_number:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved extraction to: {output_path}")

    def _save_reasoning_trace(
        self,
        page_number: int,
        reasoning: str,
        extraction: Dict[str, Any],
    ) -> None:
        """Save reasoning trace for verification."""
        trace_path = self.reasoning_dir / f"page_{page_number:03d}_reasoning.json"
        trace = {
            "page_number": page_number,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "num_entries_extracted": len(extraction.get("entries", [])),
            "extraction_summary": {
                "layout": extraction.get("layout"),
                "page_notes": extraction.get("page_notes"),
            },
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved reasoning trace to: {trace_path}")

    def batch_extract(
        self,
        image_dir: Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
        thinking_budget: int = 4096,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple pages in batch.

        Args:
            image_dir: Directory containing page images
            start_page: Starting page number
            end_page: Ending page number (None = all)
            thinking_budget: Reasoning budget per page

        Returns:
            List of extractions
        """
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))

        if end_page:
            image_paths = image_paths[start_page - 1 : end_page]
        else:
            image_paths = image_paths[start_page - 1 :]

        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION: {len(image_paths)} pages")
        print(f"{'='*60}\n")

        extractions = []
        for i, image_path in enumerate(image_paths, start=start_page):
            try:
                extraction = self.extract_page(image_path, i, thinking_budget)
                extractions.append(extraction)
            except Exception as e:
                print(f"❌ Error processing page {i}: {e}")
                # Continue with next page
                continue

        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len(extractions)} pages processed")
        print(f"{'='*60}\n")

        return extractions


def main():
    """Example usage."""
    # Initialize processor
    processor = PageProcessor(
        output_dir="data/extracted",
        reasoning_dir="data/reasoning_traces",
    )

    # Example 1: Single page extraction
    example_image = Path("Public/Dictionary.jpeg")
    if example_image.exists():
        extraction = processor.extract_page(
            image_path=example_image,
            page_number=1,
            thinking_budget=4096,  # High budget for thorough analysis
        )

        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Entries extracted: {len(extraction.get('entries', []))}")
        print(f"Layout detected: {extraction.get('layout')}")
        print(f"Page notes: {extraction.get('page_notes')}")

        if extraction.get('entries'):
            print("\nFirst entry:")
            first = extraction['entries'][0]
            print(f"  Blackfeet: {first.get('blackfeet')}")
            print(f"  English: {first.get('english')}")
            print(f"  Confidence: {first.get('confidence')}")

    # Example 2: Batch processing
    # Uncomment when you have multiple pages
    # dictionary_dir = Path("data/dictionary_pages")
    # if dictionary_dir.exists():
    #     processor.batch_extract(
    #         image_dir=dictionary_dir,
    #         start_page=1,
    #         end_page=10,
    #         thinking_budget=4096,
    #     )


if __name__ == "__main__":
    main()
