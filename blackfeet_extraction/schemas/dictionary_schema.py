"""
Dakota/Blackfeet Dictionary Entry Schema

This defines the precise structure we expect from dictionary entries
based on the 1890 Dakota-English Dictionary format by Stephen Return Riggs.

Entry Format Analysis:
- Headword in bold with syllable breaks (hyphens)
- Part of speech abbreviations (v., n., a., adv., etc.)
- Etymology markers ("of X" showing derivation)
- English definitions (italic in original)
- Inflected forms prefixed with em-dash (—)
- Cross-references ("See X")
"""

from typing import List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class DictionaryEntry:
    """
    Single dictionary entry with all linguistic metadata.

    This schema is designed for training language models with rich
    linguistic context beyond simple word-to-word translation.
    """

    # Core identification (required fields first)
    entry_id: str  # Unique identifier (e.g., "page_042_entry_015")
    headword: str  # Main Dakota/Blackfeet word (e.g., "ki'-ći-ća-šta-ka")
    definition_primary: str  # Main English meaning (required)
    column: int  # 1 or 2 (left or right column)
    page_number: int  # Page number
    source_image: str  # Filename of source image

    # Linguistic classification (optional fields after required)
    part_of_speech: Optional[str] = None  # v., n., a., adv., recip., pos.
    pos_full: Optional[str] = None  # Full expansion: "verb", "noun", etc.

    # Etymology and derivation
    derived_from: Optional[str] = None  # Root word (e.g., "kaštaka")
    derivation_type: Optional[str] = None  # "of", "from", "see"
    root_meaning: Optional[str] = None  # What the root means

    # English translation
    definition_secondary: Optional[str] = None  # Alternative meanings

    # Grammatical information
    inflected_forms: Optional[List[str]] = None  # Conjugations, plurals, etc.
    grammatical_notes: Optional[str] = None  # Any grammar explanations

    # Usage and examples
    usage_notes: Optional[str] = None  # Context, usage patterns
    example_phrases: Optional[List[str]] = None  # Example sentences if present

    # Cross-references
    see_also: Optional[List[str]] = None  # Related words to look up
    compare_with: Optional[List[str]] = None  # Similar/contrasting words

    # Extraction metadata
    line_number: Optional[int] = None  # Approximate line on page
    confidence: float = 1.0  # Extraction confidence (0.0-1.0)
    extraction_notes: Optional[str] = None  # Any issues during extraction

    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.inflected_forms is None:
            self.inflected_forms = []
        if self.example_phrases is None:
            self.example_phrases = []
        if self.see_also is None:
            self.see_also = []
        if self.compare_with is None:
            self.compare_with = []

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent=2):
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_translation_pair(self) -> dict:
        """
        Convert to simple translation pair for basic MT training.

        Returns:
            {"source": "dakota_word", "target": "english_definition"}
        """
        return {
            "source": self.headword,
            "target": self.definition_primary,
            "metadata": {
                "pos": self.part_of_speech,
                "entry_id": self.entry_id,
            }
        }

    def to_instruction_format(self) -> dict:
        """
        Convert to instruction-following format for fine-tuning.

        Returns format suitable for LLaMA, Qwen, etc.
        """
        # Build context
        context = f"Part of speech: {self.part_of_speech or 'unknown'}"
        if self.derived_from:
            context += f"\nDerived from: {self.derived_from}"
        if self.inflected_forms:
            context += f"\nForms: {', '.join(self.inflected_forms)}"

        return {
            "instruction": f"Translate this Dakota word to English: {self.headword}",
            "input": context,
            "output": self.definition_primary,
        }


# Part of speech abbreviation mappings
POS_ABBREVIATIONS = {
    "v.": "verb",
    "v. a.": "verb active",
    "v. n.": "verb neuter",
    "v. recip.": "verb reciprocal",
    "n.": "noun",
    "a.": "adjective",
    "adv.": "adverb",
    "prep.": "preposition",
    "conj.": "conjunction",
    "pron.": "pronoun",
    "interj.": "interjection",
    "pos.": "possessive",
    "part.": "participle",
    "p.": "plural",
}


def expand_pos(abbreviation: str) -> Optional[str]:
    """Expand part of speech abbreviation to full form."""
    return POS_ABBREVIATIONS.get(abbreviation)


# Example entries for testing/validation
EXAMPLE_ENTRIES = [
    {
        "headword": "ki'-ći-ća-šta-ka",
        "part_of_speech": "v.",
        "derived_from": "kaštaka",
        "definition_primary": "to smile for one",
        "inflected_forms": ["wećićaštaka", "uŋkićićaštakapį"],
    },
    {
        "headword": "ki'-ći-ća-wo-ta",
        "part_of_speech": "n.",
        "definition_primary": "one of the same age",
    },
    {
        "headword": "ki'-ći-ća-zu-ta",
        "part_of_speech": "v.",
        "derived_from": "kazuŋta",
        "definition_primary": "to weave for one",
        "inflected_forms": ["wećićazuŋta", "mićićazuŋta"],
    },
]


def validate_entry(entry: DictionaryEntry) -> tuple[bool, List[str]]:
    """
    Validate a dictionary entry.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Required fields
    if not entry.headword:
        issues.append("Missing headword")
    if not entry.definition_primary:
        issues.append("Missing primary definition")
    if not entry.entry_id:
        issues.append("Missing entry ID")

    # Validate confidence
    if not 0.0 <= entry.confidence <= 1.0:
        issues.append(f"Invalid confidence: {entry.confidence}")

    # Warn if no POS
    if not entry.part_of_speech:
        issues.append("Warning: No part of speech specified")

    # Warn if low confidence
    if entry.confidence < 0.7:
        issues.append(f"Low confidence extraction: {entry.confidence}")

    return len(issues) == 0, issues


if __name__ == "__main__":
    # Example usage
    entry = DictionaryEntry(
        entry_id="page_042_entry_001",
        headword="ki'-ći-ća-šta-ka",
        part_of_speech="v.",
        pos_full="verb",
        derived_from="kaštaka",
        definition_primary="to smile for one",
        inflected_forms=["wećićaštaka", "uŋkićićaštakapį"],
        column=1,
        page_number=42,
        source_image="grammardictionar00riggrich_0042.jp2",
        confidence=0.95,
    )

    print("Dictionary Entry:")
    print(entry.to_json())

    print("\n\nTranslation Pair:")
    print(json.dumps(entry.to_translation_pair(), indent=2))

    print("\n\nInstruction Format:")
    print(json.dumps(entry.to_instruction_format(), indent=2))

    print("\n\nValidation:")
    is_valid, issues = validate_entry(entry)
    print(f"Valid: {is_valid}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
