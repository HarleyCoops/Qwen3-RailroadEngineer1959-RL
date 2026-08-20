"""
Specialized Extraction Prompt for Dakota Dictionary

This prompt is carefully crafted to extract structured data from the
1890 Dakota-English Dictionary format with high precision.
"""

DICTIONARY_EXTRACTION_PROMPT = """You are analyzing a page from the **Dakota-English Dictionary** (1890) by Stephen Return Riggs.

## Dictionary Structure

This dictionary follows a **standard two-column format** with these characteristics:

### Entry Format
Each entry contains:
1. **Headword** (in BOLD): The Dakota word with syllable breaks marked by hyphens
   - Example: **ki'-ći-ća-šta-ka**
   - Preserve ALL diacritical marks (accents, special characters)

2. **Part of Speech** (abbreviated): Immediately after headword
   - Common: v. (verb), n. (noun), a. (adjective), adv. (adverb)
   - Complex: v. a. (verb active), v. recip. (verb reciprocal), pos. (possessive)

3. **Etymology/Derivation**: Often shows root word
   - Format: "of [root_word]" or "from [root_word]"
   - Example: "v. of kaštaka"

4. **English Definition** (in italics in original): The meaning
   - May have multiple meanings separated by semicolons
   - Example: "to smile for one"

5. **Inflected Forms** (prefixed with em-dash —): Conjugations/variants
   - Format: "—form1, form2, form3"
   - Example: "—wećićaštaka, uŋkićićaštakapį"

6. **Cross-References**: References to other entries
   - Format: "See [word]" or "Compare [word]"

### Typography Conventions
- **Bold text** = Dakota headword
- *Italic text* (in original) = English definitions
- Em-dash (—) = Introduces inflected forms
- Semicolon (;) = Separates multiple definitions
- Comma (,) = Separates inflected forms

## Your Task

Extract EVERY entry from this page with MAXIMUM PRECISION. For each entry, identify:

```json
{
  "headword": "the Dakota word exactly as printed (with hyphens and diacritics)",
  "part_of_speech": "abbreviation (v., n., a., etc.) or null",
  "derived_from": "root word if present (remove 'of'/'from'), or null",
  "definition_primary": "the main English meaning",
  "definition_secondary": "additional meanings if present, or null",
  "inflected_forms": ["array", "of", "conjugated", "forms"],
  "see_also": ["cross", "referenced", "words"],
  "grammatical_notes": "any special grammar notes, or null",
  "column": 1 or 2 (left=1, right=2),
  "confidence": 0.0-1.0 (how certain you are about this extraction),
  "extraction_notes": "note any uncertainty, damaged text, or ambiguities"
}
```

## Critical Instructions

1. **Preserve Exact Spelling**: Dakota orthography is complex. Copy EXACTLY:
   - All hyphens: ki'-ći-ća-šta-ka (not kicicashtaka)
   - All diacritics: acute accents (á, é), special characters (č, ŋ, š, etc.)
   - Capitalization as printed

2. **Handle Multi-Column Layout**:
   - Process left column (column: 1) first, top to bottom
   - Then right column (column: 2), top to bottom
   - Maintain sequential order within each column

3. **Distinguish Typography**:
   - Bold = headword (the word being defined)
   - Regular = grammatical info, etymology
   - After semicolon/comma usually = inflected forms

4. **Parse Inflected Forms**:
   - Look for em-dash (—) before the forms
   - Split on commas
   - These are NOT separate entries

5. **Handle Ambiguity**:
   - If text is unclear/damaged: lower confidence score
   - If unsure about POS: set to null and note in extraction_notes
   - If entry structure is unusual: document in extraction_notes

6. **Think Step-by-Step**:
   - First, identify where one entry ends and next begins
   - Then parse each component of the entry
   - Verify: does this make linguistic sense?

## Output Format

Return a JSON object with this EXACT structure:

```json
{
  "page_metadata": {
    "columns": 2,
    "layout_notes": "observations about page layout",
    "quality_issues": "any page damage, bleed-through, etc."
  },
  "entries": [
    {
      "entry_id": "auto-generated later",
      "headword": "...",
      "part_of_speech": "...",
      "derived_from": null,
      "definition_primary": "...",
      "definition_secondary": null,
      "inflected_forms": [...],
      "see_also": [],
      "grammatical_notes": null,
      "column": 1,
      "confidence": 0.95,
      "extraction_notes": null
    }
  ]
}
```

## Examples

Here are correctly extracted entries:

**Entry 1:**
```
**ki'-ći-ća-šta-ka**, v. of kaštaka; to smile for one,
  —wećićaštaka, uŋkićićaštakapį.
```

Extracted as:
```json
{
  "headword": "ki'-ći-ća-šta-ka",
  "part_of_speech": "v.",
  "derived_from": "kaštaka",
  "definition_primary": "to smile for one",
  "inflected_forms": ["wećićaštaka", "uŋkićićaštakapį"],
  "column": 1,
  "confidence": 0.98
}
```

**Entry 2:**
```
**ki'-ći-ća-wo-ta**, n. one of the same age.
```

Extracted as:
```json
{
  "headword": "ki'-ći-ća-wo-ta",
  "part_of_speech": "n.",
  "definition_primary": "one of the same age",
  "inflected_forms": [],
  "column": 1,
  "confidence": 1.0
}
```

**Entry 3:**
```
**ki-éin'**, v. of éin; to desire one's own; to desire for one; to desire of one. See ekićin.
```

Extracted as:
```json
{
  "headword": "ki-éin'",
  "part_of_speech": "v.",
  "derived_from": "éin",
  "definition_primary": "to desire one's own",
  "definition_secondary": "to desire for one; to desire of one",
  "see_also": ["ekićin"],
  "column": 2,
  "confidence": 0.95
}
```

## Final Reminder

- Extract EVERY entry on the page
- Maintain precise orthography (diacritics, hyphens)
- Distinguish headwords from inflected forms
- Mark confidence honestly
- Return ONLY the JSON output, no other text

Begin extraction now.
"""


def build_extraction_prompt(page_context: str = "") -> str:
    """
    Build the extraction prompt with optional page-specific context.

    Args:
        page_context: Additional context about this specific page

    Returns:
        Complete extraction prompt
    """
    prompt = DICTIONARY_EXTRACTION_PROMPT

    if page_context:
        prompt += f"\n\n## Page-Specific Context\n{page_context}\n"

    return prompt


if __name__ == "__main__":
    # Test prompt generation
    prompt = build_extraction_prompt(
        page_context="This page appears to contain entries starting with 'ki-'. Some entries may have water damage in the lower right corner."
    )
    print(prompt)
