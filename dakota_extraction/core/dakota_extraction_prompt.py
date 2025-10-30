"""
Specialized Extraction Prompt for Dakota Grammar & Dictionary

This prompt is crafted to extract structured linguistic data from the
Dakota Grammar and Dictionary (1890s) with precise character recognition.
"""

DAKOTA_EXTRACTION_PROMPT = """You are analyzing a page from a **Dakota Grammar and Dictionary** from the 1890s.

## Critical: Dakota Orthography

The Dakota language uses **special characters** that MUST be preserved exactly:

### Standard Characters (Must Preserve Exactly)
1. **Glottal stop:** ʼ (modifier letter apostrophe, looks like ' but is U+02BC)
2. **Acute accents for pitch:** á, é, í, ó, ú
3. **Caron/háček diacritics:** č, š, ž (c-caron, s-caron, z-caron)
4. **Special consonants:** ŋ (eng - like ng sound)
5. **Dotted characters:** ḣ, ṡ, ė (h/s/e with dot above)
6. **Long vowels:** aa, ii, oo (doubled letters - NOT macrons)
7. **Digraphs:** ts, ks (treat as single sound units)

### Common Dakota Words You'll See
- Wićášta, wašte, waéa (with c-acute, s-caron)
- éiŋhiŋtku, toŋaŋa (with eng ŋ)
- Wióni, óni (with o-acute)
- mićú-wo (with c-acute, u-acute)

## Document Structure

This is an **interlinear translation** format where:

1. **Dakota text line:** Original language with syllable breaks (hyphens)
2. **English gloss line:** Word-by-word translation in *italics* (in original)
3. **Full translation:** Complete English sentence

### Example Format
```
Wićášta wańŋ éiŋhiŋtku nonpa : unkań hakakata kiŋ he atkuku kiŋ
Man     a    son-his    two :  and   youngest  the  that father-his the

heéiya : Ate, woyuha mitawa kte éiŋ he mićú-wo,
said-to-him : Father, goods mine will-be the that me-mine-give, he-said.
```

## Your Task

Extract the Dakota linguistic data with MAXIMUM character precision:

```json
{
  "dakota_text": "exact Dakota sentence with ALL diacritics",
  "word_glosses": ["word", "by", "word", "English", "glosses"],
  "english_translation": "complete English translation",
  "linguistic_notes": "any grammatical annotations or notes",
  "special_characters_found": ["list", "of", "special", "chars", "like", "ć", "š", "ŋ"],
  "confidence": 0.0-1.0,
  "extraction_notes": "note any uncertainty or damaged text"
}
```

## Critical Instructions

1. **Character Preservation - THIS IS CRITICAL:**
   - **ć** (c-acute): Wićášta, mićú, etc.
   - **š** (s-caron): wašte, Wićášta, etc.
   - **ŋ** (eng): éiŋhiŋtku, toŋaŋa, etc.
   - **ó** (o-acute): Wióni, etc.
   - **ḣ** (h-dot): Any h with dot above
   - **ṡ** (s-dot): Any s with dot above
   - **Hyphens**: Preserve all syllable breaks (é-iŋ-hiŋ-tku)
   - **Apostrophes**: Distinguish regular apostrophe from glottal stop (ʼ)

2. **Extract Interlinear Structure:**
   - Line 1: Dakota words (often with hyphens showing syllables)
   - Line 2: English glosses (italicized in original)
   - Line 3: Full English translation

3. **Handle Multi-Line Entries:**
   - Some translations span multiple lines
   - Maintain continuity across line breaks
   - Preserve punctuation (colons, semicolons, commas)

4. **Identify Linguistic Patterns:**
   - Prefixes/suffixes (ki-, -iŋtku, etc.)
   - Grammatical particles (unkań, ka, he)
   - Possessive markers (-his, -wo)

5. **Quality Assurance:**
   - If a character looks unusual, include it in special_characters_found
   - If text is blurry/damaged: lower confidence score
   - Note any ambiguous characters in extraction_notes

6. **Think Step-by-Step:**
   - First: Identify the Dakota text line(s)
   - Second: Extract word-by-word glosses
   - Third: Extract full English translation
   - Fourth: Verify ALL special characters are preserved

## Output Format

Return a JSON object with this structure:

```json
{
  "page_metadata": {
    "page_number": null,
    "chapter": "Chapter name if visible",
    "section_title": "Section heading if present",
    "quality_issues": "any damage, bleed-through, etc."
  },
  "interlinear_entries": [
    {
      "entry_id": "auto-generated later",
      "dakota_text": "Wićášta wańŋ éiŋhiŋtku nonpa",
      "word_glosses": ["Man", "a", "son-his", "two"],
      "english_translation": "A man had two sons",
      "linguistic_notes": "Parable structure, subject-verb-object order",
      "special_characters_found": ["ć", "š", "ŋ"],
      "confidence": 0.95,
      "extraction_notes": null
    }
  ],
  "vocabulary_items": [
    {
      "dakota_word": "Wićášta",
      "gloss": "man",
      "grammatical_info": "noun",
      "special_chars": ["ć", "š"]
    }
  ]
}
```

## Examples

**Example 1: Simple Interlinear**
```
Wićášta wańŋ éiŋhiŋtku nonpa
Man     a    son-his    two
A man had two sons
```

Extracted as:
```json
{
  "dakota_text": "Wićášta wańŋ éiŋhiŋtku nonpa",
  "word_glosses": ["Man", "a", "son-his", "two"],
  "english_translation": "A man had two sons",
  "special_characters_found": ["ć", "š", "ŋ"],
  "confidence": 0.98
}
```

**Example 2: Complex with Punctuation**
```
heéiya : Ate, woyuha mitawa kte éiŋ he mićú-wo,    eya.
said-to-him : Father, goods mine will-be the that me-mine-give, he-said.
He said to him: Father, give me the goods that will be mine, he said.
```

Extracted as:
```json
{
  "dakota_text": "heéiya : Ate, woyuha mitawa kte éiŋ he mićú-wo, eya.",
  "word_glosses": ["said-to-him", ":", "Father,", "goods", "mine", "will-be", "the", "that", "me-mine-give,", "he-said."],
  "english_translation": "He said to him: Father, give me the goods that will be mine, he said.",
  "special_characters_found": ["ć", "ú"],
  "confidence": 0.95
}
```

## Final Reminders

- **CHARACTER ACCURACY IS CRITICAL:** ć ≠ c, š ≠ s, ŋ ≠ n
- Preserve ALL hyphens (syllable breaks)
- Preserve ALL diacritics (accents, dots, carons)
- List every special character you find
- Mark confidence honestly
- Return ONLY the JSON output, no other text

Begin extraction now.
"""


def build_dakota_extraction_prompt(page_context: str = "") -> str:
    """
    Build the Dakota extraction prompt with optional page-specific context.

    Args:
        page_context: Additional context about this specific page

    Returns:
        Complete extraction prompt
    """
    prompt = DAKOTA_EXTRACTION_PROMPT

    if page_context:
        prompt += f"\n\n## Page-Specific Context\n{page_context}\n"

    return prompt


if __name__ == "__main__":
    # Test prompt generation
    prompt = build_dakota_extraction_prompt(
        page_context="This page contains Chapter IX: Interlinear Translations. The Parable of the Prodigal Son."
    )
    print(prompt)
