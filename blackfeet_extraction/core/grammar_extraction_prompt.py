"""
Specialized Extraction Prompt for Dakota Grammar Pages (1-88)

This prompt is optimized for extracting testable grammar rules
for RL training with PrimeIntellect verifiers.

Key differences from dakota_extraction_prompt.py:
- Focus on extractable grammar rules
- Morphological transformation patterns
- Verification criteria for RL tasks
- Multi-turn dialogue examples
"""

GRAMMAR_EXTRACTION_PROMPT = """You are analyzing a page from the **Dakota Grammar** section (pages 1-88) of a historical 1890s linguistics text.

## Your Mission: Extract Testable Grammar Rules for RL Training

Your goal is to extract grammar rules in a format that can be used to generate **Reinforcement Learning training tasks** for teaching language models Dakota morphology, syntax, and translation.

## Critical: Dakota Orthography (Same as Before)

The Dakota language uses **special characters** that MUST be preserved exactly:

### Standard Characters (Must Preserve Exactly)
1. **Glottal stop:** ʼ (modifier letter apostrophe, U+02BC)
2. **Acute accents:** á, é, í, ó, ú
3. **Caron diacritics:** č, š, ž
4. **Eng:** ŋ (represents ng sound)
5. **Dotted characters:** ḣ, ṡ, ė
6. **Hyphens:** Mark syllable breaks (preserve ALL)

## What to Extract

### 1. Grammar Rules

For each grammar rule you identify:
- **Rule name**: Brief descriptive name
- **Pattern**: Formal transformation (e.g., "{root} + -ku → his/her {root}")
- **Description**: What the rule does
- **Examples**: Concrete transformations showing the rule
- **Constraints**: When does this rule apply?
- **Verification criteria**: How to check if rule is applied correctly?

### 2. Morphological Transformations

For each example of a morphological change:
- **Base form**: Root word
- **Transformed form**: After applying affixes
- **Affixes**: What was added (prefixes/suffixes)
- **Meaning change**: How meaning changed
- **Special characters**: Which special chars appear

### 3. Interlinear Translations

For interlinear examples (common in grammar texts):
- **Dakota text**: Full sentence with all diacritics
- **Word glosses**: Word-by-word English
- **Full translation**: Complete English sentence
- **Grammatical annotations**: Any morphological notes

## Output Format: JSON Schema

```json
{
  "page_metadata": {
    "page_number": null,
    "chapter": "Chapter name",
    "section_title": "Section heading",
    "quality_issues": "any image quality problems"
  },
  "grammar_rules": [
    {
      "rule_id": "auto-generated",
      "rule_type": "morphology | syntax | phonology | semantics | orthography",
      "rule_name": "Third-person possessive -ku",
      "description": "Add -ku suffix to nouns to indicate 'his/her/its'",
      "pattern": "{noun} + -ku → {noun}-ku",
      "constraints": [
        "Primarily used with kinship terms",
        "May trigger vowel harmony"
      ],
      "exceptions": ["List any exceptions to the rule"],
      "verification_criteria": [
        "Suffix -ku must be present",
        "Special characters preserved",
        "Meaning includes possessive marker"
      ],
      "transformations": [
        {
          "base_form": "iŋhiŋ",
          "transformed_form": "éiŋhiŋtku",
          "affixes": ["é-", "-ku"],
          "gloss_base": "son",
          "gloss_transformed": "his son",
          "special_chars": ["ŋ"],
          "phonological_changes": "é- prefix added for third person"
        }
      ],
      "difficulty": "basic | intermediate | advanced | expert",
      "testable": true
    }
  ],
  "interlinear_examples": [
    {
      "dakota_text": "Wićašta wań éiŋhiŋtku nonpa",
      "dakota_words": ["Wićašta", "wań", "éiŋhiŋtku", "nonpa"],
      "word_glosses": ["Man", "a", "son-his", "two"],
      "morpheme_breakdown": [
        ["Wićašta"],
        ["wań"],
        ["é-", "iŋhiŋ", "-ku"],
        ["nonpa"]
      ],
      "english_translation": "A man had two sons",
      "special_characters_found": ["ć", "š", "ŋ"]
    }
  ],
  "linguistic_notes": "Additional observations about Dakota grammar patterns",
  "extraction_confidence": 0.95
}
```

## Key Differences from Dictionary Extraction

1. **Focus on Rules, Not Definitions**
   - Dictionary: "word → meaning"
   - Grammar: "pattern → transformation → examples"

2. **Testability is Critical**
   - Each rule should be convertible to an RL task
   - Include verification criteria (how to check if correct)

3. **Morpheme Breakdown**
   - Split words into morphemes when possible
   - Show affixes explicitly: ["é-", "iŋhiŋ", "-ku"]

4. **Multiple Examples per Rule**
   - Provide 3-5+ examples of each transformation
   - Show edge cases and exceptions

## Example Extraction

**Input (Grammar Text):**
```
§ 45. Possessive Forms

The third person possessive is formed by adding -ku to the noun stem.

iŋhiŋ, son → éiŋhiŋtku, his son
atkuku, father → atkuku, his father
hunku, mother → hunku, his mother

Note: The prefix é- is often added with -ku for kinship terms.
```

**Output (JSON):**
```json
{
  "grammar_rules": [
    {
      "rule_name": "Third-person possessive -ku",
      "rule_type": "morphology",
      "description": "Form third-person possessive by adding -ku suffix to noun stem",
      "pattern": "{noun} + -ku → his/her {noun}",
      "constraints": [
        "Primarily kinship terms",
        "Often accompanied by é- prefix"
      ],
      "verification_criteria": [
        "Contains -ku suffix",
        "Meaning includes 'his/her'",
        "Special characters preserved"
      ],
      "transformations": [
        {
          "base_form": "iŋhiŋ",
          "transformed_form": "éiŋhiŋtku",
          "affixes": ["é-", "-ku"],
          "gloss_base": "son",
          "gloss_transformed": "his son",
          "special_chars": ["ŋ"]
        },
        {
          "base_form": "atkuku",
          "transformed_form": "atkuku",
          "affixes": ["-ku"],
          "gloss_base": "father",
          "gloss_transformed": "his father",
          "special_chars": []
        }
      ],
      "difficulty": "basic",
      "testable": true
    }
  ]
}
```

## RL Task Generation Guidelines

For each grammar rule, consider:

### Task Type 1: Direct Application
- **Prompt**: "Apply possessive -ku to the word 'iŋhiŋ' (son)"
- **Expected Answer**: "éiŋhiŋtku"
- **Verifiable**: Yes (exact string match + special chars)

### Task Type 2: Translation
- **Prompt**: "Translate 'his son' to Dakota"
- **Expected Answer**: "éiŋhiŋtku"
- **Verifiable**: Yes (meaning + morphology)

### Task Type 3: Rule Identification
- **Prompt**: "What morphological rule creates 'éiŋhiŋtku' from 'iŋhiŋ'?"
- **Expected Answer**: "Third-person possessive with -ku suffix and é- prefix"
- **Verifiable**: Partial (semantic matching)

### Task Type 4: Error Correction
- **Prompt**: "Fix this Dakota word: 'inhintku' (should be 'his son')"
- **Expected Answer**: "éiŋhiŋtku"
- **Verifiable**: Yes (character accuracy critical)

### Task Type 5: Multi-turn Dialogue
- **Turn 1**: "Translate 'son' to Dakota" → "iŋhiŋ"
- **Turn 2**: "Now make it possessive (his son)" → "éiŋhiŋtku"
- **Verifiable**: Yes (progressive building)

## Difficulty Classification

**Basic**: Single transformation, common characters
- Example: "iŋhiŋ + -ku → éiŋhiŋtku"

**Intermediate**: Multiple affixes, special characters
- Example: "ki-ći-ća-šta-ka" (smile for one)

**Advanced**: Complex morphology, multiple rules interact
- Example: Verb conjugations with person/number/tense

**Expert**: Rare constructions, edge cases, phonological changes
- Example: Irregular forms, dialect variations

## Extraction Priorities

1. **Morphological rules** (affixation, compounding)
2. **Interlinear translations** (word-by-word alignment)
3. **Syntactic patterns** (word order, clause structure)
4. **Phonological rules** (sound changes, vowel harmony)
5. **Semantic patterns** (meaning composition)

## Quality Assurance

- Mark `testable: false` if rule is too abstract/vague
- Include `exceptions` if any edge cases mentioned
- Lower `extraction_confidence` if text is unclear
- List all special characters found
- Preserve ALL diacritics and hyphens

## Final Reminders

1. **CHARACTER ACCURACY IS CRITICAL**: ć ≠ c, š ≠ s, ŋ ≠ n
2. **Every rule needs examples**: No examples = not testable
3. **Verification criteria**: How do we check correctness in RL?
4. **Difficulty levels**: Helps build curriculum
5. **Return ONLY JSON**: No markdown code blocks, no extra text

Begin extraction now.
"""


def build_grammar_extraction_prompt(page_context: str = "") -> str:
    """
    Build the Dakota grammar extraction prompt with optional page-specific context.

    Args:
        page_context: Additional context about this specific page
                     (e.g., "Chapter III: Verb Conjugation")

    Returns:
        Complete extraction prompt for grammar pages
    """
    prompt = GRAMMAR_EXTRACTION_PROMPT

    if page_context:
        prompt += f"\n\n## Page-Specific Context\n{page_context}\n"

    return prompt


def build_focused_rule_extraction_prompt(rule_type: str, focus_area: str = "") -> str:
    """
    Build a focused prompt for extracting a specific type of grammar rule.

    Args:
        rule_type: One of "morphology", "syntax", "phonology", "semantics"
        focus_area: Specific focus (e.g., "verb conjugation", "possessive forms")

    Returns:
        Focused extraction prompt
    """
    focus_instructions = {
        "morphology": """
**FOCUS: Morphological Transformations**

Extract all:
- Prefix additions (ki-, wa-, é-, etc.)
- Suffix additions (-ku, -pi, -wo, etc.)
- Infixation patterns
- Reduplication rules
- Compounding rules

For each transformation, provide:
- Base form
- Transformed form
- Step-by-step morpheme breakdown
- Meaning change
""",
        "syntax": """
**FOCUS: Syntactic Patterns**

Extract all:
- Word order rules (SOV, SVO, etc.)
- Clause structure
- Question formation
- Negation patterns
- Subordinate clause markers

For each pattern, provide:
- Template structure
- Example sentences
- Word-by-word gloss
- Constraints/conditions
""",
        "phonology": """
**FOCUS: Phonological Rules**

Extract all:
- Vowel harmony patterns
- Consonant alternations
- Syllable structure rules
- Stress/pitch patterns
- Sound changes in morphology

For each rule, provide:
- Phonological environment
- Sound change pattern
- Multiple examples
- Exceptions
""",
        "semantics": """
**FOCUS: Semantic Patterns**

Extract all:
- Meaning composition rules
- Derivational semantics
- Metaphorical extensions
- Semantic fields
- Polysemy patterns

For each pattern, provide:
- Semantic structure
- Meaning relationships
- Usage contexts
- Examples
"""
    }

    rule_instruction = focus_instructions.get(rule_type, "")

    prompt = GRAMMAR_EXTRACTION_PROMPT.replace(
        "## What to Extract",
        f"{rule_instruction}\n\n## What to Extract"
    )

    if focus_area:
        prompt += f"\n\n## Specific Focus Area\n{focus_area}\n"

    return prompt


if __name__ == "__main__":
    # Test prompt generation
    print("="*80)
    print("GENERAL GRAMMAR EXTRACTION PROMPT")
    print("="*80)
    prompt = build_grammar_extraction_prompt(
        page_context="Chapter III: Possessive Forms and Pronouns"
    )
    print(prompt[:500] + "...\n")

    print("="*80)
    print("FOCUSED MORPHOLOGY PROMPT")
    print("="*80)
    focused = build_focused_rule_extraction_prompt(
        rule_type="morphology",
        focus_area="Possessive suffix -ku and its variants"
    )
    print(focused[:500] + "...")
