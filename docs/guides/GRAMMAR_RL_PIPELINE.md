# Dakota Grammar RL Training Pipeline

## Overview

This pipeline extracts Dakota language grammar rules from pages 1-88 of the 1890 Riggs dictionary and converts them into RL training rules for an agent to learn Dakota grammar.

## Pipeline Status

### Phase 1: Grammar Extraction (IN PROGRESS)
**Status**: Currently running
**Script**: [extract_grammar_pages.py](extract_grammar_pages.py)

```bash
python extract_grammar_pages.py --pages 1-88 --yes
```

**Progress**: Converting images (page 68/88), then will extract with Claude Sonnet 4.5
**Estimated completion**: 2-3 hours
**Output**: [data/grammar_extracted/](data/grammar_extracted/)
- Individual pages: `grammar_page_001.json` through `grammar_page_088.json`
- Combined file: `grammar_combined_1-88.json`

**Extraction Structure**:
```json
{
  "page_number": 20,
  "grammar_rules": [
    {
      "rule_id": "grammar_p20_r1",
      "rule_type": "syntax|morphology|phonology|particles",
      "rule_description": "Description of the rule",
      "dakota_pattern": "Dakota pattern or structure",
      "english_explanation": "English explanation",
      "examples": [
        {
          "dakota": "Dakota text with diacritics",
          "gloss": "word-by-word gloss",
          "english": "English translation",
          "notes": "linguistic notes"
        }
      ],
      "constraints": "Rule constraints",
      "confidence": 0.95
    }
  ],
  "interlinear_texts": [
    {
      "text_id": "interlinear_p20_t1",
      "dakota_lines": ["Line 1", "Line 2"],
      "gloss_lines": ["Gloss 1", "Gloss 2"],
      "english_translation": "Full translation",
      "linguistic_notes": "Notes",
      "confidence": 0.85
    }
  ],
  "linguistic_terms": [
    {
      "term": "Term",
      "definition": "Definition",
      "dakota_examples": ["Example 1", "Example 2"]
    }
  ]
}
```

### Phase 2: Organize Rules for RL (READY)
**Status**: Script created, waiting for extraction to complete
**Script**: [organize_grammar_for_rl.py](organize_grammar_for_rl.py)

```bash
# Run after extraction completes
python organize_grammar_for_rl.py --input data/grammar_extracted/ --min-confidence 0.5
```

**What it does**:
1. Loads all extracted grammar pages
2. Converts each grammar rule to RL training format
3. Organizes rules by linguistic category
4. Generates positive/negative examples
5. Estimates rule difficulty (easy/medium/hard)
6. Creates verification patterns

**Output**: [data/rl_training_rules/](data/rl_training_rules/)
- `rules_morphology.json` - Morphological rules
- `rules_syntax.json` - Syntactic rules
- `rules_phonology.json` - Phonological rules
- `rules_particles.json` - Particle usage rules
- `all_rl_rules.json` - Complete rule set
- `rl_rules_summary.txt` - Human-readable summary

**RL Rule Format**:
```python
{
  "rule_id": "grammar_p20_r1",
  "rule_type": "syntax",
  "rule_name": "Negation with šni",
  "rule_description": "The negative particle šni follows the verb to negate it",
  "dakota_pattern": "[verb] šni",
  "english_explanation": "To negate a verb, place šni after it",

  "positive_examples": [
    {
      "dakota": "waú šni",
      "english": "he does not come",
      "gloss": "come NEG",
      "notes": "Standard negation pattern"
    }
  ],

  "negative_examples": [
    {
      "dakota": "*šni waú",  # Incorrect: NEG before verb
      "english": "[incorrect form]",
      "notes": "Violates: NEG must follow verb"
    }
  ],

  "verification_pattern": "verify_syntax: [verb] šni",
  "source_pages": [20],
  "confidence": 0.9,
  "difficulty": "easy",
  "constraints": "NEG particle follows verb",
  "linguistic_notes": ""
}
```

### Phase 3: Create RL Environment (READY)
**Status**: Script created
**Script**: [create_grammar_rl_environment.py](create_grammar_rl_environment.py)

```bash
# Run after organizing rules
python create_grammar_rl_environment.py --rules-dir data/rl_training_rules/
```

**What it does**:
1. Loads all RL training rules
2. Creates interactive RL environment
3. Defines action space (apply rule, translate, correct error, identify pattern)
4. Implements reward structure
5. Progressive difficulty curriculum
6. Episode management

**Environment Features**:
- **States**: Grammar challenges with Dakota patterns
- **Actions**: Apply rule, translate, correct errors, identify patterns
- **Rewards**:
  - Base reward: 1.0 for valid actions
  - Accuracy bonus: +1.0 for correct pattern application
  - Difficulty multiplier: 1.0x (easy), 1.5x (medium), 2.0x (hard)
  - Category bonus: +0.5 for exploring new rule categories
- **Curriculum**: Progressive difficulty based on agent performance

**Output**: [data/rl_environment/](data/rl_environment/)
- `environment_config.json` - Environment configuration
- Training episodes and statistics

## Rule Categories

Expected categories from grammar extraction:

1. **Morphology**: Word formation, affixes, inflection
   - Verb conjugation patterns
   - Noun declension
   - Pronoun marking
   - Possessive constructions

2. **Syntax**: Sentence structure
   - Word order patterns
   - Negation with šni
   - Question formation
   - Subordinate clauses

3. **Phonology**: Sound patterns
   - Vowel harmony
   - Consonant alternations
   - Syllable structure
   - Pitch/accent patterns (á, é, í, ó, ú)

4. **Particles**: Grammatical particles
   - Discourse markers
   - Evidentials
   - Aspectual particles
   - Connectives

5. **Translation**: Complete sentence patterns
   - Interlinear texts with glosses
   - Full translation examples

## Next Steps

### Immediate (while extraction runs):
1. Monitor extraction progress: `python -c "import os; print(len([f for f in os.listdir('data/grammar_extracted') if f.endswith('.json')]))"`
2. Review sample extractions: `cat data/grammar_extracted/grammar_page_020.json`

### After Extraction Completes:
1. **Organize Rules**:
   ```bash
   python organize_grammar_for_rl.py --input data/grammar_extracted/
   ```

2. **Review Organized Rules**:
   ```bash
   cat data/rl_training_rules/rl_rules_summary.txt
   ```

3. **Create RL Environment**:
   ```bash
   python create_grammar_rl_environment.py --rules-dir data/rl_training_rules/ --demo-episodes 5
   ```

4. **Connect to PrimeIntellect Verifier**:
   - Implement verification functions for each rule type
   - Connect to existing Dakota RL environment at [dakota_language_rl_env/](dakota_language_rl_env/)
   - Integrate with verifier system in [dakota_language_rl_env/verifiers/](dakota_language_rl_env/verifiers/)

5. **Generate Negative Examples**:
   - Create systematic rule violations
   - Pattern-specific error generation
   - Common learner errors

6. **Train RL Agent**:
   - Use rules as training curriculum
   - Progressive difficulty
   - Multi-category learning
   - Transfer to real Dakota text generation

## Integration with Existing RL Environment

The existing Dakota RL environment at [dakota_language_rl_env/](dakota_language_rl_env/) includes:
- PrimeIntellect verifier integration
- Reward system based on grammar correctness
- Observation space with Dakota text encoding
- Action space for text generation

**Integration points**:
1. Load grammar rules into verifier system
2. Use rules to validate agent outputs
3. Provide rule-specific feedback
4. Track learning progress by rule category

## Expected Output Statistics

Based on 88 grammar pages:

- **Estimated grammar rules**: 100-300 rules
  - Morphology: 40-60%
  - Syntax: 20-30%
  - Phonology: 10-20%
  - Particles: 10-20%

- **Estimated examples**: 300-800 examples
  - Positive examples: 500-700
  - Negative examples: To be generated (500-1000)

- **Interlinear texts**: 50-150 complete sentences
  - With word-by-word glosses
  - Full English translations
  - Linguistic annotations

## Quality Thresholds

- **High confidence** (>0.8): Use directly in training
- **Medium confidence** (0.5-0.8): Use with manual review
- **Low confidence** (<0.5): Filter out or flag for review

## File Structure

```
data/
├── grammar_extracted/           # Raw extraction output
│   ├── grammar_page_001.json
│   ├── grammar_page_002.json
│   └── ...
│   └── grammar_combined_1-88.json
│
├── rl_training_rules/          # Organized RL rules
│   ├── rules_morphology.json
│   ├── rules_syntax.json
│   ├── rules_phonology.json
│   ├── rules_particles.json
│   ├── all_rl_rules.json
│   └── rl_rules_summary.txt
│
└── rl_environment/             # Environment config
    └── environment_config.json

Scripts:
├── extract_grammar_pages.py         # Phase 1: Extract from images
├── organize_grammar_for_rl.py       # Phase 2: Organize for RL
└── create_grammar_rl_environment.py # Phase 3: Create environment
```

## Monitoring Extraction Progress

Check number of pages extracted:
```bash
ls data/grammar_extracted/grammar_page_*.json | wc -l
```

View extraction statistics:
```bash
python -c "
import json
from pathlib import Path

files = list(Path('data/grammar_extracted').glob('grammar_page_*.json'))
total_rules = 0
total_interlinear = 0

for f in files:
    with open(f) as fp:
        data = json.load(fp)
    total_rules += len(data.get('grammar_rules', []))
    total_interlinear += len(data.get('interlinear_texts', []))

print(f'Pages: {len(files)}')
print(f'Rules: {total_rules}')
print(f'Interlinear: {total_interlinear}')
"
```

## Cost & Time Estimates

- **Extraction**: ~$22 for 88 pages, ~2-3 hours
- **Organization**: Free, ~1 minute
- **Environment creation**: Free, instant
- **Total**: ~$22, ~3 hours end-to-end

## Notes

- All Dakota special characters are preserved (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú)
- UTF-8 encoding used throughout
- JSON output with `ensure_ascii=False`
- Confidence scores track extraction quality
- Rule IDs are unique and traceable to source pages
