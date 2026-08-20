# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Vision-Language Models to extract and preserve the **Dakota language** from a historical 1890s grammar and dictionary (Stephen Return Riggs' 665-page Dakota-English Dictionary). The goal is to create high-quality structured datasets for fine-tuning language models on Dakota, supporting Indigenous language revitalization.

**Key Achievement**: Proven that VLMs can extract complex Dakota orthography (ć, š, ŋ, ḣ) from 130-year-old texts without OCR training, achieving 92-95% accuracy through prompt engineering alone.

**Primary VLM**: Claude Sonnet 4.5 (via Anthropic API) - delivers best results for Dakota character preservation
**Secondary VLM**: Qwen3-VL-235B-A22B-Thinking (via OpenRouter) - alternative with reasoning budget support

## Common Development Commands

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
# Create .env file with:
# ANTHROPIC_API_KEY=your_key_here
# OPENROUTER_API_KEY=your_key_here (optional)
```

### Dakota Dictionary Extraction

**Test Extraction (Recommended First Step)**

```bash
# Test on page 89 (first dictionary page) - uses Claude Sonnet 4.5
python test_dakota_claude.py

# Output: data/dakota_test/dakota_extraction_test.json
# Cost: ~$0.03, Time: ~30 seconds
```

**Full Dictionary Extraction**

```bash
# Process dictionary pages 89-440 (352 pages total)
python extract_dakota_dictionary_v2.py --all-dictionary

# Cost: ~$88, Time: ~12 hours
# Output: data/extracted/*.json
```

**Batch Extraction**

```bash
# Process specific page range
python extract_dakota_dictionary_v2.py --pages 89-100

# Process using pipeline with thinking budget
python blackfeet_extraction/run_extraction.py \
    --start-page 89 \
    --end-page 440 \
    --thinking-budget 6000
```

### Build Training Datasets

```bash
# Convert extracted JSON to fine-tuning format
python blackfeet_extraction/datasets/training_dataset_builder.py

# Output: data/training_datasets/
#   - translation_train.jsonl (Dakota↔English pairs)
#   - instruction_dataset.jsonl (instruction-following format)
#   - vocabulary.json (word-level mappings)
```

### Testing and Evaluation

```bash
# Run offline smoke tests (no API calls)
OFFLINE=1 pytest -q

# Run evaluation on extraction quality
python eval/run_eval.py --pred eval/fixtures/sample_predictions.jsonl \
    --truth eval/fixtures/sample_ground_truth.jsonl \
    --out eval/report.md
```

## Architecture

### Directory Structure

- **`blackfeet_extraction/`**: Core extraction pipeline (note: name is historical; processes Dakota language)
  - `core/dakota_extraction_prompt.py`: Specialized prompt for Dakota character preservation
  - `core/claude_page_processor.py`: Claude API wrapper for page processing
  - `core/page_processor.py`: Generic VLM page processor
  - `schemas/dictionary_schema.py`: Data validation schemas (DictionaryEntry dataclass)
  - `tools/image_converter.py`: JP2→JPEG conversion
  - `datasets/training_dataset_builder.py`: Build fine-tuning datasets from extracted JSON
  - `run_extraction.py`: Complete pipeline script

- **Root-level extraction scripts**:
  - `test_dakota_claude.py`: Test extraction using Claude Sonnet 4.5 (recommended)
  - `extract_dakota_dictionary_v2.py`: Main dictionary extraction script (pages 89-440)
  - `extract_dakota_dictionary.py`: Original extraction script
  - `test_dakota_extraction.py`: Various test scripts

- **`dakota_rl_training/`**: Reinforcement Learning training pipeline
  - `verifiers/grammar_env.py`: Multi-turn & single-turn RL environments
  - `verifiers/rubrics.py`: Compositional reward functions for Dakota grammar
  - `configs/training_config.yaml`: GRPO training configuration
  - `datasets/`: Generated RL training tasks (5,657 tasks from 1,036 grammar rules)
  - `train.py`: Launch script for local or distributed training

- **`eval/`**: Extraction quality evaluation framework
  - `score_extraction.py`: Token accuracy, character distance, diacritic preservation metrics
  - `run_eval.py`: CLI wrapper for reproducible evaluation with commit tracking
  - `fixtures/`: Sample ground truth and predictions for testing

- **`data/`**: Extraction outputs (created during extraction)
  - `processed_images/`: Converted JPEG files from JP2 source
  - `extracted/`: Page-level JSON extraction results (page_089.json, page_090.json, etc.)
  - `dakota_test/`: Test extraction output
  - `training_datasets/`: Fine-tuning ready datasets
  - `reasoning_traces/`: VLM reasoning process (when using thinking budget)

- **`Dictionary/grammardictionar00riggrich_jp2/`**: Source images (440 JP2 files)
  - Pages 1-88: Grammar rules and linguistic analysis
  - Pages 89-440: Dictionary entries (extraction target)

- **Root-level pipeline scripts**:
  - `convert_all_images.py`: JP2 → JPEG batch conversion
  - `extract_grammar_pages.py`: Grammar extraction (pages 1-88)
  - `organize_grammar_for_rl.py`: Grammar rules → RL task format
  - `convert_rules_to_primeintellect.py`: Generate 5,657 RL training tasks
  - `create_grammar_rl_environment.py`: Setup RL environment with curriculum learning
  - `run_complete_grammar_pipeline.py`: End-to-end grammar extraction + RL setup

### Key Components

**1. Dakota Extraction Prompt (`blackfeet_extraction/core/dakota_extraction_prompt.py`)**

Specialized prompt engineering for Dakota character preservation:

- Explicitly instructs VLM to preserve: ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú
- Handles interlinear translation format (Dakota text, glosses, English translation)
- Maintains syllable breaks (hyphens) and grammatical structure
- Returns structured JSON with confidence scores
- Function: `build_dakota_extraction_prompt(page_context="")`

**2. Dictionary Schema (`blackfeet_extraction/schemas/dictionary_schema.py`)**

Defines structured output format:

- `DictionaryEntry` dataclass with required fields: headword, definition_primary, entry_id, page_number
- Optional fields: part_of_speech, inflected_forms, etymology, usage_notes
- Validation: `validate_entry()` checks required fields and confidence thresholds
- Export formats: `to_translation_pair()`, `to_instruction_format()` for fine-tuning

**3. Claude Page Processor (`blackfeet_extraction/core/claude_page_processor.py`)**

Wrapper for Anthropic Claude API:

- Uses Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- Handles base64 image encoding
- Applies Dakota extraction prompt
- Parses JSON responses with error handling
- Returns structured dictionary entries

**4. Training Dataset Builder (`blackfeet_extraction/datasets/training_dataset_builder.py`)**

Converts extracted JSON to fine-tuning formats:

- `build_translation_pairs()`: Simple Dakota↔English pairs
- `build_instruction_dataset()`: Instruction-following format (for LLaMA, Qwen, etc.)
- `build_vocabulary_corpus()`: Word-level mappings
- `generate_statistics()`: Quality metrics and coverage analysis

**5. RL Training Components (`dakota_rl_training/`)**

Reinforcement Learning training system built on PrimeIntellect:

- **DakotaGrammarEnv**: Multi-turn environment with progressive feedback (up to 3 turns)
- **DakotaMorphologyEnv**: Single-turn environment for fast verification
- **DakotaGrammarRubric**: Compositional reward functions:
  - Character preservation (40%): Verifies ć, š, ŋ, ḣ preservation
  - Affix accuracy (40%): Checks morphological correctness (-ku, ta-, etc.)
  - Semantic accuracy (20%): Translation quality via word overlap
  - Difficulty multiplier: 1.0x (easy) to 2.0x (hard)
- **Training config**: GRPO algorithm, Qwen2.5-7B-Instruct base, LoRA rank 64
- **Curriculum**: 3-stage progression (easy → medium → hard) with 5,657 total tasks

## Dakota Language Orthography

Dakota uses special characters that MUST be preserved during extraction:

### Critical Characters

1. **Acute accents (pitch)**: á, é, í, ó, ú
2. **Caron diacritics**: č, š, ž (c-caron, s-caron, z-caron)
3. **Eng**: ŋ (represents ng sound)
4. **Dotted consonants**: ḣ, ṡ (h-dot, s-dot)
5. **Syllable breaks**: Hyphens (e.g., é-iŋ-hiŋ-tku)
6. **Glottal stop**: ʼ (modifier letter apostrophe U+02BC)

### Common Dakota Words

- Wićášta (man) - contains ć, š
- éiŋhiŋtku (his son) - contains ŋ
- mićú-wo (give to me) - contains ć, ú

When working with Dakota text:
- Always use UTF-8 encoding (`ensure_ascii=False` in JSON)
- Character accuracy is critical: ć ≠ c, š ≠ s, ŋ ≠ n
- Preserve ALL hyphens (they mark syllable boundaries)

## Dictionary Structure

**Pages 1-88**: Grammar, linguistic analysis, interlinear translations
- Interlinear format: Dakota text line, English gloss line, full translation
- Complex multi-line entries with grammatical annotations
- Use `DAKOTA_EXTRACTION_PROMPT` from `dakota_extraction_prompt.py`

**Pages 89-440**: Dictionary entries (352 pages - PRIMARY EXTRACTION TARGET)
- Two-column format
- Headwords with syllable breaks (hyphens)
- Part of speech abbreviations (v., n., a., etc.)
- Etymology markers ("of X")
- Inflected forms prefixed with em-dash (—)
- ~10,000-15,000 total entries estimated

## Extraction Pipeline Workflow

1. **Convert images**: JP2 → JPEG (requires Pillow with OpenJPEG support)
2. **Extract with VLM**: Send to Claude/Qwen with Dakota-specific prompt
3. **Parse JSON**: Extract structured data with confidence scores
4. **Validate**: Check required fields, character preservation, confidence thresholds
5. **Build datasets**: Convert to fine-tuning formats (translation pairs, instruction-following)

## Important Environment Variables

**Required**:
- `ANTHROPIC_API_KEY`: For Claude Sonnet 4.5 (primary extraction method)

**Optional**:
- `OPENROUTER_API_KEY`: For Qwen3-VL with reasoning budget
- `OPENROUTER_REASONING_MAX_TOKENS`: Default reasoning token budget (e.g., 6000)
- `OPENROUTER_INCLUDE_REASONING`: Include reasoning traces (true/false)

## Extraction Quality Guidelines

**High-quality extractions** (confidence > 0.9):
- All special characters preserved exactly
- Complete definitions extracted
- Clear column/page positioning
- Minimal image quality issues

**Medium-quality extractions** (0.7 < confidence ≤ 0.9):
- Most special characters correct
- Possible minor ambiguities
- Some blur or ink bleed

**Low-quality extractions** (confidence ≤ 0.7):
- Damaged text regions
- Unclear characters
- Should be manually reviewed before using in training data

**Filtering recommendation**: Use only entries with confidence > 0.7 for fine-tuning datasets

## Common Pitfalls

1. **JP2 Image Support**: Requires Pillow with OpenJPEG libraries installed
   - Windows: Download from https://www.openjpeg.org/
   - Linux: `sudo apt-get install libopenjp2-7`
   - Mac: `brew install openjpeg`

2. **Unicode Encoding**: Always save JSON with `ensure_ascii=False` to preserve Dakota characters
   - Wrong: `json.dump(data, f)` → converts ć to \u0107
   - Right: `json.dump(data, f, ensure_ascii=False)` → preserves ć

3. **Page Number Confusion**: Dictionary pages start at 89, not 1
   - Pages 1-88: Grammar (different format)
   - Pages 89-440: Dictionary entries (primary target)

4. **Character Substitution**: VLMs may substitute similar-looking characters
   - Always verify: ć (c-acute) vs c, š (s-caron) vs s, ŋ (eng) vs n
   - Include `special_characters_found` field in extraction schema

5. **API Costs**: Claude Sonnet 4.5 costs ~$0.25/page for dictionary extraction
   - Full 352-page dictionary: ~$88
   - Test on page 89 first before batch processing

## Testing and Validation

**Quick validation workflow**:

```bash
# 1. Test single page extraction
python test_dakota_claude.py

# 2. Check output for special characters
cat data/dakota_test/dakota_extraction_test.json | grep -E "[ćšŋḣṡáéíóú]"

# 3. Verify confidence scores
python -c "import json; data = json.load(open('data/dakota_test/dakota_extraction_test.json')); print(f'Avg confidence: {sum(e[\"confidence\"] for e in data[\"interlinear_entries\"])/len(data[\"interlinear_entries\"])}')"
```

**Dataset validation**:
- Use `validate_entry()` from `dictionary_schema.py`
- Check for missing required fields
- Verify special character preservation
- Filter by confidence thresholds

## Complete Grammar → RL Pipeline

**End-to-end workflow for grammar-based RL training**:

```bash
# 1. Convert images (one-time setup)
python convert_all_images.py

# 2. Extract grammar rules from pages 1-88
python extract_grammar_pages.py --pages 1-88 --yes
# Output: 1,036 grammar rules in data/grammar_extracted/

# 3. Organize rules for RL training
python organize_grammar_for_rl.py --input data/grammar_extracted/
# Output: Categorized rules in data/rl_training_rules/

# 4. Generate RL training tasks
python convert_rules_to_primeintellect.py
# Output: 5,657 tasks in dakota_rl_training/datasets/

# 5. Launch RL training
cd dakota_rl_training
python train.py --config configs/training_config.yaml

# OR for distributed training via PrimeIntellect:
prime-rl train --config configs/training_config.yaml --num-workers 4 --use-toploc
```

**Complete pipeline in one command**:
```bash
python run_complete_grammar_pipeline.py
```

## Quick Reference: Extraction Code Patterns

**Claude Sonnet 4.5 Extraction (Recommended)**:

```python
import anthropic
from blackfeet_extraction.core.dakota_extraction_prompt import build_dakota_extraction_prompt
import base64
import json

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Encode image
image_bytes = Path("page_089.jpg").read_bytes()
encoded = base64.b64encode(image_bytes).decode("utf-8")

# Build prompt
prompt = build_dakota_extraction_prompt("Dictionary page with two-column format")

# Extract
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}},
            {"type": "text", "text": prompt}
        ]
    }]
)

# Parse JSON response
extracted_data = json.loads(response.content[0].text)
```

**Testing Extraction Quality**:

```python
# Run evaluation on extracted data
from eval.score_extraction import compute_metrics

# Load ground truth and predictions
ground_truth = load_jsonl("eval/fixtures/sample_ground_truth.jsonl")
predictions = load_jsonl("data/extracted/page_089_predictions.jsonl")

# Compute metrics
metrics = compute_metrics(predictions, ground_truth)
print(f"Token accuracy: {metrics['token_accuracy']:.2%}")
print(f"Character distance: {metrics['char_distance']:.3f}")
print(f"Diacritic preservation: {metrics['diacritic_preservation']:.2%}")
```

**Creating Fine-Tuning Dataset**:

```python
from blackfeet_extraction.schemas.dictionary_schema import DictionaryEntry, validate_entry

# Create entry
entry = DictionaryEntry(
    entry_id="page_089_entry_001",
    headword="ki'-ći-ća-šta-ka",
    definition_primary="to smile for one",
    part_of_speech="v.",
    column=1,
    page_number=89,
    source_image="grammardictionar00riggrich_0089.jpg",
    confidence=0.95
)

# Validate
is_valid, issues = validate_entry(entry)

# Convert to training format
translation_pair = entry.to_translation_pair()
instruction_format = entry.to_instruction_format()
```

## RL Training: Testing Verifiers Locally

**Test environment before full training**:

```python
# Test Dakota grammar environment
cd dakota_rl_training
python -c "
from verifiers.grammar_env import DakotaGrammarEnv
from verifiers.rubrics import DakotaGrammarRubric
import asyncio

async def test():
    env = DakotaGrammarEnv(max_turns=3)
    rubric = DakotaGrammarRubric()

    task = {
        'prompt': 'Apply possessive suffix to suŋka (younger brother)',
        'answer': 'Dawid suŋkaku',
        'info': {'task_type': 'morphology', 'difficulty': 'intermediate',
                 'special_chars': ['ŋ'], 'required_affixes': ['-ku']}
    }

    messages = [
        {'role': 'user', 'content': task['prompt']},
        {'role': 'assistant', 'content': 'Dawid suŋkaku'}
    ]

    feedback, state = await env.env_response(messages, {}, **task)
    reward = rubric.composite_reward('Dawid suŋkaku', task['answer'], task['info'])

    print(f'Feedback: {feedback[0][\"content\"]}')
    print(f'Reward: {reward:.2f}')

asyncio.run(test())
"
```

## Testing and CI

```bash
# Run offline smoke tests (no API calls required)
OFFLINE=1 pytest -q

# Runs test_offline_eval.py only
# Skips all tests with API keywords: claude, anthropic, openrouter, etc.
```

**CI configuration**: `.github/workflows/ci.yml` runs offline tests on every push

## Additional Resources

- Quick start: `START_HERE.md` or `QUICK_START_DAKOTA.md`
- Grammar RL pipeline: `GRAMMAR_RL_PIPELINE.md`
- Dakota RL training: `dakota_rl_training/README.md`
- Extraction results: `DAKOTA_EXTRACTION_RESULTS.md`
- Evaluation guide: `EVALUATION.md`
- Progress tracking: `PROGRESS.md`
- Historical context: `README.md` (Novel methodology section)
- Style guidance: Do not use emoji or emoticons in responses or documentation edits
