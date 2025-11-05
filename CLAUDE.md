# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dakota language preservation through VLM-based extraction and RL training. This project extracts grammar rules and dictionary entries from a historical 1890 Dakota textbook (Stephen Return Riggs' 665-page Dakota Grammar & Dictionary) using Vision-Language Models, then transforms them into RL training tasks for language model fine-tuning.

**Novel Methodology**: Single textbook → Complete training ecosystem (grammar rules become verifiable RL environments, dictionary provides vocabulary, synthetic data validated by grammar from same source).

**Primary VLM**: Claude Sonnet 4.5 (92-95% accuracy on Dakota orthography: ć, š, ŋ, ḣ)

## Common Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Add API keys to .env
ANTHROPIC_API_KEY=your_key_here
```

### Testing
```bash
# Run offline smoke tests (no API calls)
OFFLINE=1 pytest -q

# Run all tests (requires API keys)
pytest -q

# Test single page extraction
python test_dakota_claude.py
```

### Linting
```bash
# Check code style
ruff check .

# Format code (dakota-grammar-env package)
cd dakota-grammar-env && black src/
```

### Extraction Pipeline

**Image Conversion** (one-time setup):
```bash
python scripts/extraction/convert_all_images.py
# Output: 440 JPEG images in data/processed_images/
```

**Grammar Extraction** (pages 31-92):
```bash
python scripts/extraction/extract_grammar_pages.py --pages 31-92 --yes
# Output: 1,036 grammar rules in data/grammar_extracted/
```

**Dictionary Extraction** (pages 93-440):
```bash
# Test on first page
python test_dakota_claude.py

# Full extraction
python scripts/extraction/extract_dakota_dictionary_v2.py --pages 93-440
# Cost: ~$88, Time: ~12 hours
# Output: ~10,000 word pairs in data/dictionary_extracted/
```

### RL Training Pipeline

**Convert Grammar to RL Tasks**:
```bash
# Organize rules for RL
python scripts/rl/organize_grammar_for_rl.py --input data/grammar_extracted/

# Generate 5,657 RL training tasks
python scripts/conversion/convert_rules_to_primeintellect.py
# Output: dakota_rl_training/datasets/
```

**Run Complete Pipeline**:
```bash
# End-to-end: extraction → RL task generation → environment setup
python scripts/rl/run_complete_grammar_pipeline.py
```

**Launch RL Training**:
```bash
# Local training
cd dakota_rl_training
python train.py --config configs/training_config.yaml

# Distributed training via PrimeIntellect
prime-rl train --config configs/training_config.yaml --num-workers 4 --use-toploc
```

### Build Package
```bash
# Build dakota-grammar-env package
cd dakota-grammar-env
python -m build
# Output: dist/*.whl, dist/*.tar.gz
```

## Architecture

### High-Level Data Flow

```
1890 Textbook (JP2 images)
    ↓
scripts/extraction/convert_all_images.py
    ↓
JPEG images (data/processed_images/)
    ↓
VLM Extraction (Claude Sonnet 4.5)
    ├─→ Grammar (pages 31-92) → scripts/extraction/extract_grammar_pages.py
    │   └─→ 1,036 rules (data/grammar_extracted/)
    │       └─→ scripts/rl/organize_grammar_for_rl.py
    │           └─→ scripts/conversion/convert_rules_to_primeintellect.py
    │               └─→ 5,657 RL tasks (dakota_rl_training/datasets/)
    │
    └─→ Dictionary (pages 93-440) → scripts/extraction/extract_dakota_dictionary_v2.py
        └─→ ~10,000 word pairs (data/dictionary_extracted/)
            └─→ scripts/conversion/generate_synthetic_dakota.py
                └─→ Synthetic sentences validated by grammar
```

### Module Organization

**`dakota_extraction/`** - Core extraction infrastructure (legacy naming, processes Dakota)
- `core/dakota_extraction_prompt.py` - Specialized prompt for Dakota orthography preservation
- `core/claude_page_processor.py` - Claude API wrapper
- `schemas/dictionary_schema.py` - Data validation (DictionaryEntry dataclass)
- `datasets/training_dataset_builder.py` - Convert to fine-tuning formats
- `run_extraction.py` - Pipeline orchestration

**`dakota_rl_training/`** - RL training system (PrimeIntellect integration)
- `verifiers/grammar_env.py` - Multi-turn & single-turn RL environments
- `verifiers/rubrics.py` - Compositional reward functions (character preservation 40%, affix accuracy 40%, semantic accuracy 20%)
- `configs/training_config.yaml` - GRPO training config (Qwen2.5-7B-Instruct, LoRA rank 64)
- `datasets/` - Generated RL tasks (5,657 tasks from 1,036 grammar rules)
- `train.py` - Training launcher

**`dakota-grammar-env/`** - Standalone Python package for distribution
- `src/dakota_grammar_env/` - Package source
- `pyproject.toml` - Package metadata (requires Python 3.11+)
- Publishable to PyPI for use in other projects

**`scripts/`** - Utility scripts organized by function
- `extraction/` - Image conversion, grammar/dictionary extraction
- `conversion/` - Format converters (RL tasks, synthetic data, HuggingFace upload)
- `rl/` - RL environment setup, task organization, training utilities
- `utilities/` - Helper scripts

**`data/`** - All extraction outputs (created during pipeline runs)
- `processed_images/` - Converted JPEG files (440 pages)
- `grammar_extracted/` - Raw grammar rules (62 JSON files)
- `rl_training_rules/` - Organized rules for RL (1,036 rules)
- `dictionary_extracted/` - Dictionary entries (~10,000 word pairs)
- `synthetic_dataset/` - Grammar-validated synthetic sentences

**`tests/`** - Test suite
- Offline mode: `OFFLINE=1` skips API-dependent tests (see conftest.py)
- Key tests: `test_dakota_claude.py`, `test_grammar_extraction.py`, `test_offline_eval.py`

### Key Design Patterns

**1. Closed-Loop Grammar Gym**
- Grammar rules → RL reward functions
- Dictionary → Synthetic sentences
- Grammar validates synthetic outputs
- Single source ensures consistency

**2. Compositional Rewards** (`dakota_rl_training/verifiers/rubrics.py`)
```python
reward = (
    0.4 * character_preservation +  # Dakota special chars (ć, š, ŋ, ḣ)
    0.4 * affix_accuracy +          # Morphological correctness (-ku, ta-, etc.)
    0.2 * semantic_accuracy         # Translation quality
) * difficulty_multiplier           # 1.0x (easy) to 2.0x (hard)
```

**3. Curriculum Learning**
- Progressive difficulty: easy (1,998 tasks) → medium (2,155) → hard (398) → advanced (1,106)
- Automatic advancement based on accuracy thresholds (80% → 75% → 70%)

**4. TOPLOC Verification**
- Distributed RL training with verifiable Unicode preservation
- Critical for Dakota special characters (prevents corruption in untrusted workers)

**5. Offline Testing** (`conftest.py`)
- `OFFLINE=1` environment variable skips all API/network tests
- Enables CI/CD without API keys
- Keywords filtered: claude, anthropic, openrouter, qwen, inference, verifier, prime, primeintellect, rl

## Dakota Language Orthography (CRITICAL)

Dakota uses special characters that MUST be preserved exactly during all operations:

**Special characters**: ć (c-acute), š (s-caron), ŋ (eng), ḣ (h-dot), ṡ (s-dot), á é í ó ú (pitch accents)

**Character accuracy is critical**: ć ≠ c, š ≠ s, ŋ ≠ n

**Always use UTF-8 encoding**: `json.dump(data, f, ensure_ascii=False)` to preserve characters

See [docs/root/CLAUDE.md](docs/root/CLAUDE.md) for comprehensive orthography guide and common Dakota words.

## Important File References

**Core prompts**:
- [dakota_extraction/core/dakota_extraction_prompt.py](dakota_extraction/core/dakota_extraction_prompt.py) - Dakota-specific VLM prompt
- [dakota_extraction/core/grammar_extraction_prompt.py](dakota_extraction/core/grammar_extraction_prompt.py) - Grammar rule extraction prompt

**Schemas**:
- [dakota_extraction/schemas/dictionary_schema.py](dakota_extraction/schemas/dictionary_schema.py) - DictionaryEntry dataclass, validation

**RL components**:
- [dakota_rl_training/verifiers/grammar_env.py](dakota_rl_training/verifiers/grammar_env.py:20-50) - DakotaGrammarEnv class
- [dakota_rl_training/verifiers/rubrics.py](dakota_rl_training/verifiers/rubrics.py:10-80) - Reward function implementation

**Training config**:
- [dakota_rl_training/configs/training_config.yaml](dakota_rl_training/configs/training_config.yaml) - GRPO hyperparameters

**CI/CD**:
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - Ruff linting, offline pytest

## Common Workflows

**Add new grammar extraction page**:
1. Ensure images converted: `python scripts/extraction/convert_all_images.py`
2. Extract page: `python scripts/extraction/extract_grammar_pages.py --pages X-Y --yes`
3. Organize for RL: `python scripts/rl/organize_grammar_for_rl.py --input data/grammar_extracted/`
4. Generate tasks: `python scripts/conversion/convert_rules_to_primeintellect.py`

**Test RL environment locally**:
```python
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

**Validate extraction quality**:
```bash
# Check special character preservation
cat data/dakota_test/dakota_extraction_test.json | grep -E "[ćšŋḣṡáéíóú]"

# Compute accuracy metrics
python eval/run_eval.py --pred eval/fixtures/sample_predictions.jsonl \
    --truth eval/fixtures/sample_ground_truth.jsonl --out eval/report.md
```

## Additional Documentation

For detailed information, see:
- [docs/root/CLAUDE.md](docs/root/CLAUDE.md) - Comprehensive guide (orthography, extraction patterns, RL details)
- [README.md](README.md) - Research methodology, results, citation
- [dakota_rl_training/README.md](dakota_rl_training/README.md) - RL training architecture and metrics
- [docs/guides/GRAMMAR_RL_PIPELINE.md](docs/guides/GRAMMAR_RL_PIPELINE.md) - End-to-end pipeline guide
- [docs/status/](docs/status/) - Project status updates
