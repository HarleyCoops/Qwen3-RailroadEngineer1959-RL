# Stoney Nakoda → Dakota Adaptation Guide

## Executive Summary

After thoroughly examining the **HarleyCoops/stoneynakoda** repository, I've identified the complete pipeline architecture and can now map exactly what needs to be adapted for Dakota. The Stoney Nakoda codebase has **two fully functional pipelines** that Dakota is missing:

1. **Dictionary → SFT Pipeline** (Supervised Fine-Tuning)
2. **Grammar → RL Pipeline** (Reinforcement Learning)

## Complete Pipeline Architecture from Stoney Nakoda

### Pipeline 1: Dictionary → SFT (Supervised Fine-Tuning)

**Flow:**
```
Dictionary JSONL → Gemini Q&A Generation → OpenAI Chat Format → OpenAI Fine-tuning
```

**Key Files:**
1. `bilingual_qa_generator.py` - Generates Q&A pairs from dictionaries using Google Gemini
2. `finetunesetup.py` - Converts Q&A to OpenAI chat format (80/20 train/val split)
3. `openai_finetune.py` - Runs OpenAI fine-tuning with W&B tracking and HF publishing

**What Dakota Needs:**
-  Dictionary extraction exists (`data/extracted/*.json`)
-  **MISSING:** Script to convert extracted dictionary JSON → chat JSONL format
-  **MISSING:** Actual OpenAI fine-tuning script (Dakota only has placeholder)
-  **MISSING:** Q&A generation from dictionary (referenced but not implemented)

### Pipeline 2: Grammar → RL (Reinforcement Learning)

**Flow:**
```
Grammar PDF → PDF Ingest (PNG images) → Vision Extraction → Rule Organization → Task Generation → RL Environment
```

**Key Files:**
1. `stoney_rl_grammar/pdf_ingest.py` - Renders PDF pages to PNG + base64
2. `stoney_rl_grammar/rule_extractor.py` - Uses OpenAI vision models to extract structured rules
3. `stoney_rl_grammar/rule_organizer.py` - Filters, deduplicates, curates rules
4. `stoney_rl_grammar/task_generator.py` - Generates RL tasks from rules
5. `stoney_rl_grammar/pipeline.py` - Orchestrates all stages
6. `environments/stoney_nakoda_translation/` - Verifiers-compatible package

**What Dakota Has:**
-  Grammar extraction exists (`create_grammar_rl_environment.py`)
-  Rule organization exists (`organize_grammar_for_rl.py`)
-  Task generation exists (`convert_rules_to_primeintellect.py`)
-  **PARTIAL:** Environment exists but not properly packaged/integrated

**What Dakota Needs:**
-  **MISSING:** Proper verifiers-compatible environment package (like `stoney_nakoda_translation/`)
-  **MISSING:** Actual RL trainer implementation (train.py is just a stub)
-  **NEEDS IMPROVEMENT:** Task generation needs multi-step morphology tasks

## Detailed File-by-File Comparison

### 1. Dictionary → Chat Format Conversion

**Stoney Nakoda:** `finetunesetup.py`
- Reads `Dictionaries/bilingual_training_set.jsonl`
- Converts each Q&A pair to OpenAI chat format:
  ```json
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"}
  ]}
  ```
- 80/20 train/val split with shuffling
- Writes `OpenAIFineTune/stoney_train.jsonl` and `stoney_valid.jsonl`

**Dakota Status:**
- Has `data/extracted/*.json` files with dictionary entries
- **MISSING:** Equivalent script to convert to chat format

**Action Required:** Create `convert_extracted_to_chat.py` adapted from `finetunesetup.py`

### 2. OpenAI Fine-Tuning

**Stoney Nakoda:** `openai_finetune.py`
- Full implementation with:
  - File upload to OpenAI
  - Job creation with hyperparameters
  - Progress monitoring (status, tokens, accuracy, loss)
  - Optional HuggingFace dataset publishing
  - Optional Weights & Biases tracking
  - Error handling and retries

**Dakota Status:**
- README mentions LoRA fine-tuning expectations
- **MISSING:** Actual fine-tuning script

**Action Required:** Adapt `openai_finetune.py` for Dakota (or create PEFT/TRL trainer if doing local LoRA)

### 3. Q&A Generation from Dictionary

**Stoney Nakoda:** `bilingual_qa_generator.py`
- Uses Google Gemini to generate Q&A pairs
- Processes both English→Stoney and Stoney→English dictionaries
- Creates 75K pairs per language direction (150K total)
- Checkpoint system for recovery
- Context windowing (5 entries at a time)

**Dakota Status:**
- README references synthetic generator
- **MISSING:** Actual implementation

**Action Required:** Create `generate_synthetic_dakota.py` adapted from `bilingual_qa_generator.py`

### 4. RL Environment Package

**Stoney Nakoda:** `environments/stoney_nakoda_translation/`
- Proper Python package with `pyproject.toml`
- Verifiers-compatible structure:
  - `environment.py` - `StoneyTranslationEnv` inheriting from `SingleTurnEnv`
  - `StoneyTranslationRubric` with multiple reward functions
  - `load_environment()` function for easy import
  - Proper dataset loading from JSONL

**Dakota Status:**
- Has `dakota_rl_training/verifiers/grammar_env.py` and `rubrics.py`
- **MISSING:** Proper package structure and integration
- **MISSING:** `load_environment()` function pattern

**Action Required:** 
1. Create `environments/dakota_grammar_translation/` package
2. Adapt environment code to match Stoney pattern
3. Ensure verifiers compatibility

### 5. RL Task Generation

**Stoney Nakoda:** `stoney_rl_grammar/task_generator.py`
- Uses LLM to generate 3-6 tasks per rule
- Task types: morphology, forward_translation, reverse_translation, pattern_identification, syntax_analysis
- Includes hints, verification patterns, difficulty levels
- Writes JSONL format ready for verifiers

**Dakota Status:**
- Has `convert_rules_to_primeintellect.py`
- User noted tasks are "still lexical gloss drills"
- **NEEDS IMPROVEMENT:** Add multi-step morphology tasks (positive/negative evidence, affix insertion, exception triggers)

**Action Required:** Enhance `convert_rules_to_primeintellect.py` with multi-step task generation

### 6. RL Trainer Implementation

**Stoney Nakoda:** 
- Uses `verifiers` framework with `GRPOEnvTrainer`
- Environment loads tasks from JSONL
- Direct integration with PrimeIntellect/verifiers

**Dakota Status:**
- `dakota_rl_training/train.py` is just a stub that prints instructions
- **MISSING:** Actual training code

**Action Required:** 
1. Implement actual `prime_rl.Trainer` wiring
2. Load environment from verifiers package
3. Pipe curriculum splits
4. Add logging/checkpoint hooks

### 7. Grammar Pipeline Orchestration

**Stoney Nakoda:** `run_stoney_grammar_pipeline.py` → `stoney_rl_grammar/pipeline.py`
- Clean orchestration class `StoneyGrammarPipeline`
- Runs all stages sequentially
- Proper logging and error handling

**Dakota Status:**
- Has `run_complete_grammar_pipeline.py`
- **NEEDS IMPROVEMENT:** Ensure it properly connects all stages

## Implementation Priority

Based on user's feedback, here's the priority order:

### Priority 1: Core Missing Pieces
1. **SFT Data Packager** - `convert_extracted_to_chat.py` (adapt from `finetunesetup.py`)
2. **SFT Trainer** - `openai_finetune.py` (adapt from Stoney or create PEFT trainer)
3. **RL Trainer** - Replace stub in `train.py` with actual `prime_rl.Trainer` wiring
4. **Environment Package** - Create proper `environments/dakota_grammar_translation/` package

### Priority 2: Enhancements
5. **Synthetic Generator** - `generate_synthetic_dakota.py` (adapt from `bilingual_qa_generator.py`)
6. **Enhanced Task Generation** - Improve `convert_rules_to_primeintellect.py` with multi-step morphology
7. **Environment Connection** - Fix `create_grammar_rl_environment.py` to properly integrate with verifiers

## Key Adaptation Patterns

### Pattern 1: Dictionary → Chat Format
```python
# Input: data/extracted/*.json (Dakota dictionary entries)
# Output: OpenAIFineTune/dakota_train.jsonl, dakota_valid.jsonl
# Process: Convert each dictionary entry to Q&A pairs → chat format
```

### Pattern 2: OpenAI Fine-Tuning
```python
# Input: OpenAIFineTune/dakota_train.jsonl, dakota_valid.jsonl
# Process: Upload → Create job → Monitor → Return model ID
# Output: Fine-tuned model ID
```

### Pattern 3: RL Environment Package
```python
# Structure:
environments/dakota_grammar_translation/
├── pyproject.toml
├── dakota_grammar_translation/
│   ├── __init__.py          # exports load_environment
│   └── environment.py      # DakotaGrammarEnv + Rubric
```

### Pattern 4: RL Trainer
```python
# Replace stub with:
from prime_rl.trainer.rl.train import train as rl_train
from prime_rl.trainer.rl.config import RLTrainerConfig
# Load environment from verifiers package
# Configure curriculum splits
# Run actual training
```

## Next Steps

1. **Stop and Review** - User requested full understanding before proceeding
2. **Map Exact Adaptations** - This document provides the mapping
3. **Implement Systematically** - Follow priority order above
4. **Test Each Component** - Ensure each piece works before moving to next

## References

- Stoney Nakoda Repo: `../stoneynakoda/`
- Dakota Current State: `./`
- PrimeIntellect Framework: `./prime-rl-framework/`

