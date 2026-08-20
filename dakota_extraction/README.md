#  Legacy Directory Notice

This folder contains historical Blackfeet examples. Dakota-first workflows now live in the repository root and `dakota_rl_training/`. These files will transition into `examples/` in a future release; current behavior is unchanged.

# Blackfeet Language Dataset Extraction Pipeline

## Project Goal

Build a comprehensive dataset extraction tool to process the 1890 Blackfeet dictionary images and create training data for an open-source Blackfeet language model, following the approach used for the Stoney Nakoda language project.

## Pipeline Overview

```

Dictionary Images → OCR + Structure Analysis → Structured Data → Training Dataset
                    (Qwen3-VL Thinking)        (JSON/CSV)        (For Model Training)

```

## Key Features

### 1. **Intelligent OCR with Reasoning**

- Use Qwen3-VL-235B-A22B-Thinking to extract text with visible reasoning

- Handle historical typography (1890s printing)

- Preserve structure (headwords, definitions, example sentences)

- Track confidence scores

### 2. **Linguistic Structure Extraction**

- Identify Blackfeet headwords

- Extract English translations

- Capture grammatical information

- Preserve example sentences

- Track etymological notes

### 3. **Dataset Generation**

- Parallel corpus (Blackfeet ↔ English)

- Vocabulary lists with metadata

- Sentence pairs for translation

- Word embeddings preparation

- Fine-tuning ready format

### 4. **Quality Assurance**

- Reasoning traces for verification

- Confidence scoring

- Manual review interface

- Consistency checking

- Progress tracking

## Dataset Output Formats

### Format 1: Parallel Corpus (Translation Training)

```json
{
  "entries": [
    {
      "id": "page_001_entry_001",
      "blackfeet": "áíkšisskstaki",
      "english": "he is brave",
      "pos": "verb",
      "page": 1,
      "confidence": 0.95,
      "reasoning": "The model identified this as a verb form..."
    }
  ]
}

```

### Format 2: Vocabulary Cards (Language Learning)

```json
{
  "word": "áíkšisskstaki",
  "translation": "he is brave",
  "part_of_speech": "verb",
  "root": "aikš",
  "related_words": ["áíkš", "ikšiskstaki"],
  "example_sentences": [],
  "audio_pronunciation": null,
  "image_source": "page_001.jpg"
}

```

### Format 3: Fine-Tuning Dataset (Model Training)

```jsonl
{"prompt": "Translate to English: áíkšisskstaki", "completion": "he is brave"}
{"prompt": "Translate to Blackfeet: he is brave", "completion": "áíkšisskstaki"}

```

## Processing Workflow

### Phase 1: Page-Level Extraction

```python
for page in dictionary_pages:
    # 1. Upload page image to Qwen3-VL

    # 2. Request structured extraction with reasoning

    # 3. Parse response into structured format

    # 4. Store with reasoning traces

    # 5. Quality check

```

### Phase 2: Cross-Page Analysis

```python

# Find word relationships across pages

# Build vocabulary network

# Identify grammatical patterns

# Create semantic clusters

```

### Phase 3: Dataset Assembly

```python

# Generate parallel corpus

# Create training/validation/test splits

# Format for specific model architectures

# Generate metadata

```

### Phase 4: Validation & Export

```python

# Manual review interface

# Confidence filtering

# Consistency checks

# Export final datasets

```

## Technical Architecture

```

blackfeet_extraction/
├── core/
│   ├── page_processor.py          # Single page OCR + extraction

│   ├── structure_analyzer.py      # Parse dictionary structure

│   ├── language_extractor.py      # Extract linguistic features

│   └── reasoning_logger.py        # Track model reasoning

├── pipeline/
│   ├── batch_processor.py         # Process multiple pages

│   ├── cross_reference.py         # Link related entries

│   ├── pattern_detector.py        # Find linguistic patterns

│   └── quality_checker.py         # Validation & QA

├── datasets/
│   ├── parallel_corpus.py         # Translation pairs

│   ├── vocabulary_builder.py      # Word lists

│   ├── sentence_extractor.py      # Example sentences

│   └── model_formatter.py         # Training data formats

├── tools/
│   ├── review_interface.py        # Manual review UI

│   ├── confidence_filter.py       # Filter by quality

│   ├── export_manager.py          # Export utilities

│   └── progress_tracker.py        # Track extraction progress

└── examples/
    ├── single_page_extraction.py
    ├── batch_processing.py
    ├── dataset_generation.py
    └── model_training_prep.py

```

## Use Cases After Extraction

1. **Translation Model**: Train Blackfeet ↔ English translator

2. **Language Model**: Train Blackfeet text generation

3. **Spell Checker**: Build Blackfeet spell/grammar checker

4. **OCR Model**: Train specialized Blackfeet OCR

5. **Speech Model**: Pair with audio for TTS/ASR

6. **Educational Apps**: Power language learning tools

## Advantages of Qwen3-VL Thinking for This Task

1. **Handles Historical Typography**: Superior OCR for 1890s printing

2. **Reasoning Traces**: Verify extraction decisions

3. **Structure Understanding**: Parse dictionary layout intelligently

4. **Linguistic Awareness**: Understand grammatical patterns

5. **Error Detection**: Flag uncertain extractions

6. **Context Preservation**: Maintain relationships between entries

## Next Steps

1. Implement core page processor

2. Test on sample dictionary pages

3. Refine extraction prompts

4. Build quality assurance tools

5. Process full dictionary

6. Generate training datasets

7. Train baseline Blackfeet language model
