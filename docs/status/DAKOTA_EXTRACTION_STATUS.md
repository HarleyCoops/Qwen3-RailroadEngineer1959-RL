# Dakota Extraction Pipeline - Status

##  Completed

1. **Directory Renamed**: `blackfeet_extraction` → `dakota_extraction`
2. **Image Converter Updated**: Points to `Dictionary/grammardictionar00riggrich_jp2`
3. **Core Imports Fixed**: All internal imports updated to `dakota_extraction`
4. **Main Scripts Updated**: `extract_grammar_pages.py`, `extract_dakota_dictionary_v2.py`, `convert_all_images.py`

##  In Progress

### Image Conversion Pipeline
-  `dakota_extraction/tools/image_converter.py` - Updated to point to Dictionary directory
-  Default input directory: `Dictionary/grammardictionar00riggrich_jp2`
-  Supports 440 JP2 files from 1890 Dakota dictionary

### Extraction Pipeline Flow

```
Dictionary/*.jp2 files (440 pages)
    ↓
[Image Converter] JP2 → JPEG conversion
    ↓
data/processed_images/*.jpg
    ↓
[Claude Sonnet 4.5 VLM] Page-by-page extraction
    ↓
data/extracted/*.json (structured dictionary/grammar data)
    ↓
[generate_synthetic_dakota.py] Gemini Q&A generation
    ↓
data/bilingual_training_set.jsonl
    ↓
[convert_extracted_to_chat.py] OpenAI chat format
    ↓
data/dakota_train.jsonl (SFT ready)
```

##  Pages Structure

- **Pages 1-88**: Grammar rules (for RL training)
  - Extracted by: `extract_grammar_pages.py`
  - Output: `data/grammar_extracted/*.json`
  
- **Pages 89-440**: Dictionary entries (for SFT training)
  - Extracted by: `extract_dakota_dictionary_v2.py`
  - Output: `data/extracted/*.json`

##  Quick Start

1. **Convert Images**:
   ```bash
   python convert_all_images.py
   # Or use ImageConverter directly
   python dakota_extraction/tools/image_converter.py
   ```

2. **Extract Grammar** (pages 1-88):
   ```bash
   python extract_grammar_pages.py --pages 1-88
   ```

3. **Extract Dictionary** (pages 89-440):
   ```bash
   python extract_dakota_dictionary_v2.py --test  # Test on page 89
   python extract_dakota_dictionary_v2.py --pages 89-108  # First 20 pages
   ```

4. **Generate Synthetic Q&A**:
   ```bash
   python generate_synthetic_dakota.py
   ```

5. **Package for SFT**:
   ```bash
   python convert_extracted_to_chat.py
   ```

## ️ Remaining Tasks

- Update remaining "blackfeet" references in:
  - `dakota_extraction/core/page_processor.py`
  - `dakota_extraction/README.md`
  - `dakota_extraction/COMPLETE_GUIDE.md`
  - `dakota_extraction/schemas/dictionary_schema.py`
  - Other test files

- Verify image conversion works with actual .jp2 files
- Test extraction pipeline end-to-end

