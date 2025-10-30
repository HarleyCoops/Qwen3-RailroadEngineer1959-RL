# Dakota Dictionary Extraction Status

## Current Status

**Full Dictionary Extraction**: Running in background

**Command**: `python extract_dakota_dictionary_v2.py --all-dictionary`

**Pages to Extract**: 95-440 (346 pages total)

**Estimated Time**: ~11.5 hours (2 minutes per page)

**Estimated Cost**: ~$86.50 (~$0.25 per page)

## Extraction Progress

Check progress with:
```powershell
Get-ChildItem data\extracted\page_*.json | Measure-Object | Select-Object Count
```

## Output Location

- **Extracted Entries**: `data/extracted/page_095.json`, `page_096.json`, etc.
- **Claude Responses**: `data/reasoning_traces/page_095_claude_response.txt`, etc.

## What's Being Extracted

Each page produces structured dictionary entries with:
- `headword`: Dakota word (with special characters preserved)
- `definition_primary`: Main English meaning
- `definition_secondary`: Additional meanings
- `part_of_speech`: v., n., adj., etc.
- `inflected_forms`: Conjugations/variants
- `derived_from`: Etymology/root words
- `grammatical_notes`: Detailed grammar explanations
- `see_also`: Cross-references
- `compare_with`: Related words

## Next Steps (After Extraction Completes)

1. **Analyze extracted data**:
   - Count total entries
   - Identify polysemous words (same headword, different meanings)
   - Map cross-references
   - Extract morphological patterns

2. **Enhance synthetic Q&A generation**:
   - Leverage rich metadata (part of speech, inflections, etymology)
   - Create sophisticated question types
   - Generate grammar-aware answers
   - Bridge dictionary ↔ grammar knowledge

3. **Generate bidirectional Q&A pairs**:
   - English → Dakota (translation + grammar)
   - Dakota → English (meanings + usage)
   - Part of speech classification
   - Inflected forms
   - Etymology & word formation
   - Comparative questions

## Files Created

- `ENHANCED_SYNTHETIC_QA_PLAN.md`: Detailed plan for enhanced Q&A generation
- `DICTIONARY_EXTRACTION_PIPELINE.md`: Complete pipeline documentation

## Notes

- Dictionary extraction uses **Claude Sonnet 4.5** (not Gemini)
- Each entry preserves Dakota special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)
- The Dakota dictionary structure is more complex than Stoney Nakoda, with richer linguistic metadata
- This richness enables sophisticated Q&A generation beyond simple translation pairs

