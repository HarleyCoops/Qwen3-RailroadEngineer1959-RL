# Dakota Dictionary Extraction - Quick Start

## 🎯 Important: Dictionary Starts at Page 89!

**Pages 1-88**: Grammar rules and linguistic notes
**Pages 89-440**: Dictionary entries (352 pages) ← **This is what we extract**

## 🚀 Run the Test Now

```bash
python extract_dakota_dictionary_v2.py --test
```

This will:
1. Convert page 89 (first dictionary page) from JP2 to JPEG
2. Use Qwen3-VL Thinking to extract all entries
3. Show you the structured output
4. Save reasoning traces so you can verify the logic

**Cost**: ~$0.25
**Time**: ~2 minutes

## 📊 What You'll See

The test will extract entries like this from page 89:

```json
{
  "headword": "a-ća'-ka",
  "part_of_speech": "v. a.",
  "definition_primary": "to freeze on anything",
  "inflected_forms": ["waćaka", "uŋkćakapį"],
  "column": 1,
  "confidence": 0.95
}
```

## ✅ Next Steps After Test

### If extraction looks good:

**Option 1: Small batch (recommended)**
```bash
# Process 12 dictionary pages
python extract_dakota_dictionary_v2.py --pages 89-100
```
- Cost: ~$3
- Time: ~20 minutes
- Get ~300-500 entries

**Option 2: Larger sample**
```bash
# Process 60 dictionary pages
python extract_dakota_dictionary_v2.py --pages 89-150
```
- Cost: ~$15
- Time: ~2 hours
- Get ~1,500-2,500 entries

**Option 3: Full dictionary**
```bash
# Process all 352 dictionary pages
python extract_dakota_dictionary_v2.py --all-dictionary
```
- Cost: ~$88
- Time: ~12 hours
- Get complete dataset (~10,000+ entries)

## 📁 Where to Find Output

After extraction:

```
data/
├── extracted/
│   ├── page_089.json          # Full linguistic data
│   ├── page_090.json
│   └── ...
│
├── reasoning_traces/
│   ├── page_089_reasoning.json  # WHY the model made each decision
│   └── ...
│
└── training_datasets/
    ├── translation_train.jsonl  # Dakota ↔ English pairs
    ├── instruction_dataset.jsonl
    ├── vocabulary.json
    └── blackfeet_corpus.txt
```

## 🔍 Review Checklist

After test extraction, check:

1. **Headwords**: Are Dakota words spelled correctly with diacritics?
2. **Inflected forms**: Are conjugations separated from main entry?
3. **Definitions**: Is English meaning complete?
4. **Confidence**: Are scores mostly > 0.7?

**Review files:**
- `data/extracted/page_089.json` - See what was extracted
- `data/reasoning_traces/page_089_reasoning.json` - See how it decided

## 💡 Pro Tips

1. **Start small**: Test → 12 pages → review → continue
2. **Check reasoning**: Use reasoning traces to verify extraction logic
3. **Filter by confidence**: Use only entries with confidence > 0.7 for training
4. **Save grammar pages**: Pages 1-88 useful later (different extraction method)

## 🐛 Common Issues

**"PIL cannot read JP2"**
```bash
# Install OpenJPEG
# Windows: https://www.openjpeg.org/
# Linux: sudo apt-get install libopenjp2-7
# Mac: brew install openjpeg
```

**"Page 89 not found"**
- Check: `ls dictionary/grammardictionar00riggrich_jp2/*.jp2 | wc -l`
- Should show 440 files

**"OPENROUTER_API_KEY not set"**
- Add to `.env` file: `OPENROUTER_API_KEY=your_key_here`

## 📚 What's Next After Extraction

1. **Review quality**: Spot-check random pages
2. **Filter dataset**: Keep high-confidence entries (>0.7)
3. **Train model**: Use translation pairs for fine-tuning
4. **Evaluate**: Test on held-out dictionary pages
5. **Deploy**: Build Dakota language tools

## 🎓 Training Recommendations

**Small model (good for testing):**
- Model: Qwen2.5-1.5B-Instruct
- Dataset: 12 pages (~500 entries)
- Time: 30 minutes on 1 GPU
- Use: Proof of concept

**Medium model (production):**
- Model: Qwen2.5-7B-Instruct or LLaMA-3-8B
- Dataset: 100 pages (~3,000 entries)
- Time: 3-4 hours on 1 GPU
- Use: Real applications

**Full model (best quality):**
- Model: Qwen2.5-14B-Instruct
- Dataset: All 352 pages (~10,000 entries)
- Time: 8-12 hours on 2-4 GPUs
- Use: Production deployment

---

## 🏃 Ready to Start?

```bash
# 1. Test extraction
python extract_dakota_dictionary_v2.py --test

# 2. Review output
cat data/extracted/page_089.json | head -50

# 3. If good, process more
python extract_dakota_dictionary_v2.py --pages 89-100
```

Good luck! 🚀
