# Repository Organization

This document outlines the current repository structure and proposed improvements for better organization.

## Current Structure

```
Dakota1890/
├── README.md                    # Main project README
├── CLAUDE.md                    # Developer guidance
├── LICENSE                      # License file
├── CODE_OF_CONDUCT.md           # Code of conduct
├── CONTRIBUTING.md              # Contribution guidelines
├── ETHICS.md                    # Ethics statement
├── DATA_LICENSE.md              # Data licensing
├── CITATION.cff                 # Citation file
│
├── docs/                        # All documentation
│   ├── README.md                # Documentation index
│   ├── guides/                  # User guides (19 files)
│   └── status/                  # Status/completion reports (8 files)
│
├── dakota_extraction/           # Core extraction module
│   ├── core/                    # Core extraction logic
│   ├── schemas/                 # Data schemas
│   ├── tools/                   # Utility tools
│   ├── datasets/               # Dataset builders
│   └── README.md
│
├── dakota_rl_training/          # RL training module
│   ├── verifiers/               # RL verifiers
│   ├── configs/                 # Training configs
│   ├── datasets/                # Training datasets
│   └── README.md
│
├── eval/                        # Evaluation framework
├── tools/                       # Root-level tools
├── data/                        # Data directory (gitignored)
├── Dictionary/                  # Source dictionary images
│
├── scripts/                     # Extraction scripts (root level)
│   ├── extract_*.py             # Various extraction scripts
│   ├── convert_*.py             # Conversion scripts
│   ├── test_*.py                # Test scripts
│   └── run_*.py                 # Pipeline scripts
│
└── prime-rl-framework/          # External framework dependency
```

## Completed Improvements

 **Documentation Organization**
- Moved code review/status files to `docs/status/`
- Moved user guides to `docs/guides/`
- Created `docs/README.md` for navigation
- Kept essential files in root (README, LICENSE, etc.)

## Proposed Improvements

### 1. Script Organization

**Current**: Many scripts in root directory
- `extract_*.py` (6 files)
- `convert_*.py` (3 files)
- `test_*.py` (7 files)
- `run_*.py` (1 file)
- `generate_*.py` (1 file)
- `create_*.py` (1 file)
- `organize_*.py` (1 file)
- `publish_*.py` (1 file)
- `dakota_openai_finetune.py`

**Proposed**: Create `scripts/` directory structure
```
scripts/
├── extraction/
│   ├── extract_dakota_dictionary_v2.py
│   ├── extract_dakota_dictionary.py
│   ├── extract_grammar_pages.py
│   └── extract_20_pages.py
├── conversion/
│   ├── convert_all_images.py
│   ├── convert_extracted_to_chat.py
│   └── convert_rules_to_primeintellect.py
├── testing/
│   ├── test_dakota_claude.py
│   ├── test_dakota_extraction.py
│   ├── test_grammar_extraction.py
│   └── ... (other test files)
├── pipelines/
│   ├── run_complete_grammar_pipeline.py
│   └── generate_synthetic_dakota.py
└── training/
    ├── dakota_openai_finetune.py
    └── organize_grammar_for_rl.py
```

**Benefits**:
- Cleaner root directory
- Easier to find scripts by category
- Better for IDE navigation
- Follows Python project best practices

### 2. Config Files Organization

**Current**: Config files scattered
- `pytest.ini` in root
- `conftest.py` in root
- Configs in `dakota_rl_training/configs/`

**Proposed**: Centralize configs
```
configs/
├── pytest.ini
├── conftest.py
└── training/                    # Symlink or copy from dakota_rl_training/configs/
```

**Alternative**: Keep pytest configs in root (standard practice), but document structure

### 3. Temporary/Output Files

**Current**: Some temporary files in root
- `error.txt`
- `rl_env_output.txt`
- `instructions.txt`

**Proposed**: Move to `.temp/` or ensure in `.gitignore`
- These should be gitignored anyway
- Consider `.temp/` directory for development files

### 4. Module Organization

**Current**: Good module structure
- `dakota_extraction/` - well organized
- `dakota_rl_training/` - well organized
- `eval/` - well organized

**Status**:  Good - no changes needed

### 5. Data Directory

**Current**: `data/` contains multiple subdirectories
- Already gitignored 
- Good organization 

**Status**:  Good - no changes needed

## Implementation Priority

1. **High Priority** (Immediate Benefit):
   -  Documentation organization (COMPLETED)
   - Script organization into `scripts/` subdirectories

2. **Medium Priority** (Nice to Have):
   - Config file organization
   - Temporary file cleanup

3. **Low Priority** (Future):
   - Further module refactoring (if needed)
   - Additional documentation improvements

## Migration Notes

If scripts are moved:
- Update imports in other scripts
- Update documentation references
- Update CI/CD paths if applicable
- Consider creating `scripts/__init__.py` for package structure

## Standards Followed

- **Documentation**: Markdown files organized by purpose
- **Code**: Python modules follow standard structure
- **Tests**: Test files co-located or in `tests/` directory
- **Configs**: Configuration files in appropriate locations
- **Data**: Data files gitignored and organized

## Recommendations

1. **Scripts Directory**: Most impactful improvement - reduces root clutter significantly
2. **Keep Root Clean**: Only essential files (README, LICENSE, setup files)
3. **Document Structure**: This file serves as the structure documentation
4. **Gradual Migration**: Scripts can be moved incrementally without breaking functionality

