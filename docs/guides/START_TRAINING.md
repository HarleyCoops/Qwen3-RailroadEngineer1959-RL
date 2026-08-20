# Starting Dakota RL Training

## Quick Start

### 1. Install PrimeIntellect RL Framework

```powershell
pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
```

### 2. Verify Installation

```powershell
cd dakota_rl_training
python train.py --config configs/training_config.yaml
```

You should see:
```
[OK] Dataset found: datasets\grammar_tasks_complete.jsonl
READY TO TRAIN
```

### 3. Launch Training

**Option A: Local Training (Single GPU)**

```powershell
cd dakota_rl_training
uv run rl `
  --trainer @ configs/train.toml `
  --orchestrator @ configs/orch.toml `
  --inference @ configs/infer.toml `
  --trainer-gpu-ids 0 `
  --inference-gpu-ids 0
```

**Option B: Using PrimeIntellect Cloud Platform**

1. Set your API key:
```powershell
$env:PI_API_KEY = "your_api_key_here"
```

2. Launch via Python script:
```powershell
cd dakota_rl_training
python launch_primeintellect.py
```

**Option C: Direct Command (if prime-rl CLI available)**

```powershell
cd dakota_rl_training
prime-rl train `
  --config configs/training_config.yaml `
  --num-workers 4 `
  --use-toploc `
  --wandb-project dakota-rl-grammar
```

## Training Pipeline Overview

### Curriculum Stages

1. **Easy Tasks** (1,998 tasks)
   - Target: 80% accuracy
   - Duration: ~2-4 hours
   - Focus: Basic phonology, simple translations

2. **Medium Tasks** (2,155 tasks)
   - Target: 75% accuracy
   - Duration: ~3-5 hours
   - Focus: Complex morphology, multi-word translations

3. **Hard Tasks** (398 tasks)
   - Target: 70% accuracy
   - Duration: ~1-2 hours
   - Focus: Reverse translation, complex syntax

**Total Expected Time**: 6-11 hours

## Monitoring

### Key Metrics to Track

- `reward/mean`: Average reward per task
- `char_accuracy`: Dakota special character preservation (ć, š, ŋ, etc.)
- `affix_accuracy`: Morphological correctness
- `semantic_accuracy`: Translation quality

### Checkpoints

Training checkpoints saved to: `dakota_rl_training/checkpoints/`

## Troubleshooting

### Issue: ModuleNotFoundError for prime_rl
**Solution**: Install with `pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git`

### Issue: GPU Out of Memory
**Solution**: Reduce batch size in `configs/train.toml`:
```toml
per_device_train_batch_size = 2  # Reduce from 4
gradient_accumulation_steps = 8   # Increase from 4
```

### Issue: API Limit Error
**Solution**: If you hit API limits during extraction, the training data is already complete. You can proceed directly to RL training.

## Success Criteria

- Character preservation: >85% for all special chars
- Affix accuracy: >70% for common morphology
- Translation accuracy: >75% (easy), >60% (hard)

