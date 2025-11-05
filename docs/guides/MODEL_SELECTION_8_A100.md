# Model Selection for 8 A100 Training

## Problem

**Qwen 235B is too large for 8 A100 training:**
- 235B parameters (MoE with 22B active)
- Training overhead requires ~470B params in memory
- Needs 16+ A100s or H100s
- Only suitable for inference/evaluation

## Recommended: Qwen3-30B-A3B

**Best balance for 8 A100 training:**

### Why Qwen3-30B-A3B?
- ✅ **Proven on 8 GPUs**: Used in Prime RL framework configs
- ✅ **Fits comfortably**: 30B params manageable on 8 A100s
- ✅ **Good performance**: Strong capabilities for Dakota grammar
- ✅ **MoE architecture**: 3B active params per token (efficient)
- ✅ **Fast training**: Reasonable iteration speed

### Configuration (from Prime RL examples):
```toml
[model]
name = "Qwen/Qwen3-30B-A3B"

[orchestrator]
batch_size = 512
micro_batch_size = 2
rollouts_per_example = 16
seq_len = 2048
```

## Alternative Models

### Option 2: Qwen3-72B
- **Larger**: Better performance potential
- **Tighter fit**: Might work on 8 A100s depending on config
- **Slower**: More memory pressure, slower training

### Option 3: Qwen2.5-7B or Qwen3-7B
- **Smaller**: Definitely fits, very fast
- **Good for iteration**: Rapid experimentation
- **Lower ceiling**: May limit final performance

## Recommended Workflow

### 1. Baseline Evaluation
**Run eval on Qwen3-30B-A3B** (the model you'll train):
```powershell
prime env eval dakota1890 `
  -m qwen/qwen3-30b-a3b `
  -n 500 `
  -r 3 `
  -t 512 `
  -T 0.7
```

**Why evaluate the model you'll train?**
- Establishes baseline for that specific model
- Fair comparison (same model, before/after training)
- Identifies model-specific weaknesses

### 2. Training Setup
**Configure training for Qwen3-30B-A3B:**
```toml
[model]
name = "Qwen/Qwen3-30B-A3B"

[orchestrator]
batch_size = 512
micro_batch_size = 2
rollouts_per_example = 16

[trainer]
# GRPO configuration
lora_rank = 64
learning_rate = 5e-6
```

### 3. After Training
**Re-run same eval** to measure improvement:
```powershell
# Same command as baseline
prime env eval dakota1890 `
  -m qwen/qwen3-30b-a3b `
  -n 500 `
  -r 3 `
  -t 512 `
  -T 0.7
```

**Compare results:**
- Before training: Baseline scores
- After training: Improved scores
- Difference = training effectiveness

## Model Comparison

| Model | Params | A100 Fit | Training Speed | Performance |
|-------|--------|----------|----------------|--------------|
| **Qwen3-30B-A3B** | 30B (3B active) | ✅ 8 A100s | Good | Strong |
| Qwen3-72B | 72B | ⚠️ Tight | Slower | Better |
| Qwen2.5-7B | 7B | ✅✅ Easy | Very Fast | Lower |
| Qwen 235B | 235B (22B active) | ❌ No | N/A | Best (eval only) |

## Key Insight

**Use the same model for eval AND training** - this gives you:
- Fair comparison (before/after)
- Model-specific baseline
- Clear improvement measurement

**Don't eval on 235B then train 30B** - that's comparing apples to oranges!

## Next Steps

1. ✅ **Switch eval to Qwen3-30B-A3B** (baseline)
2. ✅ **Configure training for Qwen3-30B-A3B** (8 A100s)
3. ✅ **Train with GRPO** (full dataset)
4. ✅ **Re-eval Qwen3-30B-A3B** (measure improvement)

This gives you a clean before/after comparison!

