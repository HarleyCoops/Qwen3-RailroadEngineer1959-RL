# Scaling Up Evaluation Before Training

## Goal

Get a comprehensive baseline before training to:
1. **Establish baseline** performance metrics
2. **Identify hardest tasks** (focus training on these)
3. **Measure improvement** after training
4. **Validate** the evaluation setup works at scale

## Recommended Evaluation Sizes

### Option 1: Moderate Sample (Recommended)
**100 examples × 3 rollouts = 300 inferences**

- **Time**: ~1-2 hours
- **Cost**: Reasonable
- **Purpose**: Good baseline, identify problem areas
- **Good for**: Initial validation before training

### Option 2: Comprehensive Sample
**500 examples × 3 rollouts = 1,500 inferences**

- **Time**: ~5-8 hours  
- **Cost**: Moderate
- **Purpose**: Strong statistical confidence
- **Good for**: Thorough baseline before major training run

### Option 3: Large Sample
**1,000 examples × 3 rollouts = 3,000 inferences**

- **Time**: ~10-15 hours
- **Cost**: Higher but manageable
- **Purpose**: Very comprehensive baseline
- **Good for**: Research/publication, before major training

### Option 4: Full Dataset (Not Recommended)
**10,576 examples × 3 rollouts = 31,728 inferences**

- **Time**: ~100+ hours (4+ days)
- **Cost**: Very expensive
- **Purpose**: Complete coverage
- **Note**: Usually unnecessary - random sample is sufficient

## Recommended: Start with 500 Examples

**Why 500?**
-  Good statistical sample (5% of dataset)
-  Catches most difficulty patterns
-  Reasonable time/cost
-  Strong enough baseline for comparison

## Configuration

### In Prime Intellect UI:

**Environment Arguments:**
- Remove `dataset_path` (now auto-detected in v0.1.1!)
- Keep default (will use packaged dataset)

**Evaluation Settings:**
- **Number of Examples**: `500`
- **Rollouts per example**: `3` (for statistical reliability)
- **Max tokens**: `512` (slightly higher for complex tasks)
- **Temperature**: `0.7`

### Or via CLI:

```powershell
# Get dataset path (if needed for override)
$datasetPath = (Resolve-Path dakota_rl_training/datasets/grammar_tasks_complete.jsonl).Path -replace '\\', '/'
$envArgs = '{\"dataset_path\": \"' + $datasetPath + '\", \"max_examples\": 500}'

# Run comprehensive eval
prime env eval dakota1890 `
  -m qwen/qwen3-235b-a22b-2507 `
  -n 500 `
  -r 3 `
  -t 512 `
  -T 0.7 `
  --env-args $envArgs
```

## After Evaluation

### 1. Analyze Results
- Check average rewards
- Identify lowest-scoring examples
- Note which task types are hardest

### 2. Plan Training
- Focus curriculum on hardest tasks
- Adjust difficulty progression
- Set target metrics

### 3. Train on Prime Intellect GPUs
- Use Prime Intellect RL framework
- Train with GRPO (Group Relative Policy Optimization)
- Monitor progress vs baseline

### 4. Re-evaluate After Training
- Run same eval with trained model
- Compare scores: before vs after
- Measure improvement

## Expected Baseline Performance

Based on your 5-example test:
- **Average reward**: Likely 0.1-0.3 (untrained model)
- **Character preservation**: May struggle (Dakota special chars)
- **Affix accuracy**: Mixed (depends on task)
- **Semantic accuracy**: Low (needs training)

**After training**, expect:
- **Average reward**: 0.5-0.8 (significant improvement)
- **Character preservation**: Much better (critical for Dakota)
- **Affix accuracy**: Improved (model learns patterns)
- **Semantic accuracy**: Higher (better translations)

## Time & Cost Estimates

### For 500 Examples × 3 Rollouts:

**Qwen 235B:**
- **Time**: ~5-8 hours
- **Cost**: Moderate (235B is large but 500 examples is manageable)

**Smaller model (for comparison):**
- **Time**: ~2-3 hours
- **Cost**: Lower
- **Good for**: Quick baseline comparison

## Next Steps Checklist

1.  **Run 500-example eval** (baseline)
2.  **Analyze results** (identify problem areas)
3.  **Set up Prime Intellect RL training** (GRPO)
4.  **Train model** (on full dataset with RL)
5.  **Re-run eval** (measure improvement)
6.  **Compare before/after** (prove training worked!)

## Key Insight

**You don't need to eval ALL examples** - a well-chosen sample (500-1000 examples) gives you:
-  Statistical confidence
-  Representative difficulty distribution  
-  Reasonable time/cost
-  Strong baseline for comparison

**Focus training on the hardest examples** identified in the eval!

