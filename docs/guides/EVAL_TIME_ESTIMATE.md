# Eval Time Estimate for Dakota1890

## Configuration
- **Model**: Qwen3 235B-a22 (235 billion parameters - very large!)
- **Examples**: 5
- **Rollouts per example**: 1
- **Max tokens**: 256
- **Dataset**: 10,576 total tasks (but only evaluating 5)

## Time Breakdown

### Per Example Processing Time
For a 235B model on Prime Inference:
- **Prompt processing**: ~5-10 seconds (large model initialization overhead)
- **Token generation** (256 max tokens): ~10-20 seconds
- **Scoring/evaluation**: ~2-5 seconds (Dakota grammar rubric computation)
- **Total per example**: ~17-35 seconds

### Total Runtime Estimate

**Optimistic (fast cluster, low load)**:
- 5 examples × 20 seconds = 100 seconds
- Setup overhead: 30 seconds
- **Total: ~2-3 minutes**

**Realistic (typical conditions)**:
- 5 examples × 30 seconds = 150 seconds
- Setup overhead: 60 seconds
- **Total: ~3-4 minutes**

**Conservative (slow conditions, queue wait)**:
- 5 examples × 40 seconds = 200 seconds
- Setup overhead: 90 seconds
- Queue wait: 60 seconds
- **Total: ~5-7 minutes**

## Factors Affecting Time

1. **Model Size**: 235B is massive - much slower than smaller models
   - Smaller models (7B-72B): ~5-10 seconds per example
   - 235B: ~20-40 seconds per example

2. **Prime Inference Load**: 
   - Busy times = queue wait
   - Off-peak = faster

3. **Token Generation**:
   - 256 tokens is relatively short
   - Longer responses = more time

4. **Environment Overhead**:
   - Dataset loading: ~5-10 seconds
   - Environment initialization: ~5-10 seconds

## Comparison to Full Eval

If you were to run a full evaluation:
- **100 examples, 3 rollouts**: ~30-60 minutes
- **Full dataset (10,576 examples)**: ~60-120 hours (not recommended!)

## Recommendation

**Expected time for your test run: 3-5 minutes**

This is reasonable for:
- Testing pipeline connectivity
- Verifying W&B/HF integration
- Confirming dataset loading works
- Checking model responses

After this test succeeds, you can scale up:
- 10 examples: ~5-8 minutes
- 20 examples: ~8-12 minutes
- 100 examples: ~30-45 minutes

## Cost Estimate

Based on Prime Inference pricing for `qwen/qwen3-235b-a22`:
- Input: $0.22 per 1M tokens
- Output: $0.88 per 1M tokens
- 5 examples × 256 tokens ≈ 1,280 tokens
- **Estimated cost: < $0.01** (essentially free for test run)

