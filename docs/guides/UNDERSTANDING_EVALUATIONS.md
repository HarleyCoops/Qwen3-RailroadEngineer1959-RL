# What to Expect from Dakota1890 Evaluation

## What is an "Evaluation"?

An **evaluation** (or "eval") is a **benchmark test** that measures how well a model performs on specific tasks without any training. Think of it like a test or exam.

### Key Differences:

| **Evaluation** | **Training** |
|----------------|--------------|
| **Tests** model performance | **Improves** model performance |
| **Measures** how well model does | **Trains** model to do better |
| **No learning** happens | **Learning** happens |
| **Snapshot** of current ability | **Improvement** over time |

## What You're Testing

With **Qwen 235B** on **5 examples** with **3 rollouts** each, you're measuring:

1. **Baseline Performance**: How well this untrained model handles Dakota grammar tasks
2. **Task Difficulty**: Which tasks are hardest for the model
3. **Model Capabilities**: What the model can/can't do out-of-the-box

## What Results You'll See

### 1. **Reward Metrics** (per example)

Each example gets scored on:

- **`reward`**: Overall composite score (0.0 to 1.0)
  - Higher = better performance
  
- **`character_preservation_reward`**: How well Dakota special characters preserved
  - ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú
  - Critical for Dakota accuracy!
  
- **`affix_accuracy_reward`**: Correct affix application (prefixes/suffixes)
  - Dakota uses extensive affixation
  
- **`semantic_accuracy_reward`**: Translation/semantic correctness
  - Does the answer make sense?
  
- **`composite_reward`**: Weighted combination of all metrics
  - Overall quality score

### 2. **Aggregate Statistics**

- **Average rewards** across all examples
- **Standard deviation** (consistency)
- **Per-example breakdown** (what worked best/worst)

### 3. **Example Outputs**

You'll see:
- **Input prompts**: The Dakota grammar questions
- **Model responses**: What Qwen 235B generated
- **Correct answers**: What it should have said
- **Scores**: How close each response was

## Expected Timeline

With **Qwen 235B** (very large model):
- **5 examples × 3 rollouts = 15 total inferences**
- **Estimated time**: ~15-20 minutes
- **Cost**: Small amount (5 examples is cheap)

## What This Tells You

### If Scores are Low (< 0.3):
- Model struggles with Dakota grammar out-of-the-box
- Training would help significantly
- Tasks are challenging (good for RL training!)

### If Scores are Medium (0.3-0.6):
- Model has some understanding
- Room for improvement with training
- Baseline to compare against

### If Scores are High (> 0.6):
- Model already handles Dakota grammar well
- Less room for improvement
- May need harder tasks

## After Evaluation

### Compare Models:
- Run same eval on different models
- See which performs best baseline

### Compare Before/After Training:
- **Before**: This eval (baseline)
- **After Training**: Run same eval again
- **Improvement**: See how much better trained model performs

### Identify Weak Areas:
- Look at low-scoring examples
- Focus training on those problem areas
- Adjust curriculum/difficulty

## Example Evaluation Flow

```
1. Start Eval → Model generates responses
2. Compare → Score each response vs correct answer
3. Aggregate → Calculate average rewards
4. Results → See what worked, what didn't
5. Analyze → Identify patterns, weaknesses
6. Next Steps → Train model, or adjust eval
```

## What Makes This Useful

This evaluation is a **baseline** - it shows:
-  How hard the tasks are
-  What the model struggles with
-  What training should focus on
-  How much improvement is possible

After you train your model with **GRPO** (Group Relative Policy Optimization), you'll run the same evaluation again to see:
- How much better your trained model performs
- Which tasks improved most
- Overall training effectiveness

## Key Takeaway

**This eval = "How good is this model at Dakota grammar right now?"**

**After training = "How much better did we make it?"**

The difference = your training success! 

