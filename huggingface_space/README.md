---
title: Dakota Grammar RL Demo
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
pinned: false
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
tags:
  - reinforcement-learning
  - rl
  - dakota-language
  - grammar
  - composition-rewards
  - non-coding
  - prime-intellect
  - verifiers
language:
  - en
  - dak
pipeline_tag: text-generation
---

# Qwen3-0.6B-Dakota-Grammar-RL

## Model Description

This model is a reinforcement learning (RL) fine-tuned version of `Qwen/Qwen3-0.6B`, trained specifically for Dakota language grammar and translation tasks using **compositional reward functions** on **non-coding tasks**. This represents a test of the RL pipeline's effectiveness for complex, multi-component reward structures in linguistic domains.

### Key Features

- **Compositional Rewards**: Multi-component reward function combining character preservation, affix accuracy, semantic correctness, pattern matching, and length penalties
- **Non-Coding Domain**: Demonstrates RL effectiveness beyond code generation tasks
- **Dakota Language Focus**: Trained on 10,576 grammar tasks extracted from the 1890 Dakota-English Dictionary
- **Special Character Preservation**: Maintains Dakota orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)

## Training Details

### Training Data

- **Source**: 1890 Dakota-English Dictionary grammar section (pages 1-88)
- **Tasks**: 10,576 training tasks covering:
  - Morphology (affix application, word formation)
  - Translation (Dakota ↔ English)
  - Reverse translation
  - Syntax (sentence structure)
  - Pattern identification
- **Difficulty Levels**: Easy (1,973), Medium (5,294), Hard (1,172), Advanced (2,137)

### Training Procedure

- **Framework**: Prime Intellect RL (prime-rl)
- **Base Model**: Qwen/Qwen3-0.6B
- **Training Steps**: 1,000 steps
- **Batch Size**: 256
- **Sequence Length**: 1,536 tokens
- **Rollouts per Example**: 8
- **Learning Rate**: 1e-6
- **Checkpoint Interval**: Every 100 steps (kept 3 most recent)
- **GPUs**: 
  - Trainer: GPUs 4,5,6,7
  - Inference: GPUs 0,1,2,3

### Reward Function Composition

The model was trained using a **compositional reward function** with the following components:

1. **Exact Match Reward** (40% weight): Binary reward for exact normalized match
2. **Character Overlap Reward** (20% weight): F1 score for Dakota special character preservation
3. **Pattern Reward** (15% weight): Verification pattern matching and hint coverage
4. **Affix Reward** (10% weight): Accuracy of required morphological affixes
5. **Length Penalty Reward** (15% weight): Penalizes overly verbose responses (linear decay for responses >3x expected length)

This multi-component approach allows the model to learn nuanced linguistic patterns while maintaining grammatical correctness and orthographic accuracy.

### Environment

- **Environment**: `harleycooper/dakota1890` (v0.1.17)
- **Framework**: Verifiers-compatible RL environment
- **Parser**: DakotaTranslationParser (preserves Dakota orthography)

## Evaluation

### Training Metrics

- **Final Entropy**: 0.2126 (mean), 0.00813 (median)
- **Inference Probabilities**: 0.87743 (mean), 0.99876 (median)
- **Throughput**: 7,800 tokens/s
- **Model FLOPS Utilization (MFU)**: 2.6%
- **Peak Memory**: 11.5 GiB

### W&B Run

- **Project**: dakota-rl-grammar
- **Run Name**: dakota-0.6b-rl
- **View**: [W&B Run](https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/7nikv4vp)

## Intended Use

This model is intended for:
- Research on RL for non-coding linguistic tasks
- Testing compositional reward functions in RL pipelines
- Dakota language grammar and translation tasks
- Demonstrating RL effectiveness beyond code generation domains

## Limitations

- Small model size (0.6B parameters) limits capacity for complex grammar rules
- Trained on historical dictionary data (1890) which may not reflect modern Dakota usage
- Limited to single-turn and multi-turn chat formats
- Requires Dakota language knowledge for proper evaluation

## Ethical Considerations

- Trained on historical linguistic data from indigenous language documentation
- Should be used respectfully and in consultation with Dakota language communities
- Not intended to replace human language experts or native speakers

## Citation

```bibtex
@misc{dakota1890-rl-2024,
  title={Qwen3-0.6B-Dakota-Grammar-RL: A Compositional Reward RL Test for Non-Coding Tasks},
  author={Christian H. Cooper},
  year={2024},
  url={https://huggingface.co/harleycooper/Qwen3-0.6B-Dakota-Grammar-RL}
}
```

## Acknowledgments

- Base model: Qwen/Qwen3-0.6B by Alibaba Cloud
- Training framework: Prime Intellect RL
- Source material: 1890 Dakota-English Dictionary by Stephen Return Riggs
- Environment: Dakota1890 RL environment

## Model Card Contact

For questions or issues, please contact: Raise an Issue in the Repo
