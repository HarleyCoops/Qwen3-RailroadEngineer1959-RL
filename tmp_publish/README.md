---
language:
- en
- dak
license: apache-2.0
library_name: transformers
tags:
- text-generation
- pytorch
- english
- dakota
- qwen3
- reinforcement-learning
- rl
- dakota-language
- grammar
- composition-rewards
- non-coding
- qualitative-tasks
- grpo
- prime-intellect
- verifiers
- conversational
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
datasets:
- HarleyCooper/Dakota1890-Grammar
model_name: Qwen3-30B-ThinkingMachines-Dakota1890
preview_image: grammar.jpg
widget:
- text: "Translate 'my elder brother' to Dakota."
  example_title: "Translation"
- text: "Translate 'He is writing a letter' to Dakota."
  example_title: "Sentence Generation"
---

# Qwen3-30B-ThinkingMachines-Dakota1890

![Dakota Grammar - High Detail Scan](grammar.jpg)

_Exceptional level of detail preserved from the 1890 source material — every character, accent, and linguistic nuance captured with precision_

## Model Description

This model is a reinforcement learning (RL) fine-tuned version of `Qwen/Qwen3-30B-A3B-Instruct`, trained specifically for Dakota language grammar and translation tasks using **GRPO (Group Relative Policy Optimization) with compositional reward functions on qualitative linguistic tasks**.

This run was executed on the **Thinking Machines** infrastructure (Tinker), scaling the methodology to a 30B parameter model to test the limits of morphological learning and reasoning in low-resource language scenarios.

### Key Features

* **State-of-the-Art Morphology**: Achieved **100% accuracy** on affix application rules, surpassing smaller models.
* **Reasoning-Based Grammar**: Leverages the 30B parameter scale to internalize complex linguistic rules rather than just memorizing patterns.
* **Compositional Rewards**: Multi-component reward function combining character preservation (40%), morphological accuracy (40%), and semantic correctness (20%).
* **Dakota Language Focus**: Trained on 5,657 grammar tasks extracted from the 1890 Dakota-English Dictionary.
* **Special Character Preservation**: Maintains Dakota orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.).

**Complete project repository with all code, data, and training traces:**
[https://github.com/HarleyCoops/Dakota1890](https://github.com/HarleyCoops/Dakota1890)

## Training Details

### Training Data

* **Source**: 1890 Dakota-English Dictionary grammar section (pages 31-92)
* **Tasks**: 5,657 training tasks covering:
   * Morphology (affix application, word formation)
   * Translation (Dakota ↔ English)
   * Reverse translation
   * Syntax (sentence structure)
   * Pattern identification

### Training Procedure

* **Infrastructure**: Thinking Machines (Tinker)
* **Algorithm**: GRPO (Group Relative Policy Optimization)
* **Base Model**: Qwen/Qwen3-30B-A3B-Instruct-2507
* **Training Steps**: 199 steps (Checkpoint `tinker://da1ef918.../final`)
* **Batch Size**: 48
* **Group Size**: 16 rollouts per example
* **Max Tokens**: 384 (Generation)

### Reward Function Composition

The model was trained using a **compositional reward function** that decomposes qualitative linguistic tasks into verifiable quantitative components:

1. **Character Preservation (40% weight)**: Verifiable Unicode-level correctness for Dakota special characters.
2. **Morphological Accuracy (40% weight)**: Pattern-matching against grammar rules for affix application.
3. **Semantic Correctness (20% weight)**: Meaning preservation metrics for translation quality.

### Environment

* **Environment**: `dakota_grammar_translation`
* **Framework**: Prime Intellect RL (prime-rl) compatible

## Training Results

### Key Achievements

* **200%+ Improvement**: Composite reward increased from 0.105 to 0.317 (Peak: 0.442).
* **Perfect Morphology**: Affix accuracy reached **1.000 (100%)**, demonstrating the 30B model's superior ability to handle complex grammar rules compared to the 0.6B model (97.9%).
* **Character Mastery**: Character preservation improved from 0.265 to **0.699 (Peak)**, a massive gain in handling Dakota's complex orthography.
* **Rapid Convergence**: Significant improvements achieved in fewer than 200 steps.

### Training Visualizations

#### Comprehensive Dashboard
The comprehensive dashboard provides an at-a-glance view of all training metrics, combining reward progression, component performance, loss dynamics, and entropy into a single visualization.

![Comprehensive Dashboard](viz/comprehensive_dashboard.png)

#### Reward Progression
The reward progression visualization demonstrates the learning trajectory over training steps, showing both overall composite reward and individual component breakdown.

![Reward Progression](viz/reward_progression.png)

## GRPO for Qualitative Tasks: Significance

**GRPO is effective for linguistic-structure learning when qualitative goals are expressed as verifiable, compositional rewards.**

### Why This Matters

Qualitative tasks like language learning and grammar have traditionally been considered unsuitable for RL due to subjective evaluation and multi-dimensional quality metrics.

### Our Solution

By decomposing rewards into **linguistic primitives** (character preservation, morphological accuracy, semantic correctness), we transform qualitative tasks into **quantitatively optimizable objectives**.

## Intended Use

* Research on GRPO for qualitative linguistic tasks.
* Dakota language grammar and translation assistance.
* Demonstrating compositional reward functions in RL pipelines.
* Low-resource language preservation.

## Limitations

* Trained on historical dictionary data (1890), which may differ from modern usage.
* 30B model requires significant compute for inference compared to the 0.6B version.
* Hallucination risk remains, though reduced by RL fine-tuning.

## Ethical Considerations

* Trained on historical linguistic data from indigenous language documentation.
* Should be used respectfully and in consultation with Dakota language communities.
* Part of language preservation and revitalization efforts.

## Citation

```bibtex
@misc{dakota1890-rl-30b-2025,
  title={Qwen3-30B-ThinkingMachines-Dakota1890: GRPO for Qualitative Linguistic Tasks},
  author={Christian H. Cooper},
  year={2025},
  url={https://huggingface.co/HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890},
  note={Fine-tuned on Thinking Machines infrastructure}
}
```

## Acknowledgments

* **Stephen Return Riggs**: Original Dakota grammar documentation (1890)
* **Thinking Machines**: Training infrastructure and support
* **PrimeIntellect**: Distributed RL training framework
* **Dakota Language Community**: Ongoing language revitalization efforts
