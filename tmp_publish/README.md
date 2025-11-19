---
license: apache-2.0
language:
- dak
- en
tags:
- reinforcement-learning
- rl
- grpo
- dakota
- indigenous-languages
- thinking-machines
- tinker
base_model: Qwen/Qwen2.5-32B-Instruct
datasets:
- HarleyCooper/Dakota-Grammar-RL
---

# Qwen3-30B-ThinkingMachines-Dakota1890

This model is a **Reinforcement Learning (RL) fine-tune** of Qwen2.5-32B-Instruct, optimized for **Dakota language grammar and morphology**.

It was trained using the **Thinking Machines Tinker** distributed RL pipeline, leveraging the **GRPO (Group Relative Policy Optimization)** algorithm. The training process used a custom verifier environment built from Stephen Return Riggs' 1890 *Dakota Grammar & Dictionary*.

## Model Details

- **Base Model**: Qwen/Qwen2.5-32B-Instruct
- **Architecture**: LoRA Adapter (Rank 64)
- **Training Method**: GRPO (Group Relative Policy Optimization)
- **Training Infrastructure**: Thinking Machines Tinker
- **Language**: Dakota (dak), English (en)
- **License**: Apache 2.0

## Training Data & Methodology

The model was trained on a dataset of **~10,000 RL tasks** generated from the 1890 Dakota Grammar. These tasks focus on:

1.  **Morphology**: Applying prefixes/suffixes (e.g., possessives `-ku`, `-ću`, `-tku`).
2.  **Translation**: Context-aware translation between Dakota and English.
3.  **Character Preservation**: Strict adherence to Dakota orthography (ŋ, š, ć, ź, ž, ʼ).

### Reward Function

The RL training used a composite reward function (`DakotaGrammarRubric`) with the following components:

*   **Character Preservation (20%)**: Verifies correct usage of special Unicode characters.
*   **Affix Accuracy (10%)**: Checks for correct morphological transformations.
*   **Exact Match (40%)**: Rewards precise answers for rigid grammatical tasks.
*   **Pattern Matching (15%)**: Uses regex to verify structural correctness.
*   **Length Penalty (15%)**: Prevents verbosity.

## Performance

(Metrics from the final training run)

*   **Morphological Accuracy**: ~97.9%
*   **Character Preservation**: ~53.5% (on strict exact match of all special chars)
*   **Overall Composite Reward**: 0.349 (up from 0.120 baseline)

## Usage

### With Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-32B-Instruct"
adapter_name = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load adapter
model = PeftModel.from_pretrained(model, adapter_name)

# Inference
prompt = "Translate 'my elder brother' to Dakota using the correct possessive suffix."
messages = [
    {"role": "system", "content": "You are a Dakota language expert."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Thinking Machines Tinker

This checkpoint is also available directly via the Tinker platform:

```python
# Tinker path
tinker_path = "tinker://c5c6be06-cf8d-5177-acdd-1542225b0750:train:0/sampler_weights/final"
```

## Files

*   `adapter_model.safetensors`: The LoRA adapter weights.
*   `adapter_config.json`: Adapter configuration.
*   `tinker_metadata.json`: Metadata from the Thinking Machines training run.

## Citation

If you use this model, please cite the original grammar source:

> Riggs, S. R. (1890). *Dakota Grammar, Texts, and Ethnography*. Washington: Government Printing Office.

And the Thinking Machines / PrimeIntellect RL framework.

