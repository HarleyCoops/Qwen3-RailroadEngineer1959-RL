---
title: Dakota Grammar RL Demo
emoji: 🪶
colorFrom: indigo
colorTo: gray
sdk: gradio
python_version: "3.10"
models:
  - HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL
---

# Dakota Grammar RL Model Inference

Interactive inference interface for the **Qwen3-0.6B-Dakota-Grammar-RL** model trained using Reinforcement Learning on Dakota language grammar and translation tasks.

## Model Details

- **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Training Method**: Reinforcement Learning with compositional rewards
- **Training Framework**: Prime Intellect RL (prime-rl)
- **Domain**: Dakota language grammar, translation, and morphology
- **Special Features**: 
  - Preserves Dakota orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)
  - Trained on 10,576 grammar tasks from the 1890 Dakota-English Dictionary
  - Compositional reward structure (exact match, character overlap, affix accuracy, pattern matching, length penalty)

## Model Card

Full details available at: [HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL](https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL)

## Usage

Enter a prompt in the text box and click "Generate" to get a response from the model. 

The model is trained to:
- ✅ Translate between Dakota and English
- ✅ Complete grammar exercises
- ✅ Apply morphological affixes correctly
- ✅ Preserve Dakota special characters and orthography

### Example Prompts

- `Translate to Dakota: Hello`
- `Translate to English: Háu`
- `Complete: Wićaŋyaŋpi kta čha`
- `Add the affix -pi to: wićaŋyaŋ`

## Parameters

- **Max Tokens**: Maximum length of generated response (16-256)
- **Temperature**: Controls randomness (0.1 = deterministic, 1.0 = creative)

## Citation

```bibtex
@misc{dakota1890-rl-2024,
  title={Qwen3-0.6B-Dakota-Grammar-RL: A Compositional Reward RL Test for Non-Coding Tasks},
  author={Dakota Language Lab},
  year={2024},
  url={https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL}
}
```

