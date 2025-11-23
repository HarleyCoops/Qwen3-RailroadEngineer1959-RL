# Standalone HF Infrastructure Inference

This script provides standalone inference for the **Qwen3-0.6B-Dakota-Grammar-RL** model using Hugging Face infrastructure (Inference API or Inference Endpoints).

## Features

-  Uses your HF login credentials
-  Supports both Inference API and Inference Endpoints
-  Proper chat format matching training
-  Interactive and single-prompt modes
-  Configurable generation parameters

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements_hf_inference.txt
```

### 2. Authenticate with Hugging Face

You have two options:

**Option A: Login via CLI**
```powershell
huggingface-cli login
```

**Option B: Set Environment Variable**
```powershell
$env:HF_TOKEN = "your_hf_token_here"
```

## Usage

### Single Prompt Mode

```powershell
python hf_inference_standalone.py --prompt "Translate to Dakota: Hello"
```

### Interactive Mode

```powershell
python hf_inference_standalone.py --interactive
```

### Using Inference Endpoints

If you have a dedicated Inference Endpoint:

```powershell
python hf_inference_standalone.py --prompt "Translate to Dakota: Hello" --endpoint-url "https://your-endpoint-url.hf.space"
```

### Advanced Options

```powershell
python hf_inference_standalone.py `
    --prompt "Translate to Dakota: Hello" `
    --max-tokens 128 `
    --temperature 0.5 `
    --top-p 0.95 `
    --repetition-penalty 1.2
```

### JSON Output

```powershell
python hf_inference_standalone.py --prompt "Translate to Dakota: Hello" --json
```

## Command-Line Options

- `--prompt`: Input prompt (required unless `--interactive`)
- `--model-id`: Model ID (default: `HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL`)
- `--endpoint-url`: Inference Endpoint URL (optional)
- `--token`: HF token (optional, uses login if not provided)
- `--system-prompt`: Custom system prompt (optional)
- `--max-tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 0.3)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--repetition-penalty`: Repetition penalty (default: 1.1)
- `--json`: Output as JSON
- `--interactive`: Run in interactive mode

## Examples

### Translation Tasks

```powershell
# English to Dakota
python hf_inference_standalone.py --prompt "Translate to Dakota: Hello"

# Dakota to English
python hf_inference_standalone.py --prompt "Translate to English: Háu"
```

### Grammar Tasks

```powershell
# Grammar completion
python hf_inference_standalone.py --prompt "Complete: Wićaŋyaŋpi kta čha"

# Morphology
python hf_inference_standalone.py --prompt "Add the affix -pi to: wićaŋyaŋ"
```

## Using as a Python Module

You can also import and use the client in your own code:

```python
from hf_inference_standalone import DakotaInferenceClient

# Initialize client
client = DakotaInferenceClient()

# Generate response
result = client.generate(
    prompt="Translate to Dakota: Hello",
    max_new_tokens=64,
    temperature=0.3
)

print(result["response"])
```

## Inference API vs Inference Endpoints

### Inference API (Default)
-  Free tier available
-  No setup required
-  Shared infrastructure
-  Rate limits may apply
-  Slower for high-throughput

### Inference Endpoints
-  Dedicated resources
-  Better performance
-  Custom scaling
-  Requires setup and configuration
-  Paid service

## Troubleshooting

### Authentication Errors

If you see authentication errors:

1. Check your token:
```powershell
huggingface-cli whoami
```

2. Re-login:
```powershell
huggingface-cli login
```

3. Or set token explicitly:
```powershell
$env:HF_TOKEN = "your_token"
```

### Model Not Found

Ensure the model is public or you have access:
- Model: https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL

### Rate Limiting

If you hit rate limits:
- Use Inference Endpoints for dedicated resources
- Add delays between requests
- Use HF's paid tier for higher limits

## Chat Format Matching

**Important**: The chat format is automatically matched to training. The script uses the model's built-in chat template (via `tokenizer.apply_chat_template()`), which is the same template used during RL training by the verifiers framework. See `docs/CHAT_FORMAT_EXPLANATION.md` for details.

## Model Details

- **Model**: [HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL](https://huggingface.co/HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL)
- **Base Model**: Qwen/Qwen3-0.6B
- **Training**: Reinforcement Learning with compositional rewards
- **Domain**: Dakota language grammar and translation
- **Special Features**: Preserves Dakota orthography (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)

## Citation

```bibtex
@misc{dakota1890-rl-2024,
  title={Qwen3-0.6B-Dakota-Grammar-RL: A Compositional Reward RL Test for Non-Coding Tasks},
  author={Christian H. Cooper},
  year={2024},
  url={https://huggingface.co/harleycooper/Qwen3-0.6B-Dakota-Grammar-RL}
}
```

