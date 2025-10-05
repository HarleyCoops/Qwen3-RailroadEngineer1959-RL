# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository focuses on **Qwen/Qwen3-VL-235B-A22B-Thinking**, a 235B parameter MoE (Mixture of Experts) vision-language model with reasoning capabilities. The project provides multi-provider integration, academic documentation, and practical tools for working with this reasoning-optimized vision model.

**Key Goal**: Build tools and understanding to enable multimodal reasoning agents, particularly for learning the Blackfeet language from historical dictionary images.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

### Testing Inference Providers

**OpenRouter (Primary - Supports Thinking Budget)**
```bash
# Requires: OPENROUTER_API_KEY in .env
python implementation/examples/openrouter_integration.py
```

**Hugging Face (Local or Endpoint)**
```bash
# Requires: HF_API_KEY in .env
# For local: Requires serious GPU resources (235B MoE model)
# Prefer: Hugging Face Inference Endpoint (set HF_INFERENCE_ENDPOINT)
python -c "from implementation.inference_connector import InferenceConnector; print(InferenceConnector().infer('huggingface', 'Who are you?'))"
```

**Hyperbolic Labs**
```bash
# Requires: HYPERBOLIC_API_KEY and HYPERBOLIC_ENDPOINT in .env
python implementation/examples/hyperbolic_connection.py
```

### Validation and Documentation

**Validate Model Cards**
```bash
python tools/validators/model_card_validator.py implementation/model_cards/*.md
```

**Update Progress Tracking**
```bash
python tools/update_progress.py
```

**Fix Markdown Formatting**
```bash
python tools/fix_markdown.py
```

**Download External Resources**
```bash
python tools/download_dictionary.py <URL>
# Example: python tools/download_dictionary.py https://pubs.usgs.gov/unnumbered/70037986/report.pdf
```

## Architecture

### Directory Structure

- **`implementation/`**: Core inference connectors and provider integrations
  - `inference_connector.py`: Unified multi-provider inference class (`InferenceConnector`)
  - `examples/openrouter_integration.py`: Primary OpenRouter client with thinking budget controls (`Qwen3VLClient`)
  - `examples/hyperbolic_connection.py`: Hyperbolic Labs integration
  - `model_cards/`: Model card templates

- **`academic/`**: Research paper analysis and academic documentation
  - Deep dive into Qwen architecture, benchmarks, and innovations

- **`tools/`**: Automation and validation utilities
  - `download_dictionary.py`: Download external resources with progress tracking
  - `fix_markdown.py`: Markdown formatting automation
  - `update_progress.py`: Progress tracking automation
  - `validators/model_card_validator.py`: Model card validation

- **`e2b/`**: E2B sandbox deployment (experimental self-hosted alternative)

### Key Components

**1. InferenceConnector (`implementation/inference_connector.py`)**

Unified interface for routing inference requests to different providers:
- `infer_huggingface()`: Local transformers or HF Inference Endpoint
- `infer_openrouter()`: OpenRouter API with reasoning token support
- `infer_hyperbolic()`: Hyperbolic Labs deployment
- `infer(provider, prompt, image_data, **kwargs)`: Unified entrypoint

**2. Qwen3VLClient (`implementation/examples/openrouter_integration.py`)**

OpenRouter-specific client with rich thinking budget controls:
- `chat()`: Text-only prompts
- `analyze_image()`: Single image analysis
- `analyze_document()`: Document processing
- `process_video()`: Video frame sequence reasoning
- Thinking budget options: int (token count), str ("low"/"medium"/"high"), or dict

**3. Reasoning Token Controls (OpenRouter)**

The Qwen3-VL Thinking model supports extended reasoning via OpenRouter's API:
```python
# Token budget
payload["reasoning"] = {"max_tokens": 2048}
payload["include_reasoning"] = True

# Or effort level
payload["reasoning"] = {"effort": "high"}  # low, medium, high

# Exclude reasoning from response
payload["reasoning"] = {"exclude": True}
```

Environment variables:
- `OPENROUTER_REASONING_MAX_TOKENS`: Default token budget
- `OPENROUTER_REASONING_EFFORT`: Default effort level (low/medium/high)
- `OPENROUTER_REASONING_EXCLUDE`: Exclude reasoning tokens (true/false)
- `OPENROUTER_INCLUDE_REASONING`: Include reasoning in response (default: true)

## Provider Integration Pattern

When adding a new inference provider:

1. Add environment variables to `.env.template` (API key, endpoint)
2. Add provider-specific method to `InferenceConnector` (e.g., `infer_newprovider()`)
3. Update `infer()` method to route to new provider
4. Create example script in `implementation/examples/newprovider_integration.py`
5. Document in README.md under "Inference Providers Integration"
6. Update PROGRESS.md checklist

## Multimodal Input Handling

**OpenRouter Format:**
```python
content_blocks = [
    {
        "type": "input_image",
        "image": {
            "data": base64_encoded_string,
            "media_type": "image/png"  # or jpeg, webp, etc.
        }
    },
    {"type": "input_text", "text": "Your prompt"}
]
```

**Hugging Face Transformers Format:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": PIL_Image_or_path},
            {"type": "text", "text": "Your prompt"}
        ]
    }
]
```

Helper methods in `InferenceConnector`:
- `_ensure_base64_image()`: Converts Path/bytes/str to base64
- `_prepare_transformers_image()`: Converts to PIL Image
- `_build_openrouter_content_blocks()`: Builds OpenRouter message format
- `_build_transformers_messages()`: Builds HF transformers format

## Model Specifications

**Model ID**: `Qwen/Qwen3-VL-235B-A22B-Thinking`
- 235B parameters (MoE architecture with 22B active)
- Vision + Language + Reasoning capabilities
- Supports images, documents, video frames
- OpenRouter slug: `qwen/qwen3-vl-235b-a22b-thinking`

**Hardware Requirements (Local)**:
- Multiple high-memory GPUs (A100 80GB recommended)
- `device_map="auto"` for automatic distribution
- `torch_dtype=torch.bfloat16` or `torch.float32`
- Flash Attention 2 for efficiency: `attn_implementation="flash_attention_2"`

**Prefer Hosted Inference**: Most users should use OpenRouter, HF Inference Endpoints, or Hyperbolic rather than running locally.

## Automated Workflows

**GitHub Actions** (`.github/workflows/documentation.yml`):
- Runs hourly, on push to main, on PRs, or manual dispatch
- Updates progress tracking (`tools/update_progress.py`)
- Validates model cards (`tools/validators/model_card_validator.py`)
- Runs markdownlint with auto-fix
- Creates documentation PRs automatically
- Skips commits with `[skip ci]` to prevent loops

## Important Environment Variables

Required for development:
- `OPENROUTER_API_KEY`: Primary inference path (supports thinking budget)
- `HF_API_KEY`: Hugging Face authentication (for local or endpoints)

Optional:
- `QWEN_VL_MODEL_ID`: Override default model (default: `Qwen/Qwen3-VL-235B-A22B-Thinking`)
- `QWEN_OPENROUTER_MODEL_ID`: Override OpenRouter model slug
- `HF_INFERENCE_ENDPOINT`: Hugging Face hosted endpoint URL
- `HYPERBOLIC_API_KEY`, `HYPERBOLIC_ENDPOINT`: Hyperbolic Labs deployment
- `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`: OpenRouter metadata headers

See `.env.template` for full list.

## Development Notes

**Philosophy**: Prioritize exploration and learning over pure efficiency. Build comprehensive tools, understand underlying mechanisms, and document thoroughly.

**Key Learning Areas**:
1. Vision-language model architecture (window attention, MRoPE, dynamic FPS)
2. Reasoning token mechanics (thinking budgets)
3. Multi-provider API integration patterns
4. Multimodal prompt engineering

**Future Focus**:
- Fine-tuning methodologies for vision models
- Dataset preparation for multimodal tasks
- MCP (Model Context Protocol) server development
- Agent-based reasoning workflows

## Testing and Validation

No formal test suite yet. Validation is done through:
- Example scripts in `implementation/examples/`
- Model card validators in `tools/validators/`
- Manual testing with different providers

**To validate changes**:
1. Test with OpenRouter: `python implementation/examples/openrouter_integration.py`
2. Run validators: `python tools/validators/model_card_validator.py implementation/model_cards/*.md`
3. Check documentation: `python tools/fix_markdown.py`

## Common Pitfalls

1. **Local Inference**: Don't attempt local HF inference without serious GPU resources. Use OpenRouter or HF Inference Endpoints instead.
2. **Reasoning Budget Conflicts**: Don't set both `max_tokens` and `effort` in reasoning payload—choose one.
3. **Image Encoding**: Ensure images are properly base64-encoded with correct media types.
4. **Git Loops**: Always use `[skip ci]` in automated commit messages to prevent workflow loops.
5. **API Keys**: Never commit `.env` file. Always use `.env.template` for examples.

## Quick Reference: Inference Examples

**Text-only (OpenRouter)**:
```python
from implementation.examples.openrouter_integration import Qwen3VLClient
import os

client = Qwen3VLClient(os.getenv("OPENROUTER_API_KEY"))
response = client.chat("Explain quantum entanglement", thinking_budget=2048)
print(response["text"])
print(f"Reasoning tokens: {response['reasoning_tokens']}")
```

**Image analysis (OpenRouter)**:
```python
from pathlib import Path
response = client.analyze_image(
    Path("document.png"),
    "Summarize this document",
    thinking_budget="high"
)
print(response["text"])
```

**Unified connector (any provider)**:
```python
from implementation.inference_connector import InferenceConnector

connector = InferenceConnector()
result = connector.infer("openrouter", "Who are you?")
print(result["content"])
```

## Additional Resources

- Original paper analysis: `academic/paper_analysis/`
- Progress tracking: `PROGRESS.md`
- Setup instructions: `instructions.txt`
- Model card: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking
- OpenRouter reasoning docs: https://openrouter.ai/docs/use-cases/reasoning-tokens
- Transformers docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
