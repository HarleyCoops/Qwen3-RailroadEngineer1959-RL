# E2B Deployment for Qwen3-VL Thinking

This directory contains the E2B (secure sandbox) deployment implementation for running Qwen3-VL Thinking in an isolated environment.

## Overview

E2B provides a secure, sandboxed environment where the Qwen3-VL Thinking model can be run with:
- Complete isolation from the host system
- File system access within the sandbox
- Long-running processes
- Full control over the environment

## Directory Structure

```
e2b/
├── README.md              # This file
├── e2b_connector.py       # Main connector class for sandbox management
├── main.py                # Example usage script
└── sandbox_setup/
    ├── requirements.txt   # Python dependencies for the sandbox
    └── setup_model.py     # Model initialization script
```

## Prerequisites

1. **E2B Account & API Key**
   - Sign up at [E2B Dashboard](https://e2b.dev/dashboard)
   - Generate an API key from your dashboard
   - Free tier available for testing

2. **Python Environment**
   - Python 3.8 or higher
   - pip package manager

3. **Dependencies**
   - All required packages listed in `../requirements.txt`

## Installation

### 1. Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

This will install the E2B SDK along with all other project dependencies.

### 2. Configure Environment Variables

Copy the environment template and add your E2B API key:

```bash
# Copy template
cp .env.template .env

# Edit .env and add your E2B API key
# E2B_API_KEY=your_actual_e2b_api_key_here
```

Get your API key from: https://e2b.dev/dashboard

### 3. Optional: Hugging Face Token

If you need to access private models or increase rate limits:

```bash
# Add to your .env file
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Basic Usage

Run the example script:

```bash
python e2b/main.py
```

This will:
1. Start an E2B sandbox
2. Upload setup files
3. Install dependencies in the sandbox
4. Download and initialize the Qwen3-VL Thinking model
5. Run sample inference
6. Clean up and close the sandbox

### Programmatic Usage

```python
from e2b.e2b_connector import E2BConnector

# Using context manager (recommended)
with E2BConnector() as connector:
    # Upload and setup
    connector.upload_setup_files()
    connector.install_dependencies()
    connector.setup_model()
    
    # Run inference
    result = connector.run_inference(
        prompt="Describe this image",
        image_path="path/to/image.jpg"
    )
    print(result)

# Manual management (alternative)
connector = E2BConnector()
try:
    connector.start_sandbox()
    # ... perform operations ...
finally:
    connector.close()
```

## Configuration

### Sandbox Timeout

Adjust the timeout for long-running operations:

```python
connector = E2BConnector(timeout=600)  # 10 minutes
```

### Model Selection

Edit `sandbox_setup/setup_model.py` to change the model:

```python
model_name = "Qwen/Qwen3-VL-7B-Instruct"  # Change this line
```

Available models:
- `Qwen/Qwen3-VL-7B-Instruct` (default)
- `Qwen/Qwen3-VL-14B-Instruct`
- `Qwen/Qwen3-VL-72B-Instruct`

### Sandbox Dependencies

Modify `sandbox_setup/requirements.txt` to add or remove packages installed in the sandbox.

## Architecture

### E2B Connector (`e2b_connector.py`)

Main class that manages the sandbox lifecycle:

- **Initialization**: Sets up API connection and configuration
- **Sandbox Management**: Starts, monitors, and closes sandbox sessions
- **File Operations**: Uploads files to and from the sandbox
- **Execution**: Runs commands and scripts within the sandbox
- **Inference**: Handles model inference requests

### Setup Script (`sandbox_setup/setup_model.py`)

Executed inside the sandbox to:
1. Download the model from Hugging Face
2. Initialize tokenizer and model
3. Verify setup
4. Cache for future use

### Main Script (`main.py`)

Example implementation demonstrating:
- Complete setup workflow
- Inference execution
- Error handling
- Resource cleanup

## Troubleshooting

### API Key Issues

**Error**: "E2B_API_KEY must be provided"

**Solution**:
1. Verify `.env` file exists
2. Check E2B_API_KEY is set correctly
3. Ensure python-dotenv is installed
4. Load environment variables: `from dotenv import load_dotenv; load_dotenv()`

### Sandbox Timeout

**Error**: "Timeout waiting for sandbox"

**Solution**:
1. Increase timeout parameter
2. Check E2B service status
3. Verify network connectivity

### Model Download Issues

**Error**: "Model download failed"

**Solution**:
1. Check internet connectivity in sandbox
2. Verify model name is correct
3. For private models, ensure HF_TOKEN is set
4. Check available disk space

### GPU Unavailability

**Warning**: "CUDA not available - using CPU"

**Note**: E2B sandboxes may not have GPU access by default. Performance will be slower on CPU.

**Solution**:
- Use smaller models (7B instead of 72B)
- Enable GPU sandbox instances (may require paid plan)
- Use external inference providers for GPU access

## Performance Considerations

1. **First-Time Setup**: Model download can take 5-15 minutes depending on model size
2. **Caching**: Subsequent runs reuse cached models (faster)
3. **CPU vs GPU**: GPU inference is 10-100x faster
4. **Network**: Upload/download speeds affect setup time

## Security Features

- **Isolation**: Complete sandbox isolation from host system
- **No Persistence**: Sandboxes are ephemeral by default
- **Controlled Access**: Only explicitly uploaded files are available
- **Resource Limits**: Sandboxes have CPU and memory constraints

## Limitations

1. **GPU Access**: Limited or unavailable in free tier
2. **Storage**: Ephemeral storage (lost when sandbox closes)
3. **Network**: Outbound connections allowed, inbound may be restricted
4. **Runtime**: Sandboxes have maximum lifetime limits

## Cost Considerations

- Free tier available for testing
- Paid plans for:
  - Extended runtime
  - GPU access
  - Higher resource limits
  - Persistent storage

Check current pricing: https://e2b.dev/pricing

## Alternative Deployment Options

If E2B doesn't meet your needs, consider:

1. **Cloud Inference Providers** (See main README.md):
   - Hyperbolic Labs
   - Hugging Face Inference Endpoints
   - OpenRouter
   - Replicate

2. **Self-Hosted**:
   - Docker containers
   - Kubernetes deployments
   - Virtual machines

3. **Local Development**:
   - Direct Python execution
   - Jupyter notebooks

## Next Steps

1. **Customize Setup**: Modify `setup_model.py` for your specific requirements
2. **Implement Inference**: Update the inference logic in `e2b_connector.py`
3. **Add Features**: Extend with batch processing, streaming, etc.
4. **Production Ready**: Add monitoring, logging, and error recovery
5. **Integrate**: Connect with your application workflow

## Resources

- [E2B Documentation](https://e2b.dev/docs)
- [E2B Python SDK](https://github.com/e2b-dev/e2b)
- [Qwen3-VL documentation](https://huggingface.co/Qwen)
- [Project Main README](../README.md)

## Support

- E2B Issues: https://github.com/e2b-dev/e2b/issues
- Project Issues: Create issue in this repository
- E2B Discord: https://discord.gg/U7KEcGErtQ

## License

This E2B integration follows the project's main license (MIT).


