#!/bin/bash
# Setup script for remote Prime Intellect instance
# Sets required environment variables for GRPO training

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Setting up environment variables for GRPO training ===${NC}\n"

# Check if .env file exists (if running locally)
if [ -f ".env" ]; then
    echo -e "${YELLOW}Found .env file. Loading variables...${NC}\n"
    # Source .env file (handles comments and empty lines)
    set -a
    source .env
    set +a
fi

# Required environment variables for GRPO training
# These will be exported to the current shell session

# WandB API Key (REQUIRED)
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}ERROR: WANDB_API_KEY is not set!${NC}"
    echo -e "${YELLOW}Get your API key from: https://wandb.ai/settings${NC}"
    echo -e "${YELLOW}Then run: export WANDB_API_KEY='your_key_here'${NC}\n"
    exit 1
else
    export WANDB_API_KEY="$WANDB_API_KEY"
    echo -e "${GREEN}[OK] WANDB_API_KEY is set${NC}"
fi

# WandB Project (optional, defaults to dakota-rl-grammar)
if [ -z "$WANDB_PROJECT" ]; then
    export WANDB_PROJECT="dakota-rl-grammar"
    echo -e "${YELLOW}WANDB_PROJECT not set, using default: dakota-rl-grammar${NC}"
else
    export WANDB_PROJECT="$WANDB_PROJECT"
    echo -e "${GREEN}[OK] WANDB_PROJECT=$WANDB_PROJECT${NC}"
fi

# WandB Entity (optional)
if [ -n "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY="$WANDB_ENTITY"
    echo -e "${GREEN}[OK] WANDB_ENTITY=$WANDB_ENTITY${NC}"
else
    echo -e "${YELLOW}WANDB_ENTITY not set (optional)${NC}"
fi

# Hugging Face Token (optional but recommended for model access)
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set (may be needed for model access)${NC}"
    echo -e "${YELLOW}Get your token from: https://huggingface.co/settings/tokens${NC}"
else
    # Use HF_TOKEN if set, otherwise HUGGINGFACE_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN="$HF_TOKEN"
        echo -e "${GREEN}[OK] HF_TOKEN is set${NC}"
    elif [ -n "$HUGGINGFACE_TOKEN" ]; then
        export HF_TOKEN="$HUGGINGFACE_TOKEN"
        export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
        echo -e "${GREEN}[OK] HF_TOKEN is set (from HUGGINGFACE_TOKEN)${NC}"
    fi
fi

# Prime Intellect API Key (optional, for CLI)
if [ -n "$PI_API_KEY" ] || [ -n "$PRIME_API_KEY" ]; then
    if [ -n "$PI_API_KEY" ]; then
        export PI_API_KEY="$PI_API_KEY"
    else
        export PI_API_KEY="$PRIME_API_KEY"
    fi
    echo -e "${GREEN}[OK] PI_API_KEY is set${NC}"
fi

echo -e "\n${GREEN}=== Environment variables set! ===${NC}"
echo -e "\n${YELLOW}To make these permanent, add to ~/.bashrc:${NC}"
echo "export WANDB_API_KEY=\"$WANDB_API_KEY\""
echo "export WANDB_PROJECT=\"$WANDB_PROJECT\""
[ -n "$WANDB_ENTITY" ] && echo "export WANDB_ENTITY=\"$WANDB_ENTITY\""
[ -n "$HF_TOKEN" ] && echo "export HF_TOKEN=\"$HF_TOKEN\""
[ -n "$PI_API_KEY" ] && echo "export PI_API_KEY=\"$PI_API_KEY\""

echo -e "\n${GREEN}Ready to launch training!${NC}\n"

