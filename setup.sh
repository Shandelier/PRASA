#!/bin/bash

# PRASA: Diffusion Gemma Setup & Installation
# Enhanced version with macOS/Apple Silicon & CUDA support

set -e

echo "--------------------------------------------------"
echo "   PRASA: Diffusion Gemma Setup & Installation    "
echo "--------------------------------------------------"

PLATFORM=$(uname)
ARCH=$(uname -m)

echo "Detected Platform: $PLATFORM"
echo "Detected Arch: $ARCH"

# Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

# Base dependencies (torch is needed on all platforms for the transformers logic)
echo "Installing base dependencies..."
if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Installing PyTorch for Apple Silicon (MPS support)..."
    pip install torch torchvision torchaudio
    echo "Installing MLX dependencies..."
    pip install mlx-lm mlx huggingface_hub rich
elif [[ "$PLATFORM" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "Hardware: NVIDIA GPU detected. Installing CUDA-optimized PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install transformers accelerate bitsandbytes huggingface_hub rich
    else
        echo "Hardware: No NVIDIA GPU found. Installing CPU fallback..."
        pip install torch torchvision torchaudio
        pip install transformers huggingface_hub rich
    fi
else
    echo "Hardware: Unsupported or generic platform. Attempting standard install..."
    pip install torch transformers huggingface_hub rich
fi

# Verify PyTorch installation as it was a reported failure point
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Checking requirements.txt for any missed dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing remaining dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Check HF Auth
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "--------------------------------------------------"
    echo "Warning: Hugging Face CLI not authenticated."
    echo "Google DiffusionGemma models are gated. Please run:"
    echo "huggingface-cli login"
    echo "--------------------------------------------------"
else
    echo "Hugging Face authentication confirmed."
fi

echo "--------------------------------------------------"
echo "Setup complete. Run PRASA with: source .venv/bin/activate && python prasa.py"
echo "--------------------------------------------------"
