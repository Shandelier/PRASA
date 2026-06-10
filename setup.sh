#!/bin/bash

# PRASA local setup script
# Supports macOS (Apple Silicon) and Linux (CUDA/CPU)

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

# Dependency selection
if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Hardware: Apple Silicon detected. Installing MLX dependencies..."
    pip install mlx-lm mlx huggingface_hub rich
elif [[ "$PLATFORM" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "Hardware: NVIDIA GPU detected. Installing CUDA-optimized dependencies..."
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

# Verification
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "from transformers import pipeline; print('Transformers available.')"

# Check HF Auth
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Warning: Hugging Face CLI not authenticated. Google Gemma models are gated."
    echo "Please run: huggingface-cli login"
else
    echo "Hugging Face authentication confirmed."
fi

echo "--------------------------------------------------"
echo "Setup complete. Run PRASA with: source .venv/bin/activate && python prasa.py --template template.txt"
echo "--------------------------------------------------"
