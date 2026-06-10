# PRASA: Printing-press Revolution: Autoregressive Sequential Alternatives

PRASA is a CLI tool for bidirectional text infilling using Google's Diffusion Gemma 26B/9B MoE models. Unlike traditional autoregressive models that generate text like a typewriter (left-to-right), PRASA uses diffusion-based infilling to generate text blocks simultaneously, similar to a printing press.

## How It Works
Traditional LLMs are autoregressive, predicting the next token in a sequence. Diffusion Gemma allows for bidirectional context, where the model can "see" both the prefix and the suffix to intelligently fill in the missing `[INFILL]` segment.

## Requirements & Local Setup
- **Hardware**: Optimized for macOS with Apple Silicon (M4 Pro recommended).
- **Memory**: At least 24GB Unified Memory (uses ~18GB VRAM with 4-bit quantization).
- **Frameworks**: MLX (native Apple Silicon) or Transformers (with bitsandbytes for 4-bit).

### Installation
```bash
pip install -r requirements.txt
# Ensure mlx and mlx-lm are installed for optimal performance on Mac
pip install mlx-lm transformers
```

## Usage
```bash
python prasa.py --template template.txt --model google/diffusion-gemma-27b
```
