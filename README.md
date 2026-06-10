# PRASA: Printing-press Revolution: Autoregressive Sequential Alternatives

PRASA is a professional CLI tool for bidirectional text infilling using Google's Diffusion Gemma 26B/9B/2B models. Unlike traditional autoregressive models that generate text left-to-right, PRASA leverages Diffusion Gemma's bidirectional context capabilities to fill in missing segments intelligently based on both preceding and following text.

## Features
- **Bidirectional Infilling**: Uses `<|prefix|>`, `<|suffix|>`, and `<|fill|>` tokens for authentic Diffusion Gemma FIM (Fill-In-the-Middle) inference.
- **Hardware Optimized**: Native **MLX** support for Apple Silicon (macOS) and **CUDA** support with 4-bit quantization (via bitsandbytes) for Linux/Windows.
- **Production Ready**: Robust error handling, hardware detection, and clean Rich-based CLI interface.

## Quick Start

### Installation
Run the automated setup script which handles environment creation and dependency installation:
```bash
./setup.sh
```
The script will detect your hardware (Apple Silicon vs CUDA) and install the appropriate backends.

### Usage
1. Prepare a `template.txt` file with the `[INFILL]` marker where you want text generated:
   ```text
   A Polish man walks into a bar with a [INFILL]. The bartender says, "We don't serve those here!"
   ```
2. Run the PRASA engine:
   ```bash
   source .venv/bin/activate
   python prasa.py --template template.txt --model 2b
   ```

## Requirements
- **macOS**: Apple Silicon (M1/M2/M3/M4) recommended for MLX acceleration.
- **Linux**: NVIDIA GPU with 8GB+ VRAM for quantized inference.
- **Python**: 3.10+

## Models
PRASA supports the following shorthands for `--model`:
- `2b`: `google/gemma-2-2b-it` (Lightest, recommended for local testing)
- `9b`: `google/gemma-2-9b-it`
- `26b`: `google/gemma-2-27b-it` (Highest quality infilling)

Note: Diffusion Gemma models are gated on Hugging Face. Ensure you have run `huggingface-cli login` and have access to the Gemma model family.
