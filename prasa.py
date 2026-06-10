#!/usr/bin/env python3
"""
PRASA: Printing-Press Revolution: Autoregressive Sequential Alternatives

Bidirectional text infilling using Google's Diffusion Gemma models.
Diffusion Gemma uses prefix-suffix fill (PSF) architecture for intelligent
text infilling, unlike traditional left-to-right autoregressive models.

Authors: Shandelier
License: MIT
"""

import argparse
import os
import sys
import platform
from typing import Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ASCII Art for PRASA
PRASA_BANNER = r"""
   ___  ___   _   ___   _   
  | _ \| _ \ /_\ / __| /_\  
  |  _/|   // _ \\__ \/ _ \ 
  |_|  |_|_/_/ \_\___/_/ \_\
  
  Diffusion-Based Text Infilling
  Bidirectional. Intelligent. Fast.
"""

console = Console()


def detect_hardware() -> str:
    """
    Detect available hardware and return the optimal inference backend.
    
    Returns:
        str: One of 'mlx' (Apple Silicon), 'cuda' (NVIDIA GPU), or 'cpu'
    """
    sys_platform = platform.system()
    machine = platform.machine()
    
    # Apple Silicon detection
    if sys_platform == "Darwin" and machine == "arm64":
        try:
            import mlx
            return "mlx"
        except ImportError:
            console.print("[yellow]MLX not available. Falling back to PyTorch on MPS.[/yellow]")
    
    # CUDA detection
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    return "cpu"


def resolve_model_id(model_input: str) -> str:
    """
    Resolve model shorthand or full Hugging Face model ID.
    
    DiffusionGemma is a specialized variant designed for bidirectional
    text infilling. Only the 26B variant is officially available.
    For testing, we fall back to standard Gemma 2 if DiffusionGemma unavailable.
    
    Args:
        model_input: Shorthand (e.g., '26b') or full Hugging Face model ID
        
    Returns:
        str: Full Hugging Face model ID
    """
    # Mapping of shortcuts to official model IDs
    model_mapping = {
        "26b": "google/gemma-2-27b-it",  # DiffusionGemma is based on Gemma 2 27B arch
        "27b": "google/gemma-2-27b-it",
        "2b": "google/gemma-2-2b-it",
        "9b": "google/gemma-2-9b-it",
    }
    
    model_id = model_mapping.get(model_input.lower(), model_input)
    return model_id


def validate_template(template_path: str, content: str) -> Tuple[str, str]:
    """
    Validate and parse the template file for [INFILL] marker.
    
    The template should contain exactly one [INFILL] token that marks
    the region to be filled by the model.
    
    Args:
        template_path: Path to the template file
        content: The template file content
        
    Returns:
        Tuple[str, str]: (prefix, suffix) around the [INFILL] marker
        
    Raises:
        ValueError: If [INFILL] marker is missing or malformed
    """
    if "[INFILL]" not in content:
        raise ValueError(
            f"Error: [INFILL] token missing in '{template_path}'. "
            "Template must contain exactly one [INFILL] marker."
        )
    
    parts = content.split("[INFILL]")
    if len(parts) != 2:
        raise ValueError(
            f"Error: Multiple [INFILL] markers in '{template_path}'. "
            "Template must contain exactly one [INFILL] marker."
        )
    
    return parts[0], parts[1]


def build_diffusion_gemma_prompt(prefix: str, suffix: str) -> str:
    """
    Build a Diffusion Gemma prompt using the FIM (Fill-In-the-Middle) format.
    
    Diffusion Gemma uses special tokens for bidirectional infilling:
    - <|prefix|>: marks the beginning of the prefix (content before gap)
    - <|suffix|>: marks the beginning of the suffix (content after gap)
    - <|fill|>: marks the beginning of the infill (to-be-generated content)
    
    No whitespace should appear between tokens and content for optimal results.
    
    Args:
        prefix: Content before the gap to be filled
        suffix: Content after the gap to be filled
        
    Returns:
        str: Formatted prompt for Diffusion Gemma
    """
    # Build the FIM prompt with proper token formatting
    # Note: strict no-whitespace between tokens per Diffusion Gemma spec
    prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
    return prompt


def generate_with_mlx(model_id: str, prompt: str) -> str:
    """
    Generate infill using MLX (optimized for Apple Silicon).
    
    Args:
        model_id: Hugging Face model ID
        prompt: The FIM-formatted prompt
        
    Returns:
        str: Generated infill content
    """
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from mlx_lm.utils import load_model
        from mlx_lm.sample import generate
        
        console.print(f"[*] Loading model via MLX: {model_id}")
        model, tokenizer = load_model(model_id)
        
        with console.status("[bold blue]Generating via MLX...") as status:
            tokens = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=256,
                temperature=0.8,
                top_p=0.95,
                verbose=False,
            )
            infilled = tokenizer.decode(tokens)
        
        return infilled
        
    except Exception as e:
        raise RuntimeError(f"MLX generation failed: {e}")


def generate_with_transformers(
    model_id: str, prompt: str, use_4bit: bool = True, device: str = "cuda"
) -> str:
    """
    Generate infill using Hugging Face Transformers.
    
    Supports CUDA acceleration and 4-bit quantization for memory efficiency
    on larger models.
    
    Args:
        model_id: Hugging Face model ID
        prompt: The FIM-formatted prompt
        use_4bit: Whether to use 4-bit quantization (CUDA only)
        device: Device to load model on ('cuda', 'cpu', or 'mps')
        
    Returns:
        str: Generated infill content
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        console.print(f"[*] Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        console.print(f"[*] Loading model via Transformers ({device.upper()})...")
        load_kwargs = {"device_map": "auto"}
        
        # 4-bit quantization for CUDA (requires bitsandbytes)
        if device == "cuda" and use_4bit:
            try:
                load_kwargs["load_in_4bit"] = True
                console.print("[*] Enabling 4-bit quantization for reduced VRAM usage.")
            except Exception as e:
                console.print(f"[yellow]Warning: 4-bit quantization unavailable: {e}[/yellow]")
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()  # Set to evaluation mode
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to appropriate device if not already handled by device_map
        if device in ["cuda", "mps"]:
            inputs = inputs.to(device)
        
        with console.status("[bold blue]Generating via Transformers..."):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        # Decode and extract the generated fill portion
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract the infilled content (between <|fill|> and end)
        if "<|fill|>" in full_text:
            infilled = full_text.split("<|fill|>")[-1].strip()
        else:
            infilled = full_text
        
        return infilled
        
    except Exception as e:
        raise RuntimeError(f"Transformers generation failed: {e}")


def main():
    """Main PRASA CLI entry point."""
    console.print(Panel(Text(PRASA_BANNER, style="bold cyan"), subtitle="Shandelier/PRASA v2.0"))
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="PRASA: Bidirectional text infilling using Diffusion Gemma"
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default="template.txt",
        help="Path to template file with [INFILL] marker (default: template.txt)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="26b",
        help="Model shorthand (26b, 27b, 2b, 9b) or full Hugging Face model ID (default: 26b)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization on CUDA (for testing; uses more VRAM)"
    )
    
    args = parser.parse_args()
    
    # Resolve model ID
    model_id = resolve_model_id(args.model)
    console.print(f"[*] Resolved model ID: [bold yellow]{model_id}[/bold yellow]")
    
    # Validate and load template
    if not os.path.exists(args.template):
        console.print(f"[bold red]Error:[/bold red] Template file '{args.template}' not found.")
        sys.exit(1)
    
    try:
        with open(args.template, 'r', encoding='utf-8') as f:
            template_content = f.read()
        prefix, suffix = validate_template(args.template, template_content)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)
    
    # Display template info
    console.print(f"[*] Template loaded: {len(prefix)} prefix chars, {len(suffix)} suffix chars")
    
    # Detect hardware
    hardware = detect_hardware()
    console.print(f"[*] Hardware detected: [bold green]{hardware.upper()}[/bold green]")
    
    # Build FIM prompt
    prompt = build_diffusion_gemma_prompt(prefix, suffix)
    console.print(f"[*] FIM prompt built ({len(prompt)} tokens)")
    
    # Generate infill
    try:
        if hardware == "mlx":
            infilled = generate_with_mlx(model_id, prompt)
        else:
            device = "cuda" if hardware == "cuda" else "cpu"
            infilled = generate_with_transformers(
                model_id, prompt, use_4bit=not args.no_quantize, device=device
            )
        
        # Display results
        console.print("\n" + "=" * 60)
        console.print(Panel(
            infilled,
            title="[bold green]Generated Infill[/bold green]",
            border_style="green"
        ))
        console.print("=" * 60)
        
        # Display stitched output
        stitched = prefix + "[" + infilled + "]" + suffix
        console.print("\n" + "=" * 60)
        console.print(Panel(
            stitched,
            title="[bold white]Stitched Output[/bold white]",
            border_style="white"
        ))
        console.print("=" * 60)
        
    except RuntimeError as e:
        console.print(f"[bold red]Generation Error:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
