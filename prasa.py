import argparse
import os
import sys
import platform
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ASCII Art for PRASA
PRASA_BANNER = """
   ___  ___   _   ___   _   
  | _ \| _ \ /_\ / __| /_\  
  |  _/|   // _ \\__ \/ _ \ 
  |_|  |_|_/_/ \_\___/_/ \_\\
  Local Diffusion Gemma Press
"""

def detect_hardware():
    sys_platform = platform.system()
    machine = platform.machine()
    
    if sys_platform == "Darwin" and machine == "arm64":
        return "mlx"
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
        
    return "cpu"

def resolve_model_id(model_input):
    mapping = {
        "2b": "google/diffusion-gemma-2b-it",
        "9b": "google/diffusion-gemma-9b-it",
        "26b": "google/diffusion-gemma-27b-it", # Diffusion Gemma usually comes in 2b/27b
        "27b": "google/diffusion-gemma-27b-it"
    }
    return mapping.get(model_input.lower(), model_input)

def main():
    console = Console()
    console.print(Panel(Text(PRASA_BANNER, style="bold cyan"), subtitle="Shandelier/PRASA v1.1"))

    parser = argparse.ArgumentParser(description="PRASA: Diffusion Gemma Text Infilling CLI")
    parser.add_argument("--template", "-t", type=str, default="template.txt", help="Path to template.txt")
    parser.add_argument("--model", "-m", type=str, default="2b", help="Model shorthand (2b, 9b, 27b) or Hugging Face ID")
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    model_id = resolve_model_id(args.model)
    
    # Fallback logic if the -it version isn't found or as per prompt requirement
    # We use 2b-it as primary default but check for basic 2b if explicitly mentioned
    if model_id == "2b":
        model_id = "google/diffusion-gemma-2b-it"

    if not os.path.exists(args.template):
        console.print(f"[bold red]Error:[/bold red] Template file '{args.template}' not found.")
        return

    with open(args.template, 'r', encoding='utf-8') as f:
        content = f.read()

    if "[INFILL]" not in content:
        console.print("[bold red]Error:[/bold red] [INFILL] token missing in template.")
        return

    prefix, suffix = content.split("[INFILL]")
    
    hardware = detect_hardware()
    console.print(f"[*] Detected hardware: [bold green]{hardware.upper()}[/bold green]")
    console.print(f"[*] Loading model: [bold yellow]{model_id}[/bold yellow]...")

    infilled_content = ""

    # Dynamic quantization logic
    # Larger models (9b+) will always default to 4-bit unless specifically disabled
    # 2b models will also use it by default to be safe on VRAM
    use_4bit = not args.no_quantize

    if hardware == "mlx":
        try:
            import mlx_lm
            # For MLX, we assume the user wants the 4bit quantized version if they pick a large model
            # or we pass the loading args. MLX-LM handle quantization during download/load.
            model, tokenizer = mlx_lm.load(model_id)
            
            prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
            
            with console.status(f"[bold blue]Generating via MLX ({model_id})...") as status:
                infilled_content = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256)
        except Exception as e:
            console.print(f"[bold red]MLX Execution Error:[/bold red] {e}")
            return
            
    else:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            load_kwargs = {"device_map": "auto"}
            if hardware == "cuda" and use_4bit:
                # Use BitsAndBytes 4-bit
                load_kwargs["load_in_4bit"] = True
                console.print("[*] 4-bit quantization enabled.")
            
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            
            prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with console.status(f"[bold blue]Generating via Transformers ({model_id})...") as status:
                outputs = model.generate(**inputs, max_new_tokens=256)
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Diffusion Gemma tokens help define the fill
                infilled_content = full_text.replace(prefix, "").replace(suffix, "").strip()
        except Exception as e:
            console.print(f"[bold red]Transformers Execution Error:[/bold red] {e}")
            return

    console.print("\n" + Panel(infilled_content, title="[bold green]Generated Infill[/bold green]", border_style="green"))
    
    console.print("\n" + Panel(prefix + "[bold reverse]" + infilled_content + "[/bold reverse]" + suffix, 
                               title="[bold white]Stitched Output[/bold white]", 
                               border_style="white"))

if __name__ == "__main__":
    main()
