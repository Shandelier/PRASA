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

def main():
    console = Console()
    console.print(Panel(Text(PRASA_BANNER, style="bold cyan"), subtitle="Shandelier/PRASA v1.0"))

    parser = argparse.ArgumentParser(description="PRASA: Diffusion Gemma Text Infilling CLI")
    parser.add_argument("--template", type=str, default="template.txt", help="Path to template.txt")
    parser.add_argument("--model", type=str, default="google/diffusion-gemma-2b", help="Model name (Google Diffusion Gemma)")
    parser.add_argument("--quantize", action="store_true", default=True, help="Enable 4-bit/8-bit quantization for CUDA")
    args = parser.parse_args()

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
    console.print(f"[*] Preparing model: [bold yellow]{args.model}[/bold yellow]...")

    infilled_content = ""

    if hardware == "mlx":
        try:
            import mlx_lm
            model, tokenizer = mlx_lm.load(args.model)
            # Diffusion Gemma/Gemma Infilling format
            prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
            
            with console.status("[bold blue]Generating infill via MLX...") as status:
                infilled_content = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256)
        except Exception as e:
            console.print(f"[bold red]MLX Execution Error:[/bold red] {e}")
            return
            
    else:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            load_kwargs = {"device_map": "auto"}
            if hardware == "cuda" and args.quantize:
                load_kwargs["load_in_4bit"] = True
            
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
            
            prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with console.status("[bold blue]Generating infill via Transformers...") as status:
                outputs = model.generate(**inputs, max_new_tokens=256)
                # Decode and strip prompt
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Crude extraction of the filled part
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
