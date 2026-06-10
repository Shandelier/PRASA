import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="PRASA: Diffusion Gemma Text Infilling CLI")
    parser.add_argument("--template", type=str, required=True, help="Path to template.txt")
    parser.add_argument("--model", type=str, default="google/diffusion-gemma-27b", help="Model name")
    parser.add_argument("--four_bit", action="store_true", help="Enable 4-bit quantization")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        print(f"Error: Template file {args.template} not found.")
        return

    with open(args.template, 'r') as f:
        content = f.read()

    if "[INFILL]" not in content:
        print("Error: [INFILL] token missing in template.")
        return

    prefix, suffix = content.split("[INFILL]")

    print(f"Loading model: {args.model}...")
    # Hook for MLX/Transformers
    # In a real scenario, we'd use mlx_lm for M4 Pro
    try:
        import mlx_lm
        model, tokenizer = mlx_lm.load(args.model)
        prompt = f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>"
        output = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=512)
        print("\n--- Infilled Text ---")
        print(prefix + output + suffix)
    except ImportError:
        print("MLX not found, falling back to Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, load_in_4bit=args.four_bit, device_map="auto")
        inputs = tokenizer(f"<|prefix|>{prefix}<|suffix|>{suffix}<|fill|>", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(**inputs, max_new_tokens=512)
        print("\n--- Infilled Text ---")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
