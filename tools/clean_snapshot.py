import torch
import re
import argparse

def clean_snapshot(input_path, output_path):
    """Clean a PyTorch checkpoint, keeping only relevant model parameters."""
    ckpt = torch.load(input_path, map_location="cpu")

    # If checkpoint has a top-level "model" key
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # Assume it's a raw state_dict
        state_dict = ckpt

    # Compile unwanted patterns
    patterns = [
        re.compile(r"^bases"),
        re.compile(r"^layer_norm.*"),
        re.compile(r"^decoder_ema.*"),
        re.compile(r"^mlp_head_err.*"),
    ]

    def should_drop(key):
        return any(p.match(key) for p in patterns)

    # Filter keys
    cleaned_state = {k: v for k, v in state_dict.items() if not should_drop(k)}

    # Save back only the cleaned model dictionary
    torch.save(cleaned_state, output_path)
    print(f"✅ Cleaned checkpoint saved to: {output_path}")
    print(f"🗝️  Kept {len(cleaned_state)} keys (removed {len(state_dict) - len(cleaned_state)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a model snapshot for PyTorch.")
    parser.add_argument("input", help="Path to the input .pth or .pt file")
    parser.add_argument("output", help="Path to save the cleaned checkpoint")
    args = parser.parse_args()

    clean_snapshot(args.input, args.output)
