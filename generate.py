"""generate.py -- Generate text from a trained checkpoint.

Usage:
    python generate.py "The meaning of"
    python generate.py "In 1945" --temperature 0.7
    python generate.py --max-tokens 500
"""

import argparse
import os

import torch
from tokenizers import ByteLevelBPETokenizer

from model import GPT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="The")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--tokenizer-dir", default="data/tokenizer")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded: epoch {ckpt['epoch']}, val ppl {ckpt['val_ppl']:.1f}")

    # load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )

    # encode prompt
    encoded = tokenizer.encode(args.prompt)
    ids = encoded.ids or [tokenizer.token_to_id("<|endoftext|>") or 0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # generate
    out = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()