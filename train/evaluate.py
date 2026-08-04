"""Evaluate a checkpoint's perplexity on a held-out split.

Published WikiText-103 numbers are reported on test, so use --split test
for anything compared against the literature.

    python evaluate.py
    python evaluate.py --split val
    python evaluate.py --checkpoint checkpoints/last.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from model import GPT
from train import TokenDataset, evaluate as eval_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokens = torch.load(
        os.path.join(args.data_dir, f"{args.split}.pt"), weights_only=True
    )
    loader = DataLoader(
        TokenDataset(tokens, config.context_len), batch_size=args.batch_size
    )

    loss, ppl = eval_split(model, loader, device)

    print(f"checkpoint : {args.checkpoint} (epoch {ckpt['epoch']})")
    print(f"config     : {config.n_layer}L/{config.n_head}H/{config.d_model}d, "
          f"{model.param_count():,} params")
    print(f"split      : {args.split} ({len(tokens):,} tokens, {len(loader)} batches)")
    print(f"loss       : {loss:.4f}")
    print(f"perplexity : {ppl:.2f}")
    
if __name__ == "__main__":
    main()