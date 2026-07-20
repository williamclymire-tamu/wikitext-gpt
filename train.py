"""train.py -- Baseline training loop.

Intentionally simple so you can add improvements and measure the delta:

  1. IMPROVEMENT: cosine LR schedule with warmup (replace fixed LR)
  2. IMPROVEMENT: gradient clipping (torch.nn.utils.clip_grad_norm_)
  3. IMPROVEMENT: separate weight decay groups (decay 2D params only)
  4. IMPROVEMENT: mixed precision training (torch.amp)
  5. IMPROVEMENT: gradient accumulation (larger effective batch size)
  6. IMPROVEMENT: learning rate finder (sweep LRs, pick the knee)

Usage:
    python prepare_data.py   # run first to create data/*.pt
    python train.py
    python train.py --epochs 20 --lr 1e-3
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader

from model import GPT, GPTConfig


class TokenDataset(Dataset):
    """Serve (context_len + 1)-token windows from a flat token tensor.
    Non-overlapping stride so each token is seen once per epoch."""

    def __init__(self, data: torch.Tensor, context_len: int):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.context_len)

    def __getitem__(self, idx):
        start = idx * self.context_len
        end = start + self.context_len + 1
        chunk = self.data[start:end]
        return chunk[:-1], chunk[1:]


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute average loss and perplexity on a data loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # load pre-tokenized data
    train_tokens = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=True)
    val_tokens = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=True)
    print(f"train: {len(train_tokens):,} tokens  val: {len(val_tokens):,} tokens")

    train_ds = TokenDataset(train_tokens, args.context_len)
    val_ds = TokenDataset(val_tokens, args.context_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # model
    config = GPTConfig(
        vocab_size=args.vocab_size,
        context_len=args.context_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"params: {model.param_count():,}")

    # IMPROVEMENT: separate weight decay groups -- apply weight decay only to
    # 2D parameters (linear weights), not to biases or LayerNorm.
    # IMPROVEMENT: cosine LR schedule with linear warmup instead of fixed LR.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            loss.backward()

            # IMPROVEMENT: gradient clipping here --
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_ppl = math.exp(train_loss)
        dt = time.time() - t0

        val_loss, val_ppl = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:>3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} ppl {train_ppl:.1f} | "
            f"val loss {val_loss:.4f} ppl {val_ppl:.1f} | "
            f"{dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best (val ppl {val_ppl:.1f})")

    # save final
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }, os.path.join(args.out_dir, "last.pt"))

    print(f"\ndone. best val ppl: {math.exp(best_val_loss):.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")

    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--context-len", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)

    train(p.parse_args())
