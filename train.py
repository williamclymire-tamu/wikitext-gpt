"""Train GPT on pre-tokenized WikiText-103 splits.

    python prepare_data.py   # creates data/*.pt
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
    """Non-overlapping (context_len+1)-token windows from a flat tensor."""

    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.context_len)

    def __getitem__(self, idx):
        start = idx * self.context_len
        chunk = self.data[start : start + self.context_len + 1]
        return chunk[:-1], chunk[1:]


# perplexity = exp(cross-entropy loss)
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
        n += 1
    model.train()
    avg = total / max(n, 1)
    return avg, math.exp(avg)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_tokens = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=True)
    val_tokens = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=True)
    print(f"train: {len(train_tokens):,} tokens  val: {len(val_tokens):,} tokens")

    train_loader = DataLoader(
        TokenDataset(train_tokens, args.context_len),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TokenDataset(val_tokens, args.context_len),
        batch_size=args.batch_size,
    )

    config = GPTConfig(
        vocab_size=args.vocab_size, context_len=args.context_len,
        n_layer=args.n_layer, n_head=args.n_head,
        d_model=args.d_model, dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"params: {model.param_count():,}")

    # TODO: separate weight decay groups (only decay 2D params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for x, y in train_loader:
            _, loss = model(x.to(device), y.to(device))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()

        train_loss = running / len(train_loader)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"epoch {epoch:>3d}/{args.epochs} | "
            f"train {train_loss:.4f} ({math.exp(train_loss):.1f}) | "
            f"val {val_loss:.4f} ({val_ppl:.1f}) | "
            f"{dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, os.path.join(args.out_dir, "best.pt"))

    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }, os.path.join(args.out_dir, "last.pt"))

    print(f"\nbest val ppl: {math.exp(best_val):.1f}")


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
    p.add_argument("--grad-clip", type=float, default=1.0)
    train(p.parse_args())
