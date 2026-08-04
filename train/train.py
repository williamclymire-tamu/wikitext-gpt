"""Train GPT on pre-tokenized WikiText-103 splits.

    python prepare_data.py                    # creates data/*.pt
    python train.py
    python train.py --epochs 3 --lr 3e-4

Colab / preemptible use:

    python train.py --out-dir /content/drive/MyDrive/wikitext-gpt --save-every 500
    python train.py --out-dir /content/drive/MyDrive/wikitext-gpt --resume

--resume restores model, optimizer, and AMP scaler state, and replays the data
order exactly: the shuffle generator is seeded per epoch, so skipping forward
`batches_done` batches lands on the same batch the run died on. Losing a session
costs the skip-forward time, not an epoch.

Note on epoch count: at 11.6M parameters, roughly compute-optimal is ~230M
training tokens, which is 2-3 passes over WikiText-103. The default is 3. Ten
epochs mostly buys overfitting.
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
# Deliberately fp32, even when training under AMP: this number goes on a resume,
# so it should not depend on autocast rounding.
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


def save_ckpt(path, **payload):
    """Write via a temp file and rename.

    torch.save straight to the destination means a disconnect mid-write leaves a
    truncated checkpoint, which is worse than no checkpoint -- especially on a
    Drive mount, where the file that gets corrupted is the one you were relying
    on to recover.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def make_train_loader(dataset, batch_size, epoch, seed):
    """Seeded per epoch so the shuffle order is reproducible across restarts."""
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp
    print(f"device: {device}  amp: {use_amp}")

    train_tokens = torch.load(os.path.join(args.data_dir, "train.pt"), weights_only=True)
    val_tokens = torch.load(os.path.join(args.data_dir, "val.pt"), weights_only=True)
    print(f"train: {len(train_tokens):,} tokens  val: {len(val_tokens):,} tokens")

    train_ds = TokenDataset(train_tokens, args.context_len)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    start_epoch, skip_batches, global_step = 1, 0, 0
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(ckpt_path):
            print(f"--resume: nothing at {ckpt_path}, starting fresh")
        else:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optim_state"])
            if ck.get("scaler_state") is not None:
                scaler.load_state_dict(ck["scaler_state"])
            start_epoch = ck["epoch"]
            skip_batches = ck["batches_done"]
            global_step = ck["global_step"]
            best_val = ck["best_val"]
            print(f"resumed: epoch {start_epoch}, batch {skip_batches}, "
                  f"step {global_step:,}, best val ppl {math.exp(best_val):.2f}")

    val_loss, val_ppl = float("nan"), float("nan")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_train_loader(train_ds, args.batch_size, epoch, args.seed)
        n_batches = len(train_loader)

        model.train()
        running, counted = 0.0, 0
        tokens_seen = 0
        t0 = time.time()

        if skip_batches:
            print(f"  fast-forwarding {skip_batches:,}/{n_batches:,} batches...")

        for i, (x, y) in enumerate(train_loader):
            if i < skip_batches:
                continue

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                # gradients must be unscaled before the norm is meaningful
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            counted += 1
            tokens_seen += x.numel()
            global_step += 1

            if args.save_every > 0 and (i + 1) % args.save_every == 0:
                save_ckpt(
                    ckpt_path,
                    model_state=model.state_dict(),
                    optim_state=optimizer.state_dict(),
                    scaler_state=scaler.state_dict() if use_amp else None,
                    config=config, epoch=epoch, batches_done=i + 1,
                    global_step=global_step, best_val=best_val,
                    val_loss=val_loss, val_ppl=val_ppl,
                )
                rate = tokens_seen / max(time.time() - t0, 1e-9)
                print(f"  epoch {epoch} [{i+1:>6,}/{n_batches:,}] "
                      f"loss {running/max(counted,1):.4f}  "
                      f"{rate/1e3:.1f}k tok/s  (saved)")

        skip_batches = 0

        train_loss = running / max(counted, 1)
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
            save_ckpt(
                os.path.join(args.out_dir, "best.pt"),
                model_state=model.state_dict(), config=config, epoch=epoch,
                val_loss=val_loss, val_ppl=val_ppl,
            )

        # epoch boundary: batches_done=0 so a resume starts the next epoch clean
        save_ckpt(
            ckpt_path,
            model_state=model.state_dict(),
            optim_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict() if use_amp else None,
            config=config, epoch=epoch + 1, batches_done=0,
            global_step=global_step, best_val=best_val,
            val_loss=val_loss, val_ppl=val_ppl,
        )

    save_ckpt(
        os.path.join(args.out_dir, "last.pt"),
        model_state=model.state_dict(), config=config, epoch=args.epochs,
        val_loss=val_loss, val_ppl=val_ppl,
    )

    print(f"\nbest val ppl: {math.exp(best_val):.2f}")
    print("(token-level over the 16,384-entry BPE vocab -- not comparable to "
          "published word-level WikiText-103 numbers)")


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
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save-every", type=int, default=500,
                   help="steps between mid-epoch checkpoints (0 disables)")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out-dir>/ckpt.pt")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (fp32 training)")
    train(p.parse_args())
