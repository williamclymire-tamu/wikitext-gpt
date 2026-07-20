from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer

from cslm_model import CSLMConfig, CSLMModel


class CorpusDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_tokenizer(tokenizer_dir: Path) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.txt"),
    )


def tokenize_corpus(corpus_path: Path, tokenizer: ByteLevelBPETokenizer) -> list[int]:
    print(f"Tokenizing {corpus_path} ...")
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text).ids
    print(f"  {len(text):,} chars -> {len(ids):,} tokens")
    return ids


def save_checkpoint(
    path: Path,
    model: CSLMModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config.to_dict(),
            "step": step,
            "loss": loss,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    tokenizer_dir = Path(args.tokenizer_dir)
    corpus_path = Path(args.corpus)
    checkpoint_dir = Path(args.checkpoint_dir)

    tokenizer = load_tokenizer(tokenizer_dir)
    token_ids = tokenize_corpus(corpus_path, tokenizer)

    config = CSLMConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    dataset = CorpusDataset(token_ids, config.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} samples  |  {len(loader):,} batches/epoch")

    device = torch.device("cpu")
    model = CSLMModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch:3d}  step {global_step:6d}  "
                    f"loss {loss.item():.4f}  "
                    f"elapsed {elapsed:.0f}s"
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                ckpt = checkpoint_dir / f"cslm_step{global_step:07d}.pt"
                save_checkpoint(ckpt, model, optimizer, global_step, loss.item())
                print(f"  -> max_steps reached, checkpoint saved: {ckpt}")
                return

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt = checkpoint_dir / f"cslm_step{global_step:07d}.pt"
                save_checkpoint(ckpt, model, optimizer, global_step, loss.item())
                print(f"  -> checkpoint saved: {ckpt}")

        avg_loss = epoch_loss / len(loader)
        ckpt = checkpoint_dir / f"cslm_epoch{epoch:03d}.pt"
        save_checkpoint(ckpt, model, optimizer, global_step, avg_loss)
        print(f"Epoch {epoch} done  avg_loss={avg_loss:.4f}  -> {ckpt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CSLM language model.")
    p.add_argument("--corpus", default="cs_corpus.txt")
    p.add_argument("--tokenizer-dir", default="cslm/tokenizer")
    p.add_argument("--checkpoint-dir", default="cslm/checkpoint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
        help="Stop after this many gradient steps (0 = no limit)",
    )
    p.add_argument(
        "--save-every", type=int, default=500,
        help="Save a checkpoint every N steps (0 = epoch-only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

