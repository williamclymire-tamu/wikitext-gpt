from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CSLMConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "CSLMConfig":
        raw = checkpoint.get("config", {})
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(**raw)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CSLMConfig) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        return self.dropout(self.proj(attn_out))


class FeedForward(nn.Module):
    def __init__(self, config: CSLMConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: CSLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CSLMModel(nn.Module):
    def __init__(self, config: CSLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
            if targets is not None:
                targets = targets[:, -self.config.block_size :]
            seq_len = input_ids.size(1)

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1), targets.contiguous().view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.block_size :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < values[:, [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {path} must be a dict")
    if "model_state_dict" not in checkpoint and "state_dict" in checkpoint:
        checkpoint["model_state_dict"] = checkpoint["state_dict"]
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"checkpoint at {path} must contain model_state_dict or state_dict"
        )
    return checkpoint


def find_latest_checkpoint(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(
        [candidate for candidate in path.glob("*.pt") if candidate.is_file()]
        + [candidate for candidate in path.glob("*.pth") if candidate.is_file()],
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None