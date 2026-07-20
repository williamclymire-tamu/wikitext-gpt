"""model.py -- Baseline GPT from scratch in PyTorch.

Intentionally simple. Each "IMPROVEMENT" comment marks something you can add
later and measure the perplexity delta:

  1. Weight tying (lm_head.weight = tok_emb.weight)
  2. Scaled residual init (std / sqrt(2 * n_layer) for output projections)
  3. Top-k sampling in generate()
  4. Nucleus (top-p) sampling in generate()
  5. Increase model size (more layers, wider, more heads)
  6. Longer context length
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384     # must match tokenizer
    context_len: int = 256      # max sequence length
    n_layer: int = 4            # transformer blocks
    n_head: int = 4             # attention heads
    d_model: int = 256          # embedding dimension
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written from scratch.

    Q, K, V are computed via a single linear projection then split.
    A lower-triangular mask enforces causality (can't attend to future tokens).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_len, config.context_len))
                 .view(1, 1, config.context_len, config.context_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise FFN: d_model -> 4*d_model -> d_model with GELU."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> attention -> residual, LN -> FFN -> residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        # IMPROVEMENT: weight tying -- set lm_head.weight = tok_emb.weight
        # to share parameters between input embedding and output projection.
        # reduces param count and acts as regularisation.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # IMPROVEMENT: scaled init -- after apply(_init_weights), scale
        # out_proj and fc2 weights by 1/sqrt(2*n_layer) for stable deep
        # residual training (per GPT-2 paper).
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token indices
        targets: (B, T) target token indices, optional

        Returns (logits, loss). loss is None if targets not provided.
        """
        B, T = idx.shape
        assert T <= self.config.context_len

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive generation with temperature sampling.

        IMPROVEMENT: add top_k parameter -- only sample from the k most
        likely tokens. prevents low-probability garbage.

        IMPROVEMENT: add top_p (nucleus) parameter -- sample from the
        smallest set of tokens whose cumulative probability >= p.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
