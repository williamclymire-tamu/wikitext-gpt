# wikitext-gpt

From-scratch GPT-2 style transformer trained on WikiText-103, implemented entirely in PyTorch with no high-level wrappers (no `nn.MultiheadAttention`, no HuggingFace model classes).

## Architecture

| Component | Detail |
|---|---|
| Attention | Hand-written multi-head causal self-attention (fused QKV projection) |
| Normalization | Pre-norm (LayerNorm before each sublayer) |
| Activation | GELU |
| FFN expansion | 4x |
| Positional encoding | Learned embeddings |

**Baseline config:** 4 layers, 4 heads, 256-dim embeddings, ~11.6M parameters.

## Dataset

[WikiText-103](https://huggingface.co/datasets/wikitext) (raw, v1) — ~100M tokens of cleaned Wikipedia articles. Standard language modeling benchmark with published perplexity baselines across model sizes.

Tokenized with a ByteLevelBPE tokenizer (vocab size 16,384) trained on the WikiText-103 training split.

## Results

| Config | Val Perplexity | Notes |
|---|---|---|
| Baseline (4L/4H/256d) | — | Fixed LR, no clipping, no weight tying |

*Table updated as improvements are benchmarked.*

## Usage

```bash
pip install torch datasets tokenizers

# 1. download + tokenize
python prepare_data.py

# 2. train
python train.py
python train.py --n-layer 8 --d-model 512 --epochs 20  # or override defaults

# 3. generate
python generate.py "The history of"
python generate.py "In 1945" --temperature 0.7
```

## Project structure

```
prepare_data.py   WikiText-103 download, BPE tokenizer training, tokenization
model.py          GPT architecture (config, attention, FFN, transformer block)
train.py          Training loop with validation perplexity tracking
generate.py       Autoregressive text generation from a saved checkpoint
```

## Roadmap

Tracking perplexity deltas against the baseline for each change:

- [ ] Gradient clipping
- [ ] Cosine LR schedule with linear warmup
- [ ] Separate weight decay groups (decay only 2D params)
- [ ] Weight tying (embedding ↔ output projection)
- [ ] Scaled residual initialization
- [ ] Top-k and nucleus sampling
- [ ] Mixed precision training (`torch.amp`)
- [ ] KV cache for generation
- [ ] Gradient accumulation
- [ ] Model scaling experiments (8L/8H/512d, 12L/12H/768d)
- [ ] `torch.profiler` integration
- [ ] CUDA kernel for fused attention
- [ ] DDP multi-GPU training
- [ ] Deployment (FastAPI + SageMaker/ECS)

## AI disclosure

AI tools (Claude) were used for SOME code review, refactoring, documentation, and debugging during development. All architecture, training, and evaluation code was written and is fully understood and explainable by me.
