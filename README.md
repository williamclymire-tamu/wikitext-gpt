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

**Baseline config:** 4 layers, 4 heads, 256-dim embeddings, **7.42M parameters**.

(11.61M *weights* are serialized by `export_weights.py` — the tied embedding is
written twice, once as `tok_emb` and once as `lm_head`, so the engine does not
need to know about tying. Parameter count and serialized weight count differ for
exactly that reason.)

## Dataset

[WikiText-103](https://huggingface.co/datasets/wikitext) (raw, v1) — ~100M tokens of cleaned Wikipedia articles. Standard language modeling benchmark with published perplexity baselines across model sizes.

Tokenized with a ByteLevelBPE tokenizer (vocab size 16,384) trained on the WikiText-103 training split.

## Results

| Config | Val PPL | Test PPL | Notes |
|---|---|---|---|
| 4L/4H/256d, 7.42M params | **46.10** | **46.27** | 3 epochs, AdamW 3e-4, grad clip 1.0, weight tying, fp16 AMP |

Tesla T4, batch 32, context 256. ~131k tokens/sec training throughput,
~935 s/epoch over 122.4M training tokens (14,948 steps/epoch).

Val loss by epoch: 4.047 → 3.895 → 3.831. Still descending at epoch 3, but a
7.4M-parameter model has seen ~367M tokens by then — well past the ~230M that is
roughly compute-optimal at this size, so further epochs buy little.

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
prepare_data.py           WikiText-103 download, BPE tokenizer training, tokenization
model.py                  GPT architecture (config, attention, FFN, transformer block)
train.py                  Training loop with validation perplexity tracking
generate.py               Autoregressive text generation from a saved checkpoint
evaluate.py               Perplexity on a held-out split
export_weights.py         Checkpoint -> flat fp32 binaries + byte-level token table
                          + per-stage activation fixtures, for the C++/CUDA engine
tools/make_test_export.py NumPy-only generator producing the same on-disk format,
                          so the engine can be brought up without a checkpoint
```

## Note on perplexity

Reported perplexity is **token-level over a custom 16,384-entry ByteLevelBPE
vocabulary**. Published WikiText-103 numbers are word-level and are not directly
comparable — a smaller vocabulary mechanically lowers token-level perplexity.

## GELU

`FeedForward` uses `F.gelu(..., approximate="tanh")`, which is what GPT-2 shipped
and what the CUDA elementwise kernel implements. Exact (erf) GELU differs by
~1e-3, which is enough to fail engine parity for no modeling benefit.

## Roadmap

Tracking perplexity deltas against the baseline for each change:

- [ ] Gradient clipping
- [ ] Cosine LR schedule with linear warmup
- [ ] Separate weight decay groups (decay only 2D params)
- [ ] Weight tying (embedding ↔ output projection)
- [ ] Scaled residual initialization
- [ ] Top-k and nucleus sampling
- [ ] Mixed precision training (`torch.amp`)
- [x] Weight export to flat fp32 binaries (`export_weights.py`)
- [x] KV cache for generation — implemented in the C++/CUDA engine, not in `generate.py`
- [ ] Gradient accumulation
- [ ] Model scaling experiments (8L/8H/512d, 12L/12H/768d)
- [ ] `torch.profiler` integration
- [ ] CUDA kernel for fused attention
- [ ] DDP multi-GPU training
- [ ] Deployment (FastAPI + SageMaker/ECS)

## AI disclosure

AI tools (Claude) were used for SOME code review, refactoring, documentation, and debugging during development. All architecture, training, and evaluation code was written and is fully understood and explainable by me.
