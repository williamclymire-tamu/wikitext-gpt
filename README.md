# wikitext-gpt

From-scratch GPT-2 style transformer trained on WikiText-103, with a standalone
C++/CUDA inference engine. Implemented entirely in PyTorch with no high-level
wrappers (no `nn.MultiheadAttention`, no HuggingFace model classes).

The engine uses a fused FlashAttention-style kernel, KV cache, and a three-tier
correctness strategy (PyTorch -> scalar C++ reference -> CUDA). See
[engine/README.md](engine/README.md) for kernel benchmarks and architecture details.

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
need to know about tying.)

## Dataset

[WikiText-103](https://huggingface.co/datasets/wikitext) (raw, v1) — ~100M tokens
of cleaned Wikipedia articles. Tokenized with a ByteLevelBPE tokenizer (vocab size
16,384) trained on the training split.

## Results

| Config | Val PPL | Test PPL | Notes |
|---|---|---|---|
| 4L/4H/256d, 7.42M params | **46.10** | **46.27** | 3 epochs, AdamW 3e-4, grad clip 1.0, weight tying, fp16 AMP |

Tesla T4, batch 32, context 256. ~131k tokens/sec training throughput,
~935 s/epoch over 122.4M training tokens (14,948 steps/epoch).

## Usage

```bash
pip install torch datasets tokenizers

# 1. download + tokenize
python train/prepare_data.py

# 2. train
python train/train.py
python train/train.py --n-layer 8 --d-model 512 --epochs 20  # or override defaults

# 3. generate
python train/generate.py "The history of"

# 4. export for the C++/CUDA engine
python train/export_weights.py --checkpoint checkpoints/best.pt --tokenizer-dir data/tokenizer --out export

# 5. run the engine (see engine/README.md for full instructions)
cd engine && make engine ARCH=sm_75 && ./engine generate ../export --ids 464,1092 --tokens 100
```

## Project Structure

```
train/
  model.py                  GPT architecture (config, attention, FFN, transformer block)
  train.py                  Training loop with validation perplexity tracking
  prepare_data.py           WikiText-103 download, BPE tokenizer training, tokenization
  generate.py               Autoregressive text generation from a saved checkpoint
  evaluate.py               Perplexity on a held-out split
  export_weights.py         Checkpoint -> flat fp32 binaries + byte-level token table
  make_test_export.py       NumPy-only random export for engine bring-up without a checkpoint
  requirements.txt          Python dependencies

engine/                     C++/CUDA inference engine (see engine/README.md)
  run.cu                    Entry point: parity, generate, bench
  Makefile                  Builds engine, test_kernels, test_cpu
  kernels/                  CUDA kernels (fused attention, elementwise, linear)
  runtime/                  Host code (transformer, weights, reference, detokenizer)
  include/                  Shared headers

tests/
  test_kernels.cu           Unified GPU kernel tests + benchmarks
  test_cpu.cpp              GPU-free harness for CPU reference + KV cache
  gen_fixtures.py           PyTorch reference generator for all kernel tests

notebooks/
  colab_train.ipynb         WikiText-103 training with Drive-backed checkpoints
  colab_engine.ipynb        Engine build, parity, bench, generate on a free T4
```

## Note on Perplexity

Reported perplexity is **token-level over a custom 16,384-entry ByteLevelBPE
vocabulary**. Published WikiText-103 numbers are word-level and are not directly
comparable — a smaller vocabulary mechanically lowers token-level perplexity.

## GELU

`FeedForward` uses `F.gelu(..., approximate="tanh")`, which is what GPT-2 shipped
and what the CUDA elementwise kernel implements. Exact (erf) GELU differs by
~1e-3, which is enough to fail engine parity for no modeling benefit.

## Roadmap

- [x] Gradient clipping
- [x] Weight tying (embedding <-> output projection)
- [x] Mixed precision training (`torch.amp`)
- [x] Weight export to flat fp32 binaries (`export_weights.py`)
- [x] CUDA kernel for fused attention
- [x] KV cache for generation (C++/CUDA engine)
- [ ] Cosine LR schedule with linear warmup
- [ ] Separate weight decay groups (decay only 2D params)
- [ ] Scaled residual initialization
- [ ] Top-k and nucleus sampling
- [ ] Gradient accumulation
- [ ] Model scaling experiments (8L/8H/512d, 12L/12H/768d)
- [ ] `torch.profiler` integration
- [ ] DDP multi-GPU training
- [ ] Deployment (FastAPI + SageMaker/ECS)

## AI disclosure

AI tools (Claude) were used for some code review, refactoring, documentation, and
debugging during development. All architecture, training, and evaluation code was
written and is fully understood and explainable by me.
