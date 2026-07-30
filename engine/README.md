# Fused Attention CUDA Kernel

A from-scratch fused attention kernel with shared-memory tiling and online streaming softmax (FlashAttention-style), eliminating the N x N score-matrix materialization that naive attention requires.

## Results

**GPU:** Tesla T4 (SM 7.5, 40 SMs) | **Config:** B=1 H=8 d=64 causal | **Timing:** cudaEvent, 10 warmup + 100 timed

| N    | Naive (ms) | Fused (ms) | Speedup |
|------|-----------|-----------|---------|
| 128  | 0.2638    | 0.1033    | 2.55x   |
| 256  | 0.9927    | 0.3382    | 2.94x   |
| 512  | 2.3262    | 0.8053    | 2.89x   |
| 1024 | 8.8222    | 3.1242    | 2.82x   |
| 2048 | 36.7370   | 12.6644   | 2.90x   |

Absolute times vary run to run by 10-30% on a shared Colab T4; the speedup ratio
is stable to about ±0.1x across runs. One earlier run showed 3.32x at N=256,
which did not reproduce — treat the 2.5-2.9x band as the claim.

### Decode (single-query, KV cache)

| seq_len | Decode (ms) |
|---------|------------|
| 128     | 0.1197     |
| 256     | 0.2378     |
| 512     | 0.4739     |
| 1024    | 1.3724     |
| 2048    | 3.2603     |

Correctness verified against PyTorch reference (fp32, allclose atol=1e-5 rtol=1e-4), including non-power-of-two sequence lengths (N=37, 127, 200). Decode, LayerNorm, GELU, and residual add kernels all pass.

## Inference engine

The kernels are wired into a standalone C++/CUDA engine that serves a trained
checkpoint with no Python or PyTorch dependency at runtime.

```
export_weights.py  ->  export/*.bin        flat fp32, no pickle, no schema
weights.cu         ->  device upload
transformer.cu     ->  prefill / decode, KV cache, sampling
run.cu             ->  parity | generate | bench
```

**Prefill** runs the tiled fused kernel over the whole prompt with a causal mask.
**Decode** runs the single-query kernel against a persistent per-layer KV cache
(`[n_head, max_seq, head_dim]`, contiguous). Dense GEMMs go to cuBLAS; attention,
LayerNorm, GELU, residual add, and the head-layout transposes are all custom.

### Engine results

Serving the trained WikiText-103 checkpoint (4L/4H/256d, 7.42M params, 46.1 val PPL)
on a Tesla T4:

| Metric | Value |
|---|---|
| Decode throughput | **688 tok/s** (1.453 ms/token, batch 1, context 1→255) |
| Weights loaded | 11.61M fp32 (46.5 MB) |
| Prefill parity vs PyTorch | max_abs **9.0e-5** on logits, **0/32** argmax mismatches |
| Scalar CPU reference vs PyTorch | max_abs **1.3e-5** on logits |
| Decode vs prefill | max_abs **1.4e-5**, 0/32 rows differing |

Parity drift grows smoothly with depth (1.3e-5 → 2.6e-5 → 4.2e-5 → 9.0e-5 across
the four blocks), which is ordinary fp32 accumulation under `--use_fast_math`, not
a bug — a real kernel error shows up as a step change at one layer. The large
*relative* errors on `final_ln` and `logits` (2.1e-1, 3.7e-1) are on
near-zero-valued elements, which is why the check is `atol`-dominated; argmax
agreement is the metric that actually matters for sampling.

### Correctness strategy

There are three implementations of the same forward pass, and each one checks
the next:

| Implementation | Where | Role |
|---|---|---|
| PyTorch (or NumPy) | `export_weights.py` / `tools/make_test_export.py` | writes per-stage fixtures |
| Scalar C++ (`reference.cpp`) | CPU, double accumulation | the oracle; builds with plain g++ |
| CUDA (`transformer.cu`) | GPU | the thing being tested |

Fixtures are dumped **per stage** (embedding, each block, final norm, logits), so
a parity failure names the layer instead of just reporting that the logits are
wrong. `test_cpu` needs no GPU and no toolkit, so the wiring can be validated
before any CUDA is involved — and both the CPU and CUDA paths are additionally
checked for the property that incremental decode through the KV cache reproduces
a full prefill exactly.

```bash
# 1. export (from the wikitext-gpt repo)
python export_weights.py --random          # or omit --random once trained

# 2. validate the wiring on any machine, no GPU required
make test_cpu && ./test_cpu ../wikitext-gpt/export

# 3. on a GPU box
make engine HEAD_DIM=64 ARCH=sm_75
./engine parity   ../wikitext-gpt/export
./engine generate ../wikitext-gpt/export --ids 464,1092 --tokens 100 --temp 0.8 --top-k 40
./engine bench    ../wikitext-gpt/export --tokens 256
```

Tokenizer note: `export_weights.py` inverts GPT-2's byte↔unicode mapping and
ships a flat raw-byte table, so the C++ side decodes by blob lookup and `fwrite`
with no unicode handling at all. Encoding stays in Python — the engine takes
token ids.

## Architecture

The fused kernel processes attention in tiles without ever materializing the full N x N score matrix in global memory:

1. **Load Q tile** (TILE_Q x d) into shared memory once per block
2. **Stream K/V tiles** (TILE_KV x d) through shared memory
3. **QK^T** computed with warp-level shuffle reduction (32 threads across head dim)
4. **Online softmax** (Milakov & Gimelshein 2018): maintains running row-max and row-sum, applies correction `exp(old_max - new_max)` to rescale accumulated output when new tiles introduce larger values
5. **Causal masking** applied inside the tile loop, with early exit when the KV tile is entirely past the diagonal

Thread layout: `blockDim=(32, TILE_Q)` — one warp spans the head dimension, one thread per query row in the tile.

## Quick Start

### Colab (free, recommended)

| Notebook | Purpose |
|---|---|
| `colab_engine.ipynb` | Clone, export random weights, CPU validation, build, parity, bench, generate (~10 min, no training) |
| `colab_train.ipynb` | WikiText-103 training with Drive-backed checkpoints and resume (long-running) |

Both clone the repos rather than pasting sources into `%%writefile` cells.

> `colab.ipynb` is the **original kernel-only notebook** and is superseded. It
> inlines a copy of every source file, which was tractable at 6 files and is not
> at 24 — it is already missing `linear.cu` and the whole engine. Kept for
> reference; use `colab_engine.ipynb`.

### Local / AWS

```bash
# Generate references
python generate_reference.py
python generate_day1_ref.py

# Build
make ARCH=sm_75  # T4

# Test
./attention test test_data
./test_day1 test test_day1_data

# Benchmark
./attention bench
./test_day1 bench
```

## Files

| File | Purpose |
|------|---------|
| `attention.cuh` | Shared header: CUDA_CHECK, tile defines, kernel declarations, allclose check |
| `naive_attention.cu` | Three-phase baseline: Q@K^T, softmax, S@V (materializes N x N) |
| `fused_attention.cu` | Fused tiled prefill kernel with online streaming softmax |
| `fused_attention_decode.cu` | Single-query decode kernel with KV cache + cache append |
| `elementwise.cuh / .cu` | LayerNorm (warp shuffle), GELU (tanh approx), residual add |
| `main.cu` | Prefill test driver + benchmark |
| `test_day1.cu` | Decode + elementwise test driver + decode benchmark |
| `generate_reference.py` | PyTorch reference for prefill attention |
| `generate_day1_ref.py` | PyTorch reference for decode, layernorm, gelu, residual |
| `colab.ipynb` | Self-contained Colab notebook |
| **Engine** | |
| `model_types.h` | CUDA-free struct definitions shared by the CPU and GPU paths |
| `weights_host.h / .cpp` | config.json parse + `.bin` loading (no CUDA) |
| `weights.cu` | Host -> device weight upload |
| `transformer.cuh / .cu` | RunState, KV cache, prefill/decode, head-layout kernels, sampling |
| `reference.h / .cpp` | Scalar CPU forward pass — the oracle |
| `detokenizer.h / .cpp` | Raw-byte token table lookup |
| `run.cu` | Driver: `parity`, `generate`, `bench` |
| `test_cpu.cpp` | GPU-free harness for the reference and the KV cache |
| `checks.h` | Shared allclose / argmax verdicts |

## Limitations

- fp32 only (no fp16/bf16 tensor core path)
- Forward pass only (no backward)
- Fixed head dim (HEAD_DIM=64, compile-time; the engine checks this against
  `config.json` and tells you to rebuild rather than producing garbage)
- No dropout, no GQA/MQA
- KV cache is contiguous and single-sequence, not paged; batch size 1
- Prefill assumes it starts at position 0 (no prefix reuse / chunked prefill)
- Sampling runs on the host — a 64 KB copy per token, deliberately not optimized
  until it shows up in a profile
- Single-kernel design (no split-K for very long sequences)

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)

## AI disclosure

AI tools (Claude) were used for some code review, refactoring, documentation, and debugging during development. All architecture, training, and evaluation code was written and is fully understood and explainable by me.
