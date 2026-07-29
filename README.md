# Fused Attention CUDA Kernel

A from-scratch fused attention kernel with shared-memory tiling and online streaming softmax (FlashAttention-style), eliminating the N x N score-matrix materialization that naive attention requires.

## Results

**GPU:** Tesla T4 (SM 7.5, 40 SMs) | **Config:** B=1 H=8 d=64 causal | **Timing:** cudaEvent, 10 warmup + 100 timed

| N    | Naive (ms) | Fused (ms) | Speedup |
|------|-----------|-----------|---------|
| 128  | 0.1939    | 0.0759    | 2.56x   |
| 256  | 0.7145    | 0.2541    | 2.81x   |
| 512  | 2.4369    | 0.9117    | 2.67x   |
| 1024 | 9.5288    | 3.3414    | 2.85x   |
| 2048 | 40.5299   | 13.7816   | 2.94x   |

### Decode (single-query, KV cache)

| seq_len | Decode (ms) |
|---------|------------|
| 128     | 0.1197     |
| 256     | 0.2378     |
| 512     | 0.4739     |
| 1024    | 1.3724     |
| 2048    | 3.2603     |

Correctness verified against PyTorch reference (fp32, allclose atol=1e-5 rtol=1e-4), including non-power-of-two sequence lengths (N=37, 127, 200). Decode, LayerNorm, GELU, and residual add kernels all pass.

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

Open `colab.ipynb` in Google Colab with a T4 GPU runtime. It writes all source files, builds, tests, and benchmarks in ~5 minutes.

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

## Limitations

- fp32 only (no fp16/bf16 tensor core path)
- Forward pass only (no backward)
- Fixed head dim (HEAD_DIM=64, compile-time)
- No dropout, no GQA/MQA, no paged KV cache
- Single-kernel design (no split-K for very long sequences)

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)

## AI disclosure

AI tools (Claude) were used for some code review, refactoring, documentation, and debugging during development. All architecture, training, and evaluation code was written and is fully understood and explainable by me.
