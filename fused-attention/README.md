# Fused Attention CUDA Kernel

A fused multi-head causal attention kernel with shared-memory tiling and online streaming softmax, eliminating N×N score-matrix materialization.

## What this is

A from-scratch CUDA implementation of the FlashAttention algorithm (Dao et al., 2022). The kernel tiles Q, K, V into shared memory and maintains running softmax statistics (row-max `m`, sum-of-exponentials `l`, weighted accumulator `acc`) across K/V tiles, rescaling prior accumulations when a new tile introduces a larger row-max. No N×N attention matrix is ever materialized in global memory.

Verified against a PyTorch reference (explicit `Q @ K.T`, scale, causal mask, softmax, `@ V`) to fp32 relative tolerance of 1e-4.

## Benchmark

<!-- Fill in after running: ./attention bench -->

| N | Naive CUDA (ms) | Fused (ms) | Speedup |
|---:|---:|---:|---:|
| 128 | | | |
| 256 | | | |
| 512 | | | |
| 1024 | | | |
| 2048 | | | |

GPU: [fill in], CUDA [version], default clocks.
B=1, H=8, d=64, causal. 10 warmup + 100 timed iterations, `cudaEvent` timing.

### Comparison to PyTorch SDPA

PyTorch's `F.scaled_dot_product_attention` dispatches to highly optimized FlashAttention/memory-efficient kernels. This hand-rolled kernel reaches ~[X]% of SDPA throughput at N=2048, which is expected — SDPA kernels use warp-level matrix ops (WMMA), register-level tiling, and have been tuned across GPU architectures.

## Nsight Compute

<!-- Fill in after running: ncu --set full ./attention bench -->

| Metric | Value |
|---|---|
| Achieved occupancy | |
| SM throughput | |
| Memory throughput | |
| Top warp stall reason | |

## Build & run

Requires: CUDA toolkit, Python 3 with PyTorch (for reference generation).

```bash
# 1. Generate reference data
python generate_reference.py

# 2. Build (auto-detects GPU arch, or set explicitly)
make ARCH=sm_80          # A100
make ARCH=sm_89          # RTX 4090 / L40

# 3. Correctness test
./attention test test_data

# 4. Benchmark
./attention bench

# 5. Profile (needs ncu permissions)
ncu --set full -o profile_report ./attention bench
```

Or run everything at once:

```bash
python test_correctness.py
```

### Tile size tuning

Default tiles are TILE_Q=16, TILE_KV=16. Override at build time:

```bash
make TILE_Q=32 TILE_KV=32
```

Larger tiles increase shared memory usage and may reduce occupancy. Profile to find the sweet spot for your GPU.

## Architecture

**Naive kernel** (baseline): Three-phase — compute N×N score matrix in global memory, row-wise softmax, matrix multiply by V. One thread per element in each phase.

**Fused kernel**: Single-pass over K/V tiles per Q-block. Thread layout is `(32, TILE_Q)` — one warp across the head dimension, one thread per query row. Each thread maintains `HEAD_DIM/32` accumulator registers. Score computation uses warp shuffle reduction; the P×V accumulation is register-parallel across d.

Shared memory per block: `(TILE_Q + 2×TILE_KV) × HEAD_DIM × 4 + TILE_Q × TILE_KV × 4` bytes. With defaults (16, 16, 64): ~14 KB.

## Limitations

- **fp32 only.** No fp16/bf16, no tensor core (WMMA) path.
- **No backward pass.** Forward attention only.
- **Fixed head dimension.** HEAD_DIM is a compile-time constant (default 64).
- **No dropout.** Attention dropout is not implemented.
- **No grouped-query / multi-query attention.** Assumes H_q == H_kv.
- **No paged or chunked KV cache.** Processes the full sequence each call.
- **Single GPU, single stream.** No multi-GPU or stream overlap.

## References

| What | Where |
|---|---|
| FlashAttention (Algorithm 1) | [Dao et al., 2022](https://arxiv.org/abs/2205.14135) |
| Online softmax numerics | [Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867) |
| CUDA shared memory / tiling | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| Tiling intuition | [Simon Boehm, "How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM) |
