# Inference Engine

A standalone C++/CUDA inference engine for the WikiText-103 GPT model — fused
attention kernels, KV cache, scalar CPU reference, and a stage-by-stage parity
harness against the PyTorch forward pass.

## Results

**GPU:** Tesla T4 (SM 7.5, 40 SMs)

### Prefill attention (fused vs naive)

| N    | Naive (ms) | Fused (ms) | Speedup |
|------|-----------|-----------|---------|
| 128  | 0.2638    | 0.1033    | 2.55x   |
| 256  | 0.9927    | 0.3382    | 2.94x   |
| 512  | 2.3262    | 0.8053    | 2.89x   |
| 1024 | 8.8222    | 3.1242    | 2.82x   |
| 2048 | 36.7370   | 12.6644   | 2.90x   |

### Decode (single-query, KV cache)

| seq_len | Decode (ms) |
|---------|------------|
| 128     | 0.1197     |
| 256     | 0.2378     |
| 512     | 0.4739     |
| 1024    | 1.3724     |
| 2048    | 3.2603     |

### End-to-end engine (4L/4H/256d, 7.42M params, 46.1 val PPL)

| Metric | Value |
|---|---|
| Decode throughput | **688 tok/s** (1.453 ms/token, batch 1, context 1-255) |
| Weights loaded | 11.61M fp32 (46.5 MB) |
| Prefill parity vs PyTorch | max_abs **9.0e-5** on logits, **0/32** argmax mismatches |
| Scalar CPU reference vs PyTorch | max_abs **1.3e-5** on logits |
| Decode vs prefill | max_abs **1.4e-5**, 0/32 rows differing |

## Quick Start

```bash
# 1. Export weights (from repo root)
python train/export_weights.py --random          # or from a trained checkpoint

# 2. CPU-only validation (no GPU required)
cd engine
make test_cpu
../tests/test_cpu ../export

# 3. Build and run the engine (GPU)
make engine ARCH=sm_75 HEAD_DIM=64
./engine parity   ../export
./engine generate ../export --ids 464,1092 --tokens 100 --temp 0.8 --top-k 40
./engine bench    ../export --tokens 256

# 4. Kernel tests (GPU)
python ../tests/gen_fixtures.py all
make test_kernels ARCH=sm_75
../tests/test_kernels attention test ../tests/test_data
../tests/test_kernels decode    test ../tests/test_decode_data
../tests/test_kernels attention bench
../tests/test_kernels decode    bench
```

## Architecture

The fused kernel processes attention in tiles without materializing the full
N x N score matrix in global memory:

1. **Load Q tile** (TILE_Q x d) into shared memory once per block
2. **Stream K/V tiles** (TILE_KV x d) through shared memory
3. **QK^T** computed with warp-level shuffle reduction (32 threads across head dim)
4. **Online softmax** (Milakov & Gimelshein 2018): maintains running row-max and
   row-sum, applies correction `exp(old_max - new_max)` to rescale accumulated
   output when new tiles introduce larger values
5. **Causal masking** applied inside the tile loop, with early exit when the KV
   tile is entirely past the diagonal

Thread layout: `blockDim=(32, TILE_Q)` — one warp spans the head dimension, one
thread per query row in the tile.

## Correctness Strategy

Three implementations of the same forward pass, each checking the next:

| Implementation | Where | Role |
|---|---|---|
| PyTorch (or NumPy) | `train/export_weights.py` / `train/make_test_export.py` | writes per-stage fixtures |
| Scalar C++ (`runtime/reference.cpp`) | CPU, double accumulation | the oracle; builds with plain g++ |
| CUDA (`runtime/transformer.cu`) | GPU | the thing being tested |

Fixtures are dumped per stage (embedding, each block, final norm, logits), so a
parity failure names the layer instead of just reporting that the logits are wrong.

## Files

```
engine/
  run.cu                          Entry point: parity, generate, bench
  Makefile                        Builds engine, test_kernels, test_cpu
  kernels/
    fused_attention.cu            Tiled prefill kernel with online streaming softmax
    fused_attention_decode.cu     Single-query decode kernel with KV cache + append
    naive_attention.cu            Three-phase baseline (materializes N x N)
    elementwise.cu                LayerNorm (warp shuffle), GELU (tanh), residual add
    linear.cu                     cuBLAS GEMM wrapper + bias add kernel
  runtime/
    transformer.cu                RunState, KV cache, prefill/decode, sampling
    weights.cu                    Host -> device weight upload
    weights_host.cpp              config.json parse + .bin loading (no CUDA)
    reference.cpp                 Scalar CPU forward pass — the oracle
    detokenizer.cpp               Raw-byte token table lookup
  include/
    attention.cuh                 CUDA_CHECK, tile defines, kernel declarations
    elementwise.cuh               LayerNorm/GELU/residual declarations
    linear.cuh                    cuBLAS wrapper declarations
    transformer.cuh               RunState, forward pass declarations
    inference.cuh                 Top-level inference API
    model_types.h                 CUDA-free struct definitions (shared CPU/GPU)
    weights_host.h                Weight loading declarations
    reference.h                   CPU reference declarations
    detokenizer.h                 Detokenizer declarations
    checks.h                      Shared allclose / argmax verdicts + load_bin()
tests/
  test_kernels.cu                 Unified GPU kernel tests + benchmarks
  test_cpu.cpp                    GPU-free harness for reference + KV cache
  gen_fixtures.py                 PyTorch reference generator for all kernel tests
```

## Limitations

- fp32 only (no fp16/bf16 tensor core path)
- Forward pass only (no backward)
- Fixed head dim (HEAD_DIM=64, compile-time; checked against config.json at startup)
- No dropout, no GQA/MQA
- KV cache is contiguous and single-sequence, not paged; batch size 1
- Prefill assumes position 0 (no prefix reuse / chunked prefill)
- Sampling runs on the host (a 64 KB copy per token)
- Single-kernel design (no split-K for very long sequences)

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)

## AI disclosure

AI tools (Claude) were used for some code review, refactoring, documentation, and
debugging during development. All architecture, training, and evaluation code was
written and is fully understood and explainable by me.
