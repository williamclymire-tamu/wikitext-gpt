#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// ── Error checking ──────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ── Tile sizes ──────────────────────────────────────────────────────────────
// TILE_Q  = number of query rows per thread block
// TILE_KV = number of key/value rows loaded per iteration
// HEAD_DIM = head dimension (d), fixed at compile time for register allocation
#ifndef TILE_Q
#define TILE_Q 16
#endif

#ifndef TILE_KV
#define TILE_KV 16
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

// ── Utility ─────────────────────────────────────────────────────────────────
__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ── Kernel declarations ─────────────────────────────────────────────────────

// Step 2: Naive attention — materializes full N×N score matrix in global memory
// If workspace is nullptr, allocates/frees internally. Pass pre-allocated
// B*H*N*N floats for benchmark loops to avoid timing cudaMalloc.
void naive_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal,
    float* workspace = nullptr);

// Step 3: Fused tiled attention with online softmax — no N×N materialization
void fused_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal);

// ── File I/O ────────────────────────────────────────────────────────────────
inline float* load_bin(const char* path, size_t num_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* buf = (float*)malloc(num_floats * sizeof(float));
    size_t read = fread(buf, sizeof(float), num_floats, f);
    if (read != num_floats) {
        fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, num_floats, read);
        exit(1);
    }
    fclose(f);
    return buf;
}

// ── Correctness check ───────────────────────────────────────────────────────
inline bool check(const float* a, const float* b, size_t n,
                  float rtol, const char* label) {
    float max_abs = 0, max_rel = 0;
    size_t worst_i = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float denom = fmaxf(fabsf(b[i]), 1e-8f);
        if (diff > max_abs) { max_abs = diff; worst_i = i; }
        float rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
    }
    bool pass = max_rel < rtol;
    printf("  %-25s max_abs=%.2e  max_rel=%.2e  [%s]\n",
           label, max_abs, max_rel, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("    worst at i=%zu: got %.6f expected %.6f\n",
               worst_i, a[worst_i], b[worst_i]);
    }
    return pass;
}
