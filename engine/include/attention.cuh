#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#ifndef TILE_Q
#define TILE_Q 16
#endif

#ifndef TILE_KV
#define TILE_KV 16
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

__host__ __device__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

void naive_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal,
    float* workspace = nullptr);

void fused_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal);

// Single-query decode: Q [B*H, d], K/V cache [B*H, max_seq, d], O [B*H, d]
void fused_attention_decode_cuda(
    const float* Q, const float* K_cache, const float* V_cache, float* O,
    int B, int H, int seq_len, int max_seq, int d);

// Append new K,V row to cache at position pos
void kv_cache_append_cuda(
    float* K_cache, float* V_cache,
    const float* K_new, const float* V_new,
    int B, int H, int pos, int max_seq, int d);

// ── Utilities used by the standalone kernel tests ────────────────────────────
// The engine proper uses read_f32() from weights_host.h and check_close() from
// checks.h instead. These stay here so the kernel tests (test_attention,
// test_decode) can build without linking weights_host.cpp.

inline float* load_bin(const char* path, size_t num_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* buf = (float*)malloc(num_floats * sizeof(float));
    size_t nread = fread(buf, sizeof(float), num_floats, f);
    if (nread != num_floats) {
        fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, num_floats, nread);
        exit(1);
    }
    fclose(f);
    return buf;
}

// allclose check: |a-b| <= atol + rtol*|b|  (same as numpy)
inline bool check(const float* a, const float* b, size_t n,
                  float rtol, const char* label) {
    const float atol = 1e-5f;
    float max_abs = 0, max_rel = 0;
    size_t worst_abs_i = 0;
    int num_fail = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float tol = atol + rtol * fabsf(b[i]);
        if (diff > max_abs) { max_abs = diff; worst_abs_i = i; }
        float denom = fmaxf(fabsf(b[i]), 1e-8f);
        float rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        if (diff > tol) num_fail++;
    }
    bool pass = (num_fail == 0);
    printf("  %-25s max_abs=%.2e  max_rel=%.2e  fail=%d/%zu  [%s]\n",
           label, max_abs, max_rel, num_fail, n, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("    worst at i=%zu: got %.6f expected %.6f (diff=%.2e)\n",
               worst_abs_i, a[worst_abs_i], b[worst_abs_i], max_abs);
    }
    return pass;
}
