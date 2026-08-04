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

