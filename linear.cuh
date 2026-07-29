#pragma once

#include "attention.cuh"
#include <cublas_v2.h>

// Global cuBLAS handle (init once, reuse)
void cublas_init();
void cublas_destroy();
cublasHandle_t cublas_handle();

// Y[M, N] = X[M, K] @ W^T[K, N] + bias[N]
// W is stored as [N, K] (standard nn.Linear layout).
// bias can be nullptr.
void linear_cuda(
    const float* X, const float* W, const float* bias, float* Y,
    int M, int N, int K);
