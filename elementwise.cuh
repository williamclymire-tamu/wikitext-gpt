#pragma once

#include "attention.cuh"

void layernorm_cuda(
    const float* input, const float* weight, const float* bias,
    float* output, int N, int d, float eps = 1e-5f);

void gelu_cuda(const float* input, float* output, int n);

void residual_add_cuda(const float* a, const float* b, float* output, int n);
