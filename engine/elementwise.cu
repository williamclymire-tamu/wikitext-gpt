#include "elementwise.cuh"

// LayerNorm: one warp per row, two-pass (mean then variance) with shuffle reductions.

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int d, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int tx = threadIdx.x;   // 0..31

    const float* x = input + row * d;
    float* y = output + row * d;

    // mean
    float sum = 0.0f;
    for (int i = tx; i < d; i += 32)
        sum += x[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    float mean = __shfl_sync(0xffffffff, sum, 0) / (float)d;

    // variance
    float var_sum = 0.0f;
    for (int i = tx; i < d; i += 32) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    float var = __shfl_sync(0xffffffff, var_sum, 0) / (float)d;

    float inv_std = rsqrtf(var + eps);

    // normalize + scale/shift
    for (int i = tx; i < d; i += 32)
        y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}

void layernorm_cuda(
    const float* input, const float* weight, const float* bias,
    float* output, int N, int d, float eps)
{
    layernorm_kernel<<<N, 32>>>(input, weight, bias, output, N, d, eps);
    CUDA_CHECK(cudaGetLastError());
}


// GELU tanh approximation (GPT-2 style)

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    // sqrt(2/pi) ≈ 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    output[i] = 0.5f * x * (1.0f + tanhf(inner));
}

void gelu_cuda(const float* input, float* output, int n)
{
    int threads = 256;
    int blocks = cdiv(n, threads);
    gelu_kernel<<<blocks, threads>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}


// Residual add: output[i] = a[i] + b[i]

__global__ void residual_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = a[i] + b[i];
}

void residual_add_cuda(const float* a, const float* b, float* output, int n)
{
    int threads = 256;
    int blocks = cdiv(n, threads);
    residual_add_kernel<<<blocks, threads>>>(a, b, output, n);
    CUDA_CHECK(cudaGetLastError());
}
