#include "attention.cuh"

__global__ void naive_compute_scores(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N, int d, bool causal)
{
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float scale = rsqrtf((float)d);
    if (causal && col > row) {
        S[bh * N * N + row * N + col] = -INFINITY;
        return;
    }

    const float* q_row = Q + bh * N * d + row * d;
    const float* k_col = K + bh * N * d + col * d;
    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += q_row[i] * k_col[i];
    S[bh * N * N + row * N + col] = dot * scale;
}

__global__ void naive_softmax_rows(float* __restrict__ S, int N)
{
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float* s_row = S + bh * N * N + row * N;
    float m = -INFINITY;
    for (int j = 0; j < N; j++) m = fmaxf(m, s_row[j]);
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        s_row[j] = expf(s_row[j] - m);
        sum += s_row[j];
    }
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; j++) s_row[j] *= inv_sum;
}

__global__ void naive_attn_times_v(
    const float* __restrict__ S,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d)
{
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z * blockDim.x + threadIdx.x;
    if (row >= N || dim >= d) return;

    const float* s_row = S + bh * N * N + row * N;
    float acc = 0.0f;
    for (int j = 0; j < N; j++)
        acc += s_row[j] * V[bh * N * d + j * d + dim];
    O[bh * N * d + row * d + dim] = acc;
}

void naive_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal,
    float* workspace)
{
    int BH = B * H;
    float* S = workspace;
    bool own_S = (S == nullptr);
    if (own_S)
        CUDA_CHECK(cudaMalloc(&S, (size_t)BH * N * N * sizeof(float)));

    {
        dim3 block(16, 16);
        dim3 grid(BH, cdiv(N, 16), cdiv(N, 16));
        naive_compute_scores<<<grid, block>>>(Q, K, S, N, d, causal);
    }
    {
        int threads = 256;
        dim3 grid(BH, cdiv(N, threads));
        naive_softmax_rows<<<grid, threads>>>(S, N);
    }
    {
        dim3 block(16, 16);
        dim3 grid(BH, cdiv(N, 16), cdiv(d, 16));
        naive_attn_times_v<<<grid, block>>>(S, V, O, N, d);
    }

    CUDA_CHECK(cudaGetLastError());
    if (own_S) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(S));
    }
}
