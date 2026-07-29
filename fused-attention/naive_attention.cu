/*
 * Step 2: Naive CUDA attention kernel.
 *
 * One thread block per (batch, head). Materializes the full N×N score matrix
 * in global memory, applies softmax, then multiplies by V.
 *
 * This is deliberately the thing the fused kernel replaces.
 * It exists to (a) validate the correctness plumbing and (b) serve as the
 * speedup baseline.
 */

#include "attention.cuh"

// ── Phase 1: compute S = Q @ K^T * scale, apply causal mask ─────────────────
__global__ void naive_compute_scores(
    const float* __restrict__ Q,    // (B, H, N, d)
    const float* __restrict__ K,    // (B, H, N, d)
    float* __restrict__ S,          // (B, H, N, N)
    int N, int d, bool causal)
{
    int bh = blockIdx.x;            // flattened (batch, head) index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float scale = rsqrtf((float)d);

    // Causal: positions where col > row are masked out
    if (causal && col > row) {
        S[bh * N * N + row * N + col] = -INFINITY;
        return;
    }

    const float* q_row = Q + bh * N * d + row * d;
    const float* k_col = K + bh * N * d + col * d;
    float dot = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += q_row[i] * k_col[i];
    }
    S[bh * N * N + row * N + col] = dot * scale;
}

// ── Phase 2: row-wise softmax over S in-place ───────────────────────────────
__global__ void naive_softmax_rows(float* __restrict__ S, int N)
{
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float* s_row = S + bh * N * N + row * N;

    // Find max for numerical stability
    float m = -INFINITY;
    for (int j = 0; j < N; j++) m = fmaxf(m, s_row[j]);

    // Exponentiate and sum
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        s_row[j] = expf(s_row[j] - m);
        sum += s_row[j];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; j++) {
        s_row[j] *= inv_sum;
    }
}

// ── Phase 3: O = S @ V ──────────────────────────────────────────────────────
__global__ void naive_attn_times_v(
    const float* __restrict__ S,    // (B, H, N, N)
    const float* __restrict__ V,    // (B, H, N, d)
    float* __restrict__ O,          // (B, H, N, d)
    int N, int d)
{
    int bh = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.z * blockDim.x + threadIdx.x;
    if (row >= N || dim >= d) return;

    const float* s_row = S + bh * N * N + row * N;
    float acc = 0.0f;
    for (int j = 0; j < N; j++) {
        acc += s_row[j] * V[bh * N * d + j * d + dim];
    }
    O[bh * N * d + row * d + dim] = acc;
}

// ── Host wrapper ─────────────────────────────────────────────────────────────
void naive_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal,
    float* workspace)
{
    int BH = B * H;

    // Allocate N×N score matrix (or use caller-provided workspace)
    float* S = workspace;
    bool own_S = (S == nullptr);
    if (own_S)
        CUDA_CHECK(cudaMalloc(&S, (size_t)BH * N * N * sizeof(float)));

    // Phase 1: S = Q @ K^T * scale
    {
        dim3 block(16, 16);
        dim3 grid(BH, cdiv(N, 16), cdiv(N, 16));
        naive_compute_scores<<<grid, block>>>(Q, K, S, N, d, causal);
    }

    // Phase 2: softmax
    {
        int threads = 256;
        dim3 grid(BH, cdiv(N, threads));
        naive_softmax_rows<<<grid, threads>>>(S, N);
    }

    // Phase 3: O = S @ V
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
    // When using external workspace, caller is responsible for sync
}
