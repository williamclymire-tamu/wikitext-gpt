#include "attention.cuh"

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, bool causal)
{
    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * TILE_Q;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int q_row = q_start + ty;

    const float scale = rsqrtf((float)d);

    __shared__ float Q_s[TILE_Q][HEAD_DIM];
    __shared__ float K_s[TILE_KV][HEAD_DIM];
    __shared__ float V_s[TILE_KV][HEAD_DIM];
    __shared__ float S_s[TILE_Q][TILE_KV];

    float acc[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) acc[i] = 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int dd = tx; dd < d; dd += 32) {
        if (q_row < N)
            Q_s[ty][dd] = Q[bh * N * d + q_row * d + dd];
        else
            Q_s[ty][dd] = 0.0f;
    }
    __syncthreads();

    int num_kv_tiles = cdiv(N, TILE_KV);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_KV;

        // early exit: entire tile past the diagonal
        if (causal && kv_start > q_start + TILE_Q - 1) break;

        for (int r = ty; r < TILE_KV; r += TILE_Q) {
            int kv_row = kv_start + r;
            for (int dd = tx; dd < d; dd += 32) {
                if (kv_row < N)
                    K_s[r][dd] = K[bh * N * d + kv_row * d + dd];
                else
                    K_s[r][dd] = 0.0f;
            }
        }

        for (int r = ty; r < TILE_KV; r += TILE_Q) {
            int kv_row = kv_start + r;
            for (int dd = tx; dd < d; dd += 32) {
                if (kv_row < N)
                    V_s[r][dd] = V[bh * N * d + kv_row * d + dd];
                else
                    V_s[r][dd] = 0.0f;
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale (warp reduction per dot product)
        for (int j = 0; j < TILE_KV; j++) {
            int kv_col = kv_start + j;
            float dot = 0.0f;
            for (int dd = tx; dd < d; dd += 32)
                dot += Q_s[ty][dd] * K_s[j][dd];

            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);

            if (tx == 0) {
                float s = dot * scale;
                if (causal && kv_col > q_row) s = -INFINITY;
                if (q_row >= N || kv_col >= N) s = -INFINITY;
                S_s[ty][j] = s;
            }
        }
        __syncthreads();

        // online softmax correction
        float tile_max = -INFINITY;
        if (tx == 0) {
            for (int j = 0; j < TILE_KV; j++)
                tile_max = fmaxf(tile_max, S_s[ty][j]);
        }
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        float m_new = fmaxf(row_max, tile_max);
        float correction = expf(row_max - m_new);

        float tile_sum = 0.0f;
        if (tx == 0) {
            for (int j = 0; j < TILE_KV; j++) {
                S_s[ty][j] = expf(S_s[ty][j] - m_new);
                tile_sum += S_s[ty][j];
            }
        }
        tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);
        __syncthreads();

        row_sum = row_sum * correction + tile_sum;

        // rescale + accumulate P @ V
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int dd = tx + i * 32;
            acc[i] *= correction;
            for (int j = 0; j < TILE_KV; j++)
                acc[i] += S_s[ty][j] * V_s[j][dd];
        }

        row_max = m_new;
        __syncthreads();
    }

    if (q_row < N) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int dd = tx + i * 32;
            if (dd < d)
                O[bh * N * d + q_row * d + dd] = acc[i] * inv_sum;
        }
    }
}

void fused_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal)
{
    int BH = B * H;
    dim3 block(32, TILE_Q);
    dim3 grid(BH, cdiv(N, TILE_Q));
    fused_attention_kernel<<<grid, block>>>(Q, K, V, O, N, d, causal);
    CUDA_CHECK(cudaGetLastError());
}
