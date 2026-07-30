#include "attention.cuh"

// Single-query decode attention with online softmax.
// One warp per (batch, head). Q in registers, streams through KV cache tiles.
// No causal mask needed since query is always the newest position.

#ifndef TILE_KV_DECODE
#define TILE_KV_DECODE 32
#endif

__global__ void fused_attention_decode_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ O,
    int seq_len, int max_seq, int d)
{
    const int bh = blockIdx.x;
    const int tx = threadIdx.x;   // 0..31

    const float scale = rsqrtf((float)d);

    // Q stays in registers for the whole kernel
    float q[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int dd = tx + i * 32;
        q[i] = (dd < d) ? Q[bh * d + dd] : 0.0f;
    }

    // Online softmax accumulators
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) acc[i] = 0.0f;

    __shared__ float K_s[TILE_KV_DECODE][HEAD_DIM];
    __shared__ float V_s[TILE_KV_DECODE][HEAD_DIM];

    const int num_tiles = cdiv(seq_len, TILE_KV_DECODE);

    for (int tile = 0; tile < num_tiles; tile++) {
        const int kv_start = tile * TILE_KV_DECODE;
        const int tile_len = min(TILE_KV_DECODE, seq_len - kv_start);

        for (int r = 0; r < TILE_KV_DECODE; r++) {
            int kv_row = kv_start + r;
            for (int dd = tx; dd < d; dd += 32) {
                K_s[r][dd] = (kv_row < seq_len)
                    ? K_cache[(bh * max_seq + kv_row) * d + dd]
                    : 0.0f;
            }
        }

        for (int r = 0; r < TILE_KV_DECODE; r++) {
            int kv_row = kv_start + r;
            for (int dd = tx; dd < d; dd += 32) {
                V_s[r][dd] = (kv_row < seq_len)
                    ? V_cache[(bh * max_seq + kv_row) * d + dd]
                    : 0.0f;
            }
        }
        __syncwarp();

        // Q @ K^T for this tile
        float tile_max = -INFINITY;
        float scores[TILE_KV_DECODE];

        for (int j = 0; j < tile_len; j++) {
            float dot = 0.0f;
            for (int i = 0; i < HEAD_DIM / 32; i++) {
                int dd = tx + i * 32;
                dot += q[i] * K_s[j][dd];
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            float s = __shfl_sync(0xffffffff, dot, 0) * scale;
            scores[j] = s;
            tile_max = fmaxf(tile_max, s);
        }

        // online softmax: rescale running accumulators
        float m_new = fmaxf(row_max, tile_max);
        float correction = expf(row_max - m_new);
        for (int i = 0; i < HEAD_DIM / 32; i++)
            acc[i] *= correction;
        row_sum *= correction;

        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; j++) {
            float p = expf(scores[j] - m_new);
            tile_sum += p;
            for (int i = 0; i < HEAD_DIM / 32; i++) {
                int dd = tx + i * 32;
                acc[i] += p * V_s[j][dd];
            }
        }

        row_sum += tile_sum;
        row_max = m_new;
        __syncwarp();
    }

    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int dd = tx + i * 32;
        if (dd < d)
            O[bh * d + dd] = acc[i] * inv_sum;
    }
}

void fused_attention_decode_cuda(
    const float* Q, const float* K_cache, const float* V_cache, float* O,
    int B, int H, int seq_len, int max_seq, int d)
{
    int BH = B * H;
    dim3 block(32);
    dim3 grid(BH);
    fused_attention_decode_kernel<<<grid, block>>>(
        Q, K_cache, V_cache, O, seq_len, max_seq, d);
    CUDA_CHECK(cudaGetLastError());
}


// Append one K,V row to cache at position `pos`.

__global__ void kv_cache_append_kernel(
    float* __restrict__ K_cache,
    float* __restrict__ V_cache,
    const float* __restrict__ K_new,
    const float* __restrict__ V_new,
    int pos, int max_seq, int d)
{
    int bh = blockIdx.x;
    for (int dd = threadIdx.x; dd < d; dd += blockDim.x) {
        K_cache[(bh * max_seq + pos) * d + dd] = K_new[bh * d + dd];
        V_cache[(bh * max_seq + pos) * d + dd] = V_new[bh * d + dd];
    }
}

void kv_cache_append_cuda(
    float* K_cache, float* V_cache,
    const float* K_new, const float* V_new,
    int B, int H, int pos, int max_seq, int d)
{
    int BH = B * H;
    kv_cache_append_kernel<<<BH, 64>>>(
        K_cache, V_cache, K_new, V_new, pos, max_seq, d);
    CUDA_CHECK(cudaGetLastError());
}
