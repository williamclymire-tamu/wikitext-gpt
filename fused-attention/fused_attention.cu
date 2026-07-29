/*
 * Step 3: Fused tiled attention with online streaming softmax.
 *
 * FlashAttention-style: loops over K/V tiles, never materializes the N×N
 * score matrix. Maintains running (m, l, acc) per query row and rescales
 * accumulated output when a new tile introduces a larger row-max.
 *
 * Thread layout:
 *   gridDim.x  = B * H
 *   gridDim.y  = ceil(N / TILE_Q)
 *   blockDim.x = 32          (one warp, handles HEAD_DIM via stride loop)
 *   blockDim.y = TILE_Q      (one thread per query row in the tile)
 *
 * Shared memory:
 *   Q_s  [TILE_Q ][HEAD_DIM]
 *   K_s  [TILE_KV][HEAD_DIM]
 *   V_s  [TILE_KV][HEAD_DIM]
 *   S_s  [TILE_Q ][TILE_KV]   (scores after QK^T reduction)
 *
 * Algorithm reference: FlashAttention (Dao et al., 2022), Algorithm 1.
 * Numerics reference: Online softmax (Milakov & Gimelshein, 2018).
 */

#include "attention.cuh"

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,    // (B*H, N, d)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int d, bool causal)
{
    const int bh = blockIdx.x;                      // which (batch, head)
    const int q_start = blockIdx.y * TILE_Q;        // first query row for this block
    const int ty = threadIdx.y;                     // query row within tile [0, TILE_Q)
    const int tx = threadIdx.x;                     // lane within warp [0, 32)
    const int q_row = q_start + ty;                 // global query row

    const float scale = rsqrtf((float)d);

    // ── Shared memory ───────────────────────────────────────────────────────
    __shared__ float Q_s[TILE_Q][HEAD_DIM];
    __shared__ float K_s[TILE_KV][HEAD_DIM];
    __shared__ float V_s[TILE_KV][HEAD_DIM];
    __shared__ float S_s[TILE_Q][TILE_KV];

    // ── Per-thread running state (registers) ────────────────────────────────
    // Each thread (tx, ty) maintains HEAD_DIM/32 accumulator elements.
    // For HEAD_DIM=64, that's 2 floats per thread.
    float acc[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) acc[i] = 0.0f;

    float row_max = -INFINITY;  // m_i: running max of all scores for this query row
    float row_sum = 0.0f;       // l_i: running sum of exp(s - m) for this query row

    // ── Load Q tile into shared memory ──────────────────────────────────────
    // Each thread loads HEAD_DIM/32 elements of its query row
    for (int dd = tx; dd < d; dd += 32) {
        if (q_row < N)
            Q_s[ty][dd] = Q[bh * N * d + q_row * d + dd];
        else
            Q_s[ty][dd] = 0.0f;
    }
    __syncthreads();

    // ── Loop over K/V tiles ─────────────────────────────────────────────────
    int num_kv_tiles = cdiv(N, TILE_KV);
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * TILE_KV;

        // Early exit for causal: if the entire KV tile is above the diagonal
        // for all query rows in this Q tile, skip it.
        if (causal && kv_start > q_start + TILE_Q - 1) break;

        // ── Load K tile ─────────────────────────────────────────────────────
        for (int r = ty; r < TILE_KV; r += TILE_Q) {
            int kv_row = kv_start + r;
            for (int dd = tx; dd < d; dd += 32) {
                if (kv_row < N)
                    K_s[r][dd] = K[bh * N * d + kv_row * d + dd];
                else
                    K_s[r][dd] = 0.0f;
            }
        }

        // ── Load V tile ─────────────────────────────────────────────────────
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

        // ── Compute S = Q_s @ K_s^T * scale ─────────────────────────────────
        // Each thread (tx, ty) computes partial dot products, then we reduce
        // across tx (the 32 lanes of the warp) to get S_s[ty][j].
        for (int j = 0; j < TILE_KV; j++) {
            int kv_col = kv_start + j;

            float dot = 0.0f;
            for (int dd = tx; dd < d; dd += 32) {
                dot += Q_s[ty][dd] * K_s[j][dd];
            }

            // Warp reduction (32 threads → 1 sum)
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);

            if (tx == 0) {
                float s = dot * scale;

                // Causal mask: if key position > query position, mask out
                if (causal && kv_col > q_row) s = -INFINITY;

                // Out-of-bounds keys
                if (q_row >= N || kv_col >= N) s = -INFINITY;

                S_s[ty][j] = s;
            }
        }
        __syncthreads();

        // ── Online softmax update ───────────────────────────────────────────
        // Only thread tx==0 computes the row statistics, then broadcasts.
        // But all threads need the result for the P @ V step.

        // Find tile max and compute new running max
        float tile_max = -INFINITY;
        if (tx == 0) {
            for (int j = 0; j < TILE_KV; j++)
                tile_max = fmaxf(tile_max, S_s[ty][j]);
        }
        // Broadcast tile_max from lane 0 to all lanes
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        float m_new = fmaxf(row_max, tile_max);

        // Correction factor: rescale previous accumulations to new max
        float correction = expf(row_max - m_new);

        // Update running sum: scale old sum by correction, add new exponentials
        float tile_sum = 0.0f;
        if (tx == 0) {
            for (int j = 0; j < TILE_KV; j++) {
                S_s[ty][j] = expf(S_s[ty][j] - m_new);  // P[ty][j]
                tile_sum += S_s[ty][j];
            }
        }
        tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);
        __syncthreads();  // S_s now contains P values, needed by all threads

        row_sum = row_sum * correction + tile_sum;

        // Rescale existing accumulator and add P @ V_tile contribution
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int dd = tx + i * 32;
            acc[i] *= correction;
            for (int j = 0; j < TILE_KV; j++) {
                acc[i] += S_s[ty][j] * V_s[j][dd];
            }
        }

        row_max = m_new;
        __syncthreads();  // Protect shared memory before next tile overwrites it
    }

    // ── Write output: O = acc / row_sum ─────────────────────────────────────
    if (q_row < N) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int dd = tx + i * 32;
            if (dd < d)
                O[bh * N * d + q_row * d + dd] = acc[i] * inv_sum;
        }
    }
}

// ── Host wrapper ─────────────────────────────────────────────────────────────
void fused_attention_cuda(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, bool causal)
{
    int BH = B * H;
    dim3 block(32, TILE_Q);                     // 32 × TILE_Q threads
    dim3 grid(BH, cdiv(N, TILE_Q));

    fused_attention_kernel<<<grid, block>>>(Q, K, V, O, N, d, causal);

    CUDA_CHECK(cudaGetLastError());
    // No sync here — caller syncs via cudaEvent or cudaMemcpy
}
