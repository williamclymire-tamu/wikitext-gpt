#include "transformer.cuh"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

// ------------------------------------------------------------------ layout kernels
//
// The compute kernels already existed; what was missing was the plumbing between
// them. All three of these exist because attention wants [n_head, T, head_dim]
// while nn.Linear produces [T, d_model]. Getting this transpose wrong produces a
// model that runs at full speed and emits plausible-looking garbage, which is
// why the CPU reference writes the same index arithmetic out longhand.

__global__ void embed_kernel(
    const int* __restrict__ tokens,
    const float* __restrict__ tok_emb,
    const float* __restrict__ pos_emb,
    float* __restrict__ x,
    int T, int d, int pos0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * d) return;
    int t = idx / d;
    int j = idx - t * d;
    x[idx] = tok_emb[(size_t)tokens[t] * d + j]
           + pos_emb[(size_t)(pos0 + t) * d + j];
}

// [T, 3*d] -> q, k, v each [H, T, hd]
__global__ void split_qkv_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    int T, int d, int H, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * d) return;
    int t   = idx / d;
    int rem = idx - t * d;
    int h   = rem / hd;
    int j   = rem - h * hd;

    size_t src = (size_t)t * 3 * d + rem;
    size_t dst = ((size_t)h * T + t) * hd + j;

    q[dst] = qkv[src];
    k[dst] = qkv[src + d];
    v[dst] = qkv[src + 2 * d];
}

// [H, T, hd] -> [T, d]
__global__ void merge_heads_kernel(
    const float* __restrict__ heads, float* __restrict__ out,
    int T, int d, int H, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * d) return;
    int t   = idx / d;
    int rem = idx - t * d;
    int h   = rem / hd;
    int j   = rem - h * hd;
    out[idx] = heads[((size_t)h * T + t) * hd + j];
}

// Bulk-load a prefill's worth of K/V into the cache. Decode uses the existing
// single-position kv_cache_append_cuda instead.
__global__ void kv_store_prefill_kernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    int T, int H, int hd, int max_seq, int pos0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * hd;
    if (idx >= total) return;

    int j = idx % hd;
    int t = (idx / hd) % T;
    int h = idx / (hd * T);

    size_t dst = ((size_t)h * max_seq + pos0 + t) * hd + j;
    k_cache[dst] = k[idx];
    v_cache[dst] = v[idx];
}

// All the layout kernels are 1D and element-per-thread; this keeps the launch
// arithmetic in one place instead of repeated at every call site.
#define LAUNCH1D(kernel, n, ...)                                   \
    do {                                                           \
        int _t = 256;                                              \
        kernel<<<cdiv((n), _t), _t>>>(__VA_ARGS__);                \
        CUDA_CHECK(cudaGetLastError());                            \
    } while (0)

// ------------------------------------------------------------------ run state

RunState runstate_alloc(const ModelConfig& c, int max_seq, int max_tokens)
{
    RunState s;
    s.cfg = c;
    s.max_seq = max_seq;
    s.max_tokens = max_tokens;
    s.pos = 0;

    size_t cache_elems = (size_t)c.n_head * max_seq * c.head_dim;
    s.k_cache = (float**)malloc(sizeof(float*) * c.n_layer);
    s.v_cache = (float**)malloc(sizeof(float*) * c.n_layer);
    for (int i = 0; i < c.n_layer; i++) {
        CUDA_CHECK(cudaMalloc(&s.k_cache[i], cache_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.v_cache[i], cache_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(s.k_cache[i], 0, cache_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(s.v_cache[i], 0, cache_elems * sizeof(float)));
    }

    size_t M = (size_t)max_tokens;
    CUDA_CHECK(cudaMalloc(&s.d_tokens, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.x,      M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.h,      M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.qkv,    M * 3 * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.q,      M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.k,      M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.v,      M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.attn,   M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.merged, M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.proj,   M * c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.ff,     M * c.d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.logits, M * (size_t)c.vocab_size * sizeof(float)));

    // pinned: sampling copies a vocab row back every single token
    CUDA_CHECK(cudaMallocHost(&s.h_logits, (size_t)c.vocab_size * sizeof(float)));

    return s;
}

void runstate_free(RunState& s)
{
    for (int i = 0; i < s.cfg.n_layer; i++) {
        cudaFree(s.k_cache[i]);
        cudaFree(s.v_cache[i]);
    }
    free(s.k_cache);
    free(s.v_cache);

    cudaFree(s.d_tokens);
    cudaFree(s.x); cudaFree(s.h); cudaFree(s.qkv);
    cudaFree(s.q); cudaFree(s.k); cudaFree(s.v);
    cudaFree(s.attn); cudaFree(s.merged); cudaFree(s.proj);
    cudaFree(s.ff); cudaFree(s.logits);
    cudaFreeHost(s.h_logits);
}

void runstate_reset(RunState& s)
{
    s.pos = 0;
    size_t cache_elems = (size_t)s.cfg.n_head * s.max_seq * s.cfg.head_dim;
    for (int i = 0; i < s.cfg.n_layer; i++) {
        CUDA_CHECK(cudaMemset(s.k_cache[i], 0, cache_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(s.v_cache[i], 0, cache_elems * sizeof(float)));
    }
}

static void dump(float* host_dst, const float* dev_src, size_t n)
{
    if (!host_dst) return;
    CUDA_CHECK(cudaMemcpy(host_dst, dev_src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------ prefill

float* engine_prefill(const ModelWeights& m, RunState& s,
                      const int* h_tokens, int T, EngineTrace* trace)
{
    const ModelConfig& c = m.cfg;
    const int d = c.d_model, H = c.n_head, hd = c.head_dim, dff = c.d_ff;

    if (T > s.max_tokens) {
        fprintf(stderr, "prefill: T=%d exceeds max_tokens=%d\n", T, s.max_tokens);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(s.d_tokens, h_tokens, T * sizeof(int), cudaMemcpyHostToDevice));
    LAUNCH1D(embed_kernel, T * d, s.d_tokens, m.tok_emb, m.pos_emb, s.x, T, d, 0);
    if (trace) dump(trace->embed, s.x, (size_t)T * d);

    for (int l = 0; l < c.n_layer; l++) {
        const LayerWeights& L = m.layers[l];

        layernorm_cuda(s.x, L.ln1_weight, L.ln1_bias, s.h, T, d);
        linear_cuda(s.h, L.qkv_weight, L.qkv_bias, s.qkv, T, 3 * d, d);

        LAUNCH1D(split_qkv_kernel, T * d, s.qkv, s.q, s.k, s.v, T, d, H, hd);
        LAUNCH1D(kv_store_prefill_kernel, H * T * hd,
                 s.k, s.v, s.k_cache[l], s.v_cache[l], T, H, hd, s.max_seq, 0);

        fused_attention_cuda(s.q, s.k, s.v, s.attn, 1, H, T, hd, /*causal=*/true);

        LAUNCH1D(merge_heads_kernel, T * d, s.attn, s.merged, T, d, H, hd);
        linear_cuda(s.merged, L.out_weight, L.out_bias, s.proj, T, d, d);
        residual_add_cuda(s.x, s.proj, s.x, T * d);

        layernorm_cuda(s.x, L.ln2_weight, L.ln2_bias, s.h, T, d);
        linear_cuda(s.h, L.fc1_weight, L.fc1_bias, s.ff, T, dff, d);
        gelu_cuda(s.ff, s.ff, T * dff);
        linear_cuda(s.ff, L.fc2_weight, L.fc2_bias, s.proj, T, d, dff);
        residual_add_cuda(s.x, s.proj, s.x, T * d);

        if (trace && trace->layer_out) dump(trace->layer_out[l], s.x, (size_t)T * d);
    }

    layernorm_cuda(s.x, m.ln_f_weight, m.ln_f_bias, s.h, T, d);
    if (trace) dump(trace->final_ln, s.h, (size_t)T * d);

    linear_cuda(s.h, m.lm_head, nullptr, s.logits, T, c.vocab_size, d);
    if (trace) dump(trace->logits, s.logits, (size_t)T * c.vocab_size);

    s.pos = T;
    return s.logits;
}

// ------------------------------------------------------------------ decode

float* engine_decode(const ModelWeights& m, RunState& s, int token)
{
    const ModelConfig& c = m.cfg;
    const int d = c.d_model, H = c.n_head, hd = c.head_dim, dff = c.d_ff;

    if (s.pos >= s.max_seq) {
        fprintf(stderr, "decode: KV cache full at %d\n", s.max_seq);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(s.d_tokens, &token, sizeof(int), cudaMemcpyHostToDevice));
    LAUNCH1D(embed_kernel, d, s.d_tokens, m.tok_emb, m.pos_emb, s.x, 1, d, s.pos);

    for (int l = 0; l < c.n_layer; l++) {
        const LayerWeights& L = m.layers[l];

        layernorm_cuda(s.x, L.ln1_weight, L.ln1_bias, s.h, 1, d);
        linear_cuda(s.h, L.qkv_weight, L.qkv_bias, s.qkv, 1, 3 * d, d);

        // For a single token, [1, 3d] already has q, k and v each laid out as
        // [n_head, head_dim] contiguously -- exactly what the decode kernel and
        // the cache-append kernel expect. No transpose needed on this path.
        const float* q_new = s.qkv;
        const float* k_new = s.qkv + d;
        const float* v_new = s.qkv + 2 * d;

        kv_cache_append_cuda(s.k_cache[l], s.v_cache[l], k_new, v_new,
                             1, H, s.pos, s.max_seq, hd);

        fused_attention_decode_cuda(q_new, s.k_cache[l], s.v_cache[l], s.attn,
                                    1, H, s.pos + 1, s.max_seq, hd);

        linear_cuda(s.attn, L.out_weight, L.out_bias, s.proj, 1, d, d);
        residual_add_cuda(s.x, s.proj, s.x, d);

        layernorm_cuda(s.x, L.ln2_weight, L.ln2_bias, s.h, 1, d);
        linear_cuda(s.h, L.fc1_weight, L.fc1_bias, s.ff, 1, dff, d);
        gelu_cuda(s.ff, s.ff, dff);
        linear_cuda(s.ff, L.fc2_weight, L.fc2_bias, s.proj, 1, d, dff);
        residual_add_cuda(s.x, s.proj, s.x, d);
    }

    layernorm_cuda(s.x, m.ln_f_weight, m.ln_f_bias, s.h, 1, d);
    linear_cuda(s.h, m.lm_head, nullptr, s.logits, 1, c.vocab_size, d);

    s.pos++;
    return s.logits;
}

// ------------------------------------------------------------------ sampling
//
// Host-side on purpose. A vocab row is 64 KB at V=16384, so the copy back is a
// few microseconds against a decode step that costs far more, and partial-sorting
// on the CPU is trivially correct. If the vocabulary or the batch grows enough
// for this to show up in a profile, it moves to the device -- but it should be
// measured first, not assumed.

static float rand_uniform(unsigned long long* state)
{
    // xorshift64*, deterministic across machines so a seeded run is reproducible
    unsigned long long x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return (float)(((x * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0));
}

int sample_token(RunState& s, const float* d_logits,
                 float temperature, int top_k, unsigned long long* rng)
{
    const int V = s.cfg.vocab_size;
    CUDA_CHECK(cudaMemcpy(s.h_logits, d_logits, V * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float* lg = s.h_logits;

    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        return best;
    }

    for (int i = 0; i < V; i++) lg[i] /= temperature;

    int k = (top_k > 0 && top_k < V) ? top_k : V;

    // partial sort by index; only the top k matter
    static std::vector<int> idx;
    idx.resize(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });

    float max_logit = lg[idx[0]];
    float sum = 0.0f;
    static std::vector<float> probs;
    probs.resize(k);
    for (int i = 0; i < k; i++) {
        probs[i] = expf(lg[idx[i]] - max_logit);
        sum += probs[i];
    }

    float r = rand_uniform(rng) * sum;
    float acc = 0.0f;
    for (int i = 0; i < k; i++) {
        acc += probs[i];
        if (r <= acc) return idx[i];
    }
    return idx[k - 1];
}
