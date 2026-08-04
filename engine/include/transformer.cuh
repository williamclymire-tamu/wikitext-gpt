#pragma once

#include "inference.cuh"

// Everything that is not a weight: activation scratch, the KV cache, and the
// current sequence position. Allocated once at startup so the decode loop never
// touches cudaMalloc -- allocation on the hot path is the easiest way to turn a
// latency benchmark into a measurement of the allocator.

struct RunState {
    ModelConfig cfg;
    int max_seq;
    int max_tokens;   // widest prefill this state can handle
    int pos;          // tokens currently in the KV cache

    // KV cache: n_layer separate [n_head, max_seq, head_dim] device buffers.
    // Contiguous per layer, not paged -- one sequence, known max length.
    float** k_cache;  // host array of device pointers
    float** v_cache;

    int*   d_tokens;  // [max_tokens]
    float* x;         // [max_tokens, d_model]   residual stream
    float* h;         // [max_tokens, d_model]   post-norm scratch
    float* qkv;       // [max_tokens, 3*d_model]
    float* q;         // [n_head, max_tokens, head_dim]
    float* k;
    float* v;
    float* attn;      // [n_head, max_tokens, head_dim]
    float* merged;    // [max_tokens, d_model]
    float* proj;      // [max_tokens, d_model]
    float* ff;        // [max_tokens, d_ff]
    float* logits;    // [max_tokens, vocab_size]

    float* h_logits;  // host staging for sampling, [vocab_size]
};

RunState runstate_alloc(const ModelConfig& c, int max_seq, int max_tokens);
void     runstate_free(RunState& s);
void     runstate_reset(RunState& s);

// Optional per-stage capture for the parity harness. Any pointer may be null.
// Buffers are host-side and must be sized [T, d_model].
struct EngineTrace {
    float*  embed;
    float** layer_out;   // [n_layer][T, d_model]
    float*  final_ln;
    float*  logits;      // [T, vocab_size]
};

// Prefill positions 0..T-1. Populates the KV cache and leaves pos == T.
// Returns a device pointer to the [T, vocab_size] logits.
float* engine_prefill(const ModelWeights& m, RunState& s,
                      const int* h_tokens, int T, EngineTrace* trace = nullptr);

// One token at the current position. Returns a device pointer to [vocab_size].
float* engine_decode(const ModelWeights& m, RunState& s, int token);

// Copies logits to the host, applies temperature and top-k, samples.
// top_k <= 0 means full-vocabulary sampling; temperature <= 0 means greedy.
int sample_token(RunState& s, const float* d_logits,
                 float temperature, int top_k, unsigned long long* rng);
