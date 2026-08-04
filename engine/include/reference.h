#pragma once

#include "model_types.h"

// Scalar CPU forward pass. This is the oracle: it is the slowest and the most
// obviously correct implementation in the repo, it accumulates in double, and
// it is the thing every CUDA kernel gets diffed against.
//
// Deliberately no SIMD, no threading, no blocking. If it is ever fast enough to
// matter, something has gone wrong with the priorities.

// Per-stage activations, so a parity failure localizes to a layer instead of
// just reporting "logits are wrong".
struct RefTrace {
    int T;
    float*  embed;      // [T, d_model]      after tok + pos embedding
    float** layer_out;  // [n_layer][T, d_model]
    float*  final_ln;   // [T, d_model]      after ln_f
    float*  logits;     // [T, vocab_size]
};

RefTrace ref_trace_alloc(const ModelConfig& c, int T);
void     ref_trace_free(RefTrace& t, const ModelConfig& c);

// Full prefill over T tokens with a causal mask.
void ref_forward(const ModelWeights& m, const int* tokens, int T, RefTrace& trace);

// ---------------------------------------------------------------- decode

// Incremental decode with a KV cache, mirroring what the CUDA engine does at
// generation time. Exists so the cache logic can be validated against ref_forward
// on CPU before any GPU is involved: decoding token t must produce the same
// logits as row t of a full prefill.
struct RefState {
    ModelConfig cfg;
    int max_seq;
    int pos;            // number of tokens currently in the cache
    float** k_cache;    // [n_layer][n_head, max_seq, head_dim]
    float** v_cache;
    float*  x;          // [d_model] running activation
    float*  scratch;    // sized for the widest intermediate
};

RefState ref_state_alloc(const ModelConfig& c, int max_seq);
void     ref_state_free(RefState& s);

// Consumes one token at position s.pos, advances pos, writes [vocab_size] logits.
void ref_decode_step(const ModelWeights& m, RefState& s, int token, float* logits_out);
