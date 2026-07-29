#pragma once

#include "attention.cuh"
#include "elementwise.cuh"
#include "linear.cuh"

// Model config (matches config.py defaults)
struct ModelConfig {
    int vocab_size;
    int context_len;
    int n_layer;
    int n_head;
    int d_model;
    int d_ff;
    int head_dim;  // d_model / n_head
};

// Per-layer weights (all on GPU)
struct LayerWeights {
    float* ln1_weight;   // [d_model]
    float* ln1_bias;     // [d_model]
    float* qkv_weight;   // [3*d_model, d_model]
    float* qkv_bias;     // [3*d_model]
    float* out_weight;   // [d_model, d_model]
    float* out_bias;     // [d_model]
    float* ln2_weight;   // [d_model]
    float* ln2_bias;     // [d_model]
    float* fc1_weight;   // [d_ff, d_model]
    float* fc1_bias;     // [d_ff]
    float* fc2_weight;   // [d_model, d_ff]
    float* fc2_bias;     // [d_model]
};

// Full model weights
struct ModelWeights {
    ModelConfig cfg;
    float* tok_emb;      // [vocab_size, d_model]
    float* pos_emb;      // [context_len, d_model]
    float* ln_f_weight;  // [d_model]
    float* ln_f_bias;    // [d_model]
    float* lm_head;      // [vocab_size, d_model]
    LayerWeights* layers; // [n_layer]
};

// Load all weights from a directory of .bin files exported by export_weights.py.
// Allocates GPU memory. Call free_model_weights() when done.
ModelWeights load_model_weights(const char* dir);
void free_model_weights(ModelWeights& m);
