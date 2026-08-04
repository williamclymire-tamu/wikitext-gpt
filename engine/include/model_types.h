#pragma once

// Plain C++ model description -- deliberately free of any CUDA include so the
// scalar CPU reference and the test harness can be built with g++ alone, on a
// machine with no toolkit and no GPU.
//
// The same structs describe host and device weights; only the pointers differ.

struct ModelConfig {
    int vocab_size;
    int context_len;
    int n_layer;
    int n_head;
    int d_model;
    int d_ff;
    int head_dim;  // d_model / n_head
};

struct LayerWeights {
    float* ln1_weight;   // [d_model]
    float* ln1_bias;     // [d_model]
    float* qkv_weight;   // [3*d_model, d_model]   nn.Linear layout: [out, in]
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

struct ModelWeights {
    ModelConfig cfg;
    float* tok_emb;       // [vocab_size, d_model]
    float* pos_emb;       // [context_len, d_model]
    float* ln_f_weight;   // [d_model]
    float* ln_f_bias;     // [d_model]
    float* lm_head;       // [vocab_size, d_model]
    LayerWeights* layers; // [n_layer]  (always host-side array of pointers)
};
