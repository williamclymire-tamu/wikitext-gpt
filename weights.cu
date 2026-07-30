#include "inference.cuh"
#include "weights_host.h"

#include <cstdlib>

// Host -> device staging. Loading and uploading are kept separate on purpose:
// weights_host.cpp is the only code that touches the file format, and it builds
// without CUDA, so the CPU reference and the GPU engine can never disagree about
// what a given .bin file contains.

static float* upload(const float* host, size_t n)
{
    float* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

ModelWeights load_model_weights(const char* dir)
{
    ModelWeights h = load_host_weights(dir);
    const ModelConfig& c = h.cfg;

    ModelWeights d;
    d.cfg = c;

    d.tok_emb     = upload(h.tok_emb,     (size_t)c.vocab_size * c.d_model);
    d.pos_emb     = upload(h.pos_emb,     (size_t)c.context_len * c.d_model);
    d.ln_f_weight = upload(h.ln_f_weight, c.d_model);
    d.ln_f_bias   = upload(h.ln_f_bias,   c.d_model);
    d.lm_head     = upload(h.lm_head,     (size_t)c.vocab_size * c.d_model);

    // The array of LayerWeights stays on the host; only the tensors it points
    // at live on the device. Nothing on the GPU ever dereferences this struct,
    // so there is no reason to pay for a device-side pointer table.
    d.layers = (LayerWeights*)calloc(c.n_layer, sizeof(LayerWeights));

    for (int i = 0; i < c.n_layer; i++) {
        const LayerWeights& s = h.layers[i];
        LayerWeights& t = d.layers[i];

        t.ln1_weight = upload(s.ln1_weight, c.d_model);
        t.ln1_bias   = upload(s.ln1_bias,   c.d_model);
        t.qkv_weight = upload(s.qkv_weight, (size_t)3 * c.d_model * c.d_model);
        t.qkv_bias   = upload(s.qkv_bias,   (size_t)3 * c.d_model);
        t.out_weight = upload(s.out_weight, (size_t)c.d_model * c.d_model);
        t.out_bias   = upload(s.out_bias,   c.d_model);
        t.ln2_weight = upload(s.ln2_weight, c.d_model);
        t.ln2_bias   = upload(s.ln2_bias,   c.d_model);
        t.fc1_weight = upload(s.fc1_weight, (size_t)c.d_ff * c.d_model);
        t.fc1_bias   = upload(s.fc1_bias,   c.d_ff);
        t.fc2_weight = upload(s.fc2_weight, (size_t)c.d_model * c.d_ff);
        t.fc2_bias   = upload(s.fc2_bias,   c.d_model);
    }

    free_host_weights(h);
    return d;
}

void free_model_weights(ModelWeights& m)
{
    cudaFree(m.tok_emb);
    cudaFree(m.pos_emb);
    cudaFree(m.ln_f_weight);
    cudaFree(m.ln_f_bias);
    cudaFree(m.lm_head);

    for (int i = 0; i < m.cfg.n_layer; i++) {
        LayerWeights& L = m.layers[i];
        cudaFree(L.ln1_weight); cudaFree(L.ln1_bias);
        cudaFree(L.qkv_weight); cudaFree(L.qkv_bias);
        cudaFree(L.out_weight); cudaFree(L.out_bias);
        cudaFree(L.ln2_weight); cudaFree(L.ln2_bias);
        cudaFree(L.fc1_weight); cudaFree(L.fc1_bias);
        cudaFree(L.fc2_weight); cudaFree(L.fc2_bias);
    }
    free(m.layers);
    m.layers = nullptr;
}
