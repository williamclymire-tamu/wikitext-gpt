#include "weights_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ------------------------------------------------------------------ file io

float* read_f32(const char* path, size_t num_floats)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weights: cannot open %s\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t want = num_floats * sizeof(float);
    if ((size_t)bytes != want) {
        fprintf(stderr, "weights: %s is %ld bytes, expected %zu "
                        "(config.json disagrees with the export)\n",
                path, bytes, want);
        exit(1);
    }

    float* buf = (float*)malloc(want);
    if (!buf) { fprintf(stderr, "weights: OOM on %s\n", path); exit(1); }

    size_t got = fread(buf, sizeof(float), num_floats, f);
    if (got != num_floats) {
        fprintf(stderr, "weights: short read on %s (%zu/%zu)\n", path, got, num_floats);
        exit(1);
    }
    fclose(f);
    return buf;
}

int* read_i32(const char* path, int* count)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "weights: cannot open %s\n", path); exit(1); }

    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    int n = (int)(bytes / 4);
    int* buf = (int*)malloc(bytes);
    if (fread(buf, 4, n, f) != (size_t)n) {
        fprintf(stderr, "weights: short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    *count = n;
    return buf;
}

// ------------------------------------------------------------------ config
// config.json is a flat object of integers, so a full JSON parser would be
// three orders of magnitude more machinery than the problem deserves.

static int json_int(const char* text, const char* key)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(text, pattern);
    if (!p) {
        fprintf(stderr, "config.json: missing key \"%s\"\n", key);
        exit(1);
    }
    p = strchr(p, ':');
    if (!p) {
        fprintf(stderr, "config.json: malformed entry for \"%s\"\n", key);
        exit(1);
    }
    return atoi(p + 1);
}

ModelConfig load_config(const char* dir)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    char text[4096] = {0};
    size_t n = fread(text, 1, sizeof(text) - 1, f);
    text[n] = '\0';
    fclose(f);

    ModelConfig c;
    c.vocab_size  = json_int(text, "vocab_size");
    c.context_len = json_int(text, "context_len");
    c.n_layer     = json_int(text, "n_layer");
    c.n_head      = json_int(text, "n_head");
    c.d_model     = json_int(text, "d_model");
    c.d_ff        = json_int(text, "d_ff");
    c.head_dim    = json_int(text, "head_dim");

    if (c.head_dim * c.n_head != c.d_model) {
        fprintf(stderr, "config: head_dim(%d) * n_head(%d) != d_model(%d)\n",
                c.head_dim, c.n_head, c.d_model);
        exit(1);
    }
    return c;
}

// ------------------------------------------------------------------ weights

ModelWeights load_host_weights(const char* dir)
{
    ModelWeights m;
    m.cfg = load_config(dir);
    const ModelConfig& c = m.cfg;

    char p[1024];
    #define LOAD(field, name, count)                                  \
        snprintf(p, sizeof(p), "%s/" name, dir);                      \
        field = read_f32(p, (size_t)(count));

    LOAD(m.tok_emb,      "tok_emb.bin",     (size_t)c.vocab_size * c.d_model)
    LOAD(m.pos_emb,      "pos_emb.bin",     (size_t)c.context_len * c.d_model)
    LOAD(m.ln_f_weight,  "ln_f.weight.bin", c.d_model)
    LOAD(m.ln_f_bias,    "ln_f.bias.bin",   c.d_model)
    LOAD(m.lm_head,      "lm_head.bin",     (size_t)c.vocab_size * c.d_model)
    #undef LOAD

    m.layers = (LayerWeights*)calloc(c.n_layer, sizeof(LayerWeights));

    for (int i = 0; i < c.n_layer; i++) {
        LayerWeights& L = m.layers[i];
        #define LOADL(field, name, count)                             \
            snprintf(p, sizeof(p), "%s/layer%d." name, dir, i);        \
            field = read_f32(p, (size_t)(count));

        LOADL(L.ln1_weight, "ln1.weight.bin", c.d_model)
        LOADL(L.ln1_bias,   "ln1.bias.bin",   c.d_model)
        LOADL(L.qkv_weight, "qkv.weight.bin", (size_t)3 * c.d_model * c.d_model)
        LOADL(L.qkv_bias,   "qkv.bias.bin",   (size_t)3 * c.d_model)
        LOADL(L.out_weight, "out.weight.bin", (size_t)c.d_model * c.d_model)
        LOADL(L.out_bias,   "out.bias.bin",   c.d_model)
        LOADL(L.ln2_weight, "ln2.weight.bin", c.d_model)
        LOADL(L.ln2_bias,   "ln2.bias.bin",   c.d_model)
        LOADL(L.fc1_weight, "fc1.weight.bin", (size_t)c.d_ff * c.d_model)
        LOADL(L.fc1_bias,   "fc1.bias.bin",   c.d_ff)
        LOADL(L.fc2_weight, "fc2.weight.bin", (size_t)c.d_model * c.d_ff)
        LOADL(L.fc2_bias,   "fc2.bias.bin",   c.d_model)
        #undef LOADL
    }

    return m;
}

void free_host_weights(ModelWeights& m)
{
    free(m.tok_emb);
    free(m.pos_emb);
    free(m.ln_f_weight);
    free(m.ln_f_bias);
    free(m.lm_head);

    for (int i = 0; i < m.cfg.n_layer; i++) {
        LayerWeights& L = m.layers[i];
        free(L.ln1_weight); free(L.ln1_bias);
        free(L.qkv_weight); free(L.qkv_bias);
        free(L.out_weight); free(L.out_bias);
        free(L.ln2_weight); free(L.ln2_bias);
        free(L.fc1_weight); free(L.fc1_bias);
        free(L.fc2_weight); free(L.fc2_bias);
    }
    free(m.layers);
    m.layers = nullptr;
}
