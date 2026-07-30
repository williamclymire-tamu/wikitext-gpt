// Engine driver.
//
//   ./engine parity   <export_dir>
//   ./engine generate <export_dir> --ids 12,45,9 --tokens 100 --temp 0.8 --top-k 40
//   ./engine bench    <export_dir> --tokens 256
//
// parity is the one that matters during bring-up: it runs the same prompt the
// exporter ran and diffs every stage against PyTorch. Everything else is only
// worth looking at once parity passes.

#include "transformer.cuh"
#include "weights_host.h"
#include "detokenizer.h"
#include "checks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ------------------------------------------------------------------ helpers

static const char* arg_str(int argc, char** argv, const char* flag, const char* def)
{
    for (int i = 0; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

static int arg_int(int argc, char** argv, const char* flag, int def)
{
    const char* s = arg_str(argc, argv, flag, nullptr);
    return s ? atoi(s) : def;
}

static float arg_float(int argc, char** argv, const char* flag, float def)
{
    const char* s = arg_str(argc, argv, flag, nullptr);
    return s ? (float)atof(s) : def;
}

static std::vector<int> parse_ids(const char* csv)
{
    std::vector<int> out;
    if (!csv) return out;
    const char* p = csv;
    while (*p) {
        out.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return out;
}

// ------------------------------------------------------------------ parity

static int cmd_parity(const char* dir)
{
    ModelWeights m = load_model_weights(dir);
    const ModelConfig& c = m.cfg;

    printf("engine parity vs %s\n", dir);
    printf("  vocab=%d ctx=%d layers=%d heads=%d d_model=%d head_dim=%d\n\n",
           c.vocab_size, c.context_len, c.n_layer, c.n_head, c.d_model, c.head_dim);

    if (c.head_dim != HEAD_DIM) {
        fprintf(stderr, "  ! built with HEAD_DIM=%d but the model has head_dim=%d.\n"
                        "    Rebuild: make HEAD_DIM=%d\n",
                HEAD_DIM, c.head_dim, c.head_dim);
        return 1;
    }

    char path[1024];
    int T = 0;
    snprintf(path, sizeof(path), "%s/parity/prompt.bin", dir);
    int* tokens = read_i32(path, &T);

    RunState s = runstate_alloc(c, c.context_len, T);

    // host landing buffers for the per-stage taps
    EngineTrace tr;
    tr.embed     = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);
    tr.final_ln  = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);
    tr.logits    = (float*)malloc(sizeof(float) * (size_t)T * c.vocab_size);
    tr.layer_out = (float**)malloc(sizeof(float*) * c.n_layer);
    for (int i = 0; i < c.n_layer; i++)
        tr.layer_out[i] = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);

    engine_prefill(m, s, tokens, T, &tr);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    size_t act = (size_t)T * c.d_model;

    printf("[1] CUDA prefill vs exported fixtures\n");
    snprintf(path, sizeof(path), "%s/parity/embed.bin", dir);
    { float* w = read_f32(path, act); ok &= check_close(tr.embed, w, act, "embed"); free(w); }

    for (int i = 0; i < c.n_layer; i++) {
        snprintf(path, sizeof(path), "%s/parity/layer%d.bin", dir, i);
        float* w = read_f32(path, act);
        char label[32];
        snprintf(label, sizeof(label), "layer%d", i);
        // looser than the CPU reference: --use_fast_math changes expf/tanhf, and
        // cuBLAS reorders accumulation. Drift here is expected; growth across
        // layers is what would indicate a real bug.
        ok &= check_close(tr.layer_out[i], w, act, label, 1e-4f, 1e-3f);
        free(w);
    }

    snprintf(path, sizeof(path), "%s/parity/final.bin", dir);
    { float* w = read_f32(path, act); ok &= check_close(tr.final_ln, w, act, "final_ln", 1e-4f, 1e-3f); free(w); }

    snprintf(path, sizeof(path), "%s/parity/logits.bin", dir);
    {
        size_t n = (size_t)T * c.vocab_size;
        float* w = read_f32(path, n);
        ok &= check_close(tr.logits, w, n, "logits", 1e-3f, 1e-2f);
        ok &= check_argmax(tr.logits, w, T, c.vocab_size, "logits");
        free(w);
    }

    printf("\n[2] CUDA decode (KV cache) vs CUDA prefill\n");
    {
        runstate_reset(s);
        std::vector<float> host_logits(c.vocab_size);
        int bad = 0;
        float worst = 0.0f;

        for (int t = 0; t < T; t++) {
            float* d_logits = engine_decode(m, s, tokens[t]);
            CUDA_CHECK(cudaMemcpy(host_logits.data(), d_logits,
                                  c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

            const float* want = tr.logits + (size_t)t * c.vocab_size;
            float row = 0.0f;
            for (int i = 0; i < c.vocab_size; i++) {
                float diff = fabsf(host_logits[i] - want[i]);
                if (diff > row) row = diff;
            }
            if (row > worst) worst = row;
            if (row > 1e-2f) bad++;
        }
        bool pass = (bad == 0);
        printf("  %-16s max_abs=%.3e  bad_rows=%d/%d  [%s]\n",
               "decode==prefill", worst, bad, T, pass ? "PASS" : "FAIL");
        ok &= pass;
    }

    printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");

    free(tr.embed); free(tr.final_ln); free(tr.logits);
    for (int i = 0; i < c.n_layer; i++) free(tr.layer_out[i]);
    free(tr.layer_out);
    free(tokens);
    runstate_free(s);
    free_model_weights(m);
    return ok ? 0 : 1;
}

// ------------------------------------------------------------------ generate

static int cmd_generate(int argc, char** argv, const char* dir)
{
    ModelWeights m = load_model_weights(dir);
    const ModelConfig& c = m.cfg;

    std::vector<int> prompt = parse_ids(arg_str(argc, argv, "--ids", nullptr));
    if (prompt.empty()) prompt.push_back(0);

    int max_new = arg_int(argc, argv, "--tokens", 64);
    float temp  = arg_float(argc, argv, "--temp", 0.8f);
    int top_k   = arg_int(argc, argv, "--top-k", 40);
    int eos     = arg_int(argc, argv, "--eos", -1);
    unsigned long long rng = (unsigned long long)arg_int(argc, argv, "--seed", 1234) + 88172645463325252ULL;

    if ((int)prompt.size() + max_new > c.context_len) {
        max_new = c.context_len - (int)prompt.size();
        fprintf(stderr, "note: clamped to %d new tokens (context_len=%d)\n",
                max_new, c.context_len);
    }

    Detokenizer dt = detok_load(dir, c.vocab_size);
    RunState s = runstate_alloc(c, c.context_len, (int)prompt.size());

    float* d_logits = engine_prefill(m, s, prompt.data(), (int)prompt.size());
    // logits for the *last* prompt position are what condition the first sample
    d_logits += (size_t)(prompt.size() - 1) * c.vocab_size;

    for (int i = 0; i < (int)prompt.size(); i++) detok_emit(dt, prompt[i], stdout);

    for (int i = 0; i < max_new; i++) {
        int next = sample_token(s, d_logits, temp, top_k, &rng);
        if (next == eos) break;
        detok_emit(dt, next, stdout);
        fflush(stdout);
        d_logits = engine_decode(m, s, next);
    }
    printf("\n");

    detok_free(dt);
    runstate_free(s);
    free_model_weights(m);
    return 0;
}

// ------------------------------------------------------------------ bench

static int cmd_bench(int argc, char** argv, const char* dir)
{
    ModelWeights m = load_model_weights(dir);
    const ModelConfig& c = m.cfg;

    int n_tokens = arg_int(argc, argv, "--tokens", 128);
    if (n_tokens > c.context_len - 1) n_tokens = c.context_len - 1;

    RunState s = runstate_alloc(c, c.context_len, 1);
    int bos = 0;

    // warmup: first launches pay cuBLAS handle setup and kernel load
    engine_prefill(m, s, &bos, 1);
    for (int i = 0; i < 16; i++) engine_decode(m, s, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    runstate_reset(s);
    engine_prefill(m, s, &bos, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < n_tokens; i++) engine_decode(m, s, 0);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    printf("decode: %d tokens in %.2f ms  ->  %.3f ms/token, %.1f tok/s\n",
           n_tokens, ms, ms / n_tokens, 1000.0f * n_tokens / ms);
    printf("  (batch 1, ctx grows 1..%d, sampling excluded)\n", n_tokens);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    runstate_free(s);
    free_model_weights(m);
    return 0;
}

// ------------------------------------------------------------------ main

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  %s parity   <export_dir>\n"
            "  %s generate <export_dir> [--ids 1,2,3] [--tokens N] [--temp F] [--top-k K] [--seed S] [--eos ID]\n"
            "  %s bench    <export_dir> [--tokens N]\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    const char* dir  = argv[2];

    cublas_init();
    int rc;
    if      (strcmp(mode, "parity")   == 0) rc = cmd_parity(dir);
    else if (strcmp(mode, "generate") == 0) rc = cmd_generate(argc, argv, dir);
    else if (strcmp(mode, "bench")    == 0) rc = cmd_bench(argc, argv, dir);
    else { fprintf(stderr, "unknown mode: %s\n", mode); rc = 1; }
    cublas_destroy();

    return rc;
}
