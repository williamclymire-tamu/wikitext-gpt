// Unified kernel test driver
// Usage:
//   test_kernels attention test [data_dir]    — prefill attention correctness
//   test_kernels attention bench              — prefill attention benchmark
//   test_kernels decode    test [data_dir]    — decode + elementwise correctness
//   test_kernels decode    bench              — decode benchmark

#include "attention.cuh"
#include "elementwise.cuh"
#include "checks.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

// ── Shared helpers ───────────────────────────────────────────────────────────

static bool parse_meta_int(const char* buf, const char* key, int* out) {
    const char* p = strstr(buf, key);
    if (!p) return false;
    p = strchr(p, ':');
    *out = atoi(p + 1);
    return true;
}

static bool load_meta(const char* dir, char* buf, size_t bufsize) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.json", dir);
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    size_t n = fread(buf, 1, bufsize - 1, f);
    (void)n;
    buf[bufsize - 1] = 0;
    fclose(f);
    return true;
}

// ── Prefill attention tests ──────────────────────────────────────────────────

static int run_attention_test(const char* data_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.json", data_dir);
    FILE* mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    char meta_buf[1024];
    size_t meta_len = fread(meta_buf, 1, sizeof(meta_buf) - 1, mf);
    (void)meta_len;
    meta_buf[sizeof(meta_buf) - 1] = 0;
    fclose(mf);

    auto parse_int = [&](const char* key) -> int {
        const char* p = strstr(meta_buf, key);
        if (!p) { fprintf(stderr, "Missing key: %s\n", key); exit(1); }
        p = strchr(p, ':');
        return atoi(p + 1);
    };

    int B = parse_int("\"B\"");
    int H = parse_int("\"H\"");
    int N = parse_int("\"N\"");
    int d = parse_int("\"d\"");
    bool causal = strstr(meta_buf, "\"causal\": true") != nullptr;

    printf("Test: B=%d H=%d N=%d d=%d causal=%d  [%s]\n", B, H, N, d, causal, data_dir);

    size_t qkv_size = (size_t)B * H * N * d;
    snprintf(path, sizeof(path), "%s/Q.bin", data_dir);
    float* h_Q = load_bin(path, qkv_size);
    snprintf(path, sizeof(path), "%s/K.bin", data_dir);
    float* h_K = load_bin(path, qkv_size);
    snprintf(path, sizeof(path), "%s/V.bin", data_dir);
    float* h_V = load_bin(path, qkv_size);
    snprintf(path, sizeof(path), "%s/O_ref.bin", data_dir);
    float* h_O_ref = load_bin(path, qkv_size);

    float *d_Q, *d_K, *d_V, *d_O;
    size_t bytes = qkv_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    float* h_O_out = (float*)malloc(bytes);
    bool all_pass = true;

    CUDA_CHECK(cudaMemset(d_O, 0, bytes));
    naive_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal);
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, bytes, cudaMemcpyDeviceToHost));
    all_pass &= check_close(h_O_out, h_O_ref, qkv_size, "naive", 1e-5f, 1e-4f);

    CUDA_CHECK(cudaMemset(d_O, 0, bytes));
    fused_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, bytes, cudaMemcpyDeviceToHost));
    all_pass &= check_close(h_O_out, h_O_ref, qkv_size, "fused", 1e-5f, 1e-4f);

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return all_pass ? 0 : 1;
}

static void run_attention_bench() {
    int B = 1, H = 8, d = 64;
    bool causal = true;
    int Ns[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(Ns) / sizeof(Ns[0]);
    int warmup = 10, iters = 100;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("B=%d H=%d d=%d causal=%d  warmup=%d iters=%d\n\n",
           B, H, d, causal, warmup, iters);
    printf("%-8s  %12s  %12s  %8s\n", "N", "Naive (ms)", "Fused (ms)", "Speedup");
    printf("----------------------------------------------\n");

    for (int ni = 0; ni < num_sizes; ni++) {
        int N = Ns[ni];
        size_t qkv_size = (size_t)B * H * N * d;
        size_t bytes = qkv_size * sizeof(float);

        float *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, bytes));
        CUDA_CHECK(cudaMalloc(&d_K, bytes));
        CUDA_CHECK(cudaMalloc(&d_V, bytes));
        CUDA_CHECK(cudaMalloc(&d_O, bytes));

        float* h_tmp = (float*)malloc(bytes);
        srand(42);
        for (size_t i = 0; i < qkv_size; i++)
            h_tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        CUDA_CHECK(cudaMemcpy(d_Q, h_tmp, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_tmp, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_tmp, bytes, cudaMemcpyHostToDevice));
        free(h_tmp);

        float* naive_ws;
        CUDA_CHECK(cudaMalloc(&naive_ws, (size_t)B * H * N * N * sizeof(float)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < warmup; i++)
            naive_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal, naive_ws);
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            naive_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal, naive_ws);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float naive_ms;
        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
        naive_ms /= iters;

        for (int i = 0; i < warmup; i++)
            fused_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal);
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            fused_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float fused_ms;
        CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
        fused_ms /= iters;

        printf("%-8d  %12.4f  %12.4f  %7.2fx\n",
               N, naive_ms, fused_ms, naive_ms / fused_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(naive_ws));
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ── Decode attention tests ───────────────────────────────────────────────────

static int test_decode(const char* dir) {
    char meta[1024];
    if (!load_meta(dir, meta, sizeof(meta))) return 1;

    int B, H, d, seq_len, max_seq;
    parse_meta_int(meta, "\"B\"", &B);
    parse_meta_int(meta, "\"H\"", &H);
    parse_meta_int(meta, "\"d\"", &d);
    parse_meta_int(meta, "\"seq_len\"", &seq_len);
    parse_meta_int(meta, "\"max_seq\"", &max_seq);

    int BH = B * H;
    printf("Decode: B=%d H=%d seq_len=%d max_seq=%d d=%d\n", B, H, seq_len, max_seq, d);

    char path[512];
    snprintf(path, sizeof(path), "%s/Q.bin", dir);
    float* h_Q = load_bin(path, (size_t)BH * d);
    snprintf(path, sizeof(path), "%s/K_cache.bin", dir);
    float* h_K = load_bin(path, (size_t)BH * seq_len * d);
    snprintf(path, sizeof(path), "%s/V_cache.bin", dir);
    float* h_V = load_bin(path, (size_t)BH * seq_len * d);
    snprintf(path, sizeof(path), "%s/O_ref.bin", dir);
    float* h_O_ref = load_bin(path, (size_t)BH * d);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, (size_t)BH * d * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_K, 0, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 0, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, (size_t)BH * d * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int bh = 0; bh < BH; bh++) {
        CUDA_CHECK(cudaMemcpy(
            d_K + bh * max_seq * d, h_K + bh * seq_len * d,
            (size_t)seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_V + bh * max_seq * d, h_V + bh * seq_len * d,
            (size_t)seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(d_O, 0, (size_t)BH * d * sizeof(float)));
    fused_attention_decode_cuda(d_Q, d_K, d_V, d_O, B, H, seq_len, max_seq, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_O_out = (float*)malloc((size_t)BH * d * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, (size_t)BH * d * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool pass = check_close(h_O_out, h_O_ref, (size_t)BH * d,
                            "decode", 1e-5f, 1e-4f);

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    return pass ? 0 : 1;
}

static int test_kv_cache_append(const char* dir) {
    char meta[1024];
    if (!load_meta(dir, meta, sizeof(meta))) return 1;

    int B, H, d, seq_len, max_seq;
    parse_meta_int(meta, "\"B\"", &B);
    parse_meta_int(meta, "\"H\"", &H);
    parse_meta_int(meta, "\"d\"", &d);
    parse_meta_int(meta, "\"seq_len\"", &seq_len);
    parse_meta_int(meta, "\"max_seq\"", &max_seq);

    int BH = B * H;
    printf("KV append: B=%d H=%d seq_len=%d d=%d\n", B, H, seq_len, d);

    char path[512];
    snprintf(path, sizeof(path), "%s/Q.bin", dir);
    float* h_Q = load_bin(path, (size_t)BH * d);
    snprintf(path, sizeof(path), "%s/K_cache.bin", dir);
    float* h_K = load_bin(path, (size_t)BH * seq_len * d);
    snprintf(path, sizeof(path), "%s/V_cache.bin", dir);
    float* h_V = load_bin(path, (size_t)BH * seq_len * d);
    snprintf(path, sizeof(path), "%s/O_ref.bin", dir);
    float* h_O_ref = load_bin(path, (size_t)BH * d);

    float *d_K_cache, *d_V_cache, *d_K_row, *d_V_row, *d_Q, *d_O;
    CUDA_CHECK(cudaMalloc(&d_K_cache, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_cache, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_row, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_row, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, (size_t)BH * d * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_K_cache, 0, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V_cache, 0, (size_t)BH * max_seq * d * sizeof(float)));

    for (int pos = 0; pos < seq_len; pos++) {
        for (int bh = 0; bh < BH; bh++) {
            CUDA_CHECK(cudaMemcpy(
                d_K_row + bh * d, h_K + (bh * seq_len + pos) * d,
                d * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_V_row + bh * d, h_V + (bh * seq_len + pos) * d,
                d * sizeof(float), cudaMemcpyHostToDevice));
        }
        kv_cache_append_cuda(d_K_cache, d_V_cache, d_K_row, d_V_row,
                             B, H, pos, max_seq, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, (size_t)BH * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, (size_t)BH * d * sizeof(float)));

    fused_attention_decode_cuda(d_Q, d_K_cache, d_V_cache, d_O,
                                B, H, seq_len, max_seq, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_O_out = (float*)malloc((size_t)BH * d * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, (size_t)BH * d * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool pass = check_close(h_O_out, h_O_ref, (size_t)BH * d,
                            "append+decode", 1e-5f, 1e-4f);

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_K_cache)); CUDA_CHECK(cudaFree(d_V_cache));
    CUDA_CHECK(cudaFree(d_K_row)); CUDA_CHECK(cudaFree(d_V_row));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_O));
    return pass ? 0 : 1;
}

// ── Elementwise kernel tests ─────────────────────────────────────────────────

static int test_layernorm(const char* dir) {
    char meta[1024];
    if (!load_meta(dir, meta, sizeof(meta))) return 1;

    int N, d;
    parse_meta_int(meta, "\"N\"", &N);
    parse_meta_int(meta, "\"d\"", &d);
    printf("LayerNorm: N=%d d=%d\n", N, d);

    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", dir);
    float* h_input = load_bin(path, (size_t)N * d);
    snprintf(path, sizeof(path), "%s/weight.bin", dir);
    float* h_weight = load_bin(path, d);
    snprintf(path, sizeof(path), "%s/bias.bin", dir);
    float* h_bias = load_bin(path, d);
    snprintf(path, sizeof(path), "%s/output_ref.bin", dir);
    float* h_ref = load_bin(path, (size_t)N * d);

    float *d_input, *d_weight, *d_bias, *d_output;
    size_t bytes = (size_t)N * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, d * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_cuda(d_input, d_weight, d_bias, d_output, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_output, bytes, cudaMemcpyDeviceToHost));

    bool pass = check_close(h_out, h_ref, (size_t)N * d, "layernorm", 1e-5f, 1e-3f);

    free(h_input); free(h_weight); free(h_bias); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias)); CUDA_CHECK(cudaFree(d_output));
    return pass ? 0 : 1;
}

static int test_gelu(const char* dir) {
    char meta[1024];
    if (!load_meta(dir, meta, sizeof(meta))) return 1;

    int n;
    parse_meta_int(meta, "\"n\"", &n);
    printf("GELU: n=%d\n", n);

    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", dir);
    float* h_input = load_bin(path, n);
    snprintf(path, sizeof(path), "%s/output_ref.bin", dir);
    float* h_ref = load_bin(path, n);

    float *d_input, *d_output;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    gelu_cuda(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_output, bytes, cudaMemcpyDeviceToHost));

    bool pass = check_close(h_out, h_ref, n, "gelu", 1e-5f, 1e-4f);

    free(h_input); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    return pass ? 0 : 1;
}

static int test_residual(const char* dir) {
    char meta[1024];
    if (!load_meta(dir, meta, sizeof(meta))) return 1;

    int n;
    parse_meta_int(meta, "\"n\"", &n);
    printf("Residual: n=%d\n", n);

    char path[512];
    snprintf(path, sizeof(path), "%s/a.bin", dir);
    float* h_a = load_bin(path, n);
    snprintf(path, sizeof(path), "%s/b.bin", dir);
    float* h_b = load_bin(path, n);
    snprintf(path, sizeof(path), "%s/output_ref.bin", dir);
    float* h_ref = load_bin(path, n);

    float *d_a, *d_b, *d_out;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    residual_add_cuda(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool pass = check_close(h_out, h_ref, n, "residual", 1e-5f, 1e-6f);

    free(h_a); free(h_b); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    return pass ? 0 : 1;
}

// ── Decode benchmark ─────────────────────────────────────────────────────────

static void bench_decode() {
    int B = 1, H = 8, d = 64;
    int max_seq = 2048 + 64;
    int warmup = 10, iters = 100;
    int seq_lens[] = {128, 256, 512, 1024, 2048};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nDecode benchmark on %s\n", prop.name);
    printf("B=%d H=%d d=%d  warmup=%d iters=%d\n\n", B, H, d, warmup, iters);
    printf("%-10s  %12s\n", "seq_len", "Decode (ms)");
    printf("-------------------------\n");

    int BH = B * H;
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, (size_t)BH * d * sizeof(float)));

    float* h_tmp = (float*)malloc((size_t)BH * max_seq * d * sizeof(float));
    srand(42);
    for (size_t i = 0; i < (size_t)BH * max_seq * d; i++)
        h_tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    CUDA_CHECK(cudaMemcpy(d_K, h_tmp, (size_t)BH * max_seq * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_tmp, (size_t)BH * max_seq * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    for (size_t i = 0; i < (size_t)BH * d; i++)
        h_tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    CUDA_CHECK(cudaMemcpy(d_Q, h_tmp, (size_t)BH * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_tmp);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int si = 0; si < 5; si++) {
        int seq_len = seq_lens[si];
        for (int i = 0; i < warmup; i++)
            fused_attention_decode_cuda(d_Q, d_K, d_V, d_O, B, H, seq_len, max_seq, d);
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            fused_attention_decode_cuda(d_Q, d_K, d_V, d_O, B, H, seq_len, max_seq, d);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iters;
        printf("%-10d  %12.4f\n", seq_len, ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
}

// ── Main ─────────────────────────────────────────────────────────────────────

static int run_attention_suite(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s attention test [dir] | bench\n", argv[0]); return 1; }

    if (strcmp(argv[2], "test") == 0) {
        const char* dir = (argc > 3) ? argv[3] : "test_data";
        int rc = run_attention_test(dir);
        if (rc == 0) {
            int extra_ns[] = {37, 127, 200};
            for (int i = 0; i < 3; i++) {
                char subdir[256];
                snprintf(subdir, sizeof(subdir), "%s/N%d", dir, extra_ns[i]);
                char meta_path[300];
                snprintf(meta_path, sizeof(meta_path), "%s/meta.json", subdir);
                FILE* f = fopen(meta_path, "r");
                if (f) { fclose(f); rc |= run_attention_test(subdir); }
            }
        }
        printf("\n%s\n", rc == 0 ? "ALL ATTENTION TESTS PASSED" : "SOME TESTS FAILED");
        return rc;
    } else if (strcmp(argv[2], "bench") == 0) {
        run_attention_bench();
        return 0;
    }
    fprintf(stderr, "Unknown subcommand: %s\n", argv[2]);
    return 1;
}

static int run_decode_suite(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s decode test [dir] | bench\n", argv[0]); return 1; }

    if (strcmp(argv[2], "test") == 0) {
        const char* dir = (argc > 3) ? argv[3] : "test_decode_data";
        int rc = 0;

        printf("\n=== Decode Attention Tests ===\n");
        { char s[256]; snprintf(s, sizeof(s), "%s/decode_B1_H4_S128", dir); rc |= test_decode(s); }
        { char s[256]; snprintf(s, sizeof(s), "%s/decode_B1_H4_S37",  dir); rc |= test_decode(s); }
        { char s[256]; snprintf(s, sizeof(s), "%s/decode_B2_H4_S256", dir); rc |= test_decode(s); }

        printf("\n=== KV Cache Append Tests ===\n");
        { char s[256]; snprintf(s, sizeof(s), "%s/decode_B1_H4_S128", dir); rc |= test_kv_cache_append(s); }

        printf("\n=== LayerNorm Tests ===\n");
        { char s[256]; snprintf(s, sizeof(s), "%s/layernorm_N128_d256", dir); rc |= test_layernorm(s); }
        { char s[256]; snprintf(s, sizeof(s), "%s/layernorm_N37_d256",  dir); rc |= test_layernorm(s); }
        { char s[256]; snprintf(s, sizeof(s), "%s/layernorm_N128_d64",  dir); rc |= test_layernorm(s); }

        printf("\n=== GELU Tests ===\n");
        { char s[256]; snprintf(s, sizeof(s), "%s/gelu_n32768", dir); rc |= test_gelu(s); }
        { char s[256]; snprintf(s, sizeof(s), "%s/gelu_n1000",  dir); rc |= test_gelu(s); }

        printf("\n=== Residual Add Tests ===\n");
        { char s[256]; snprintf(s, sizeof(s), "%s/residual_n32768", dir); rc |= test_residual(s); }

        printf("\n%s\n", rc == 0 ? "ALL DECODE TESTS PASSED" : "SOME TESTS FAILED");
        return rc;
    } else if (strcmp(argv[2], "bench") == 0) {
        bench_decode();
        return 0;
    }
    fprintf(stderr, "Unknown subcommand: %s\n", argv[2]);
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <attention|decode> <test|bench> [data_dir]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "attention") == 0) return run_attention_suite(argc, argv);
    if (strcmp(argv[1], "decode")    == 0) return run_decode_suite(argc, argv);

    fprintf(stderr, "Unknown suite: %s\n", argv[1]);
    return 1;
}
