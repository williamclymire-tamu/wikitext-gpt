#include "attention.cuh"
#include "elementwise.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdio>

// helpers

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

// decode attention test

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

    // Allocate device memory — cache is max_seq-sized (larger than seq_len)
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, (size_t)BH * d * sizeof(float)));

    // Zero the cache (max_seq > seq_len, so there's padding)
    CUDA_CHECK(cudaMemset(d_K, 0, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 0, (size_t)BH * max_seq * d * sizeof(float)));

    // Copy Q directly
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, (size_t)BH * d * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Copy K,V cache — need to handle stride (max_seq vs seq_len)
    // K_cache on host is [BH, seq_len, d] contiguous
    // K_cache on device is [BH, max_seq, d] — copy each head's slice
    for (int bh = 0; bh < BH; bh++) {
        CUDA_CHECK(cudaMemcpy(
            d_K + bh * max_seq * d,
            h_K + bh * seq_len * d,
            (size_t)seq_len * d * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_V + bh * max_seq * d,
            h_V + bh * seq_len * d,
            (size_t)seq_len * d * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(d_O, 0, (size_t)BH * d * sizeof(float)));

    fused_attention_decode_cuda(d_Q, d_K, d_V, d_O, B, H, seq_len, max_seq, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_O_out = (float*)malloc((size_t)BH * d * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, (size_t)BH * d * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool pass = check(h_O_out, h_O_ref, (size_t)BH * d, 1e-4f,
                      "decode vs reference");

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    return pass ? 0 : 1;
}

// build cache one row at a time, then decode -- should match pre-filled result

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

    // Allocate cache and fill it one row at a time via kv_cache_append
    float *d_K_cache, *d_V_cache, *d_K_row, *d_V_row, *d_Q, *d_O;
    CUDA_CHECK(cudaMalloc(&d_K_cache, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_cache, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_row, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_row, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, (size_t)BH * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, (size_t)BH * d * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_K_cache, 0, (size_t)BH * max_seq * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V_cache, 0, (size_t)BH * max_seq * d * sizeof(float)));

    // Append rows one at a time
    for (int pos = 0; pos < seq_len; pos++) {
        // Upload row `pos` for each head
        for (int bh = 0; bh < BH; bh++) {
            CUDA_CHECK(cudaMemcpy(
                d_K_row + bh * d,
                h_K + (bh * seq_len + pos) * d,
                d * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_V_row + bh * d,
                h_V + (bh * seq_len + pos) * d,
                d * sizeof(float), cudaMemcpyHostToDevice));
        }
        kv_cache_append_cuda(d_K_cache, d_V_cache, d_K_row, d_V_row,
                             B, H, pos, max_seq, d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now decode using the incrementally-built cache
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, (size_t)BH * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, (size_t)BH * d * sizeof(float)));

    fused_attention_decode_cuda(d_Q, d_K_cache, d_V_cache, d_O,
                                B, H, seq_len, max_seq, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_O_out = (float*)malloc((size_t)BH * d * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, (size_t)BH * d * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool pass = check(h_O_out, h_O_ref, (size_t)BH * d, 1e-4f,
                      "append+decode vs reference");

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_K_cache)); CUDA_CHECK(cudaFree(d_V_cache));
    CUDA_CHECK(cudaFree(d_K_row)); CUDA_CHECK(cudaFree(d_V_row));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_O));
    return pass ? 0 : 1;
}

// layernorm test

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

    bool pass = check(h_out, h_ref, (size_t)N * d, 1e-3f, "layernorm vs reference");

    free(h_input); free(h_weight); free(h_bias); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias)); CUDA_CHECK(cudaFree(d_output));
    return pass ? 0 : 1;
}

// gelu test

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

    bool pass = check(h_out, h_ref, n, 1e-4f, "gelu vs reference");

    free(h_input); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    return pass ? 0 : 1;
}

// residual add test

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

    bool pass = check(h_out, h_ref, n, 1e-6f, "residual vs reference");

    free(h_a); free(h_b); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    return pass ? 0 : 1;
}

// decode benchmark

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

    // Fill with random data
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

// main

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s test [data_dir] | bench\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "test") == 0) {
        const char* dir = (argc > 2) ? argv[2] : "test_day1";
        int rc = 0;

        printf("\n=== Decode Attention Tests ===\n");
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/decode_B1_H4_S128", dir);
            rc |= test_decode(sub);
        }
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/decode_B1_H4_S37", dir);
            rc |= test_decode(sub);
        }
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/decode_B2_H4_S256", dir);
            rc |= test_decode(sub);
        }

        printf("\n=== KV Cache Append Tests ===\n");
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/decode_B1_H4_S128", dir);
            rc |= test_kv_cache_append(sub);
        }

        printf("\n=== LayerNorm Tests ===\n");
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/layernorm_N128_d256", dir);
            rc |= test_layernorm(sub);
        }
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/layernorm_N37_d256", dir);
            rc |= test_layernorm(sub);
        }
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/layernorm_N128_d64", dir);
            rc |= test_layernorm(sub);
        }

        printf("\n=== GELU Tests ===\n");
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/gelu_n32768", dir);
            rc |= test_gelu(sub);
        }
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/gelu_n1000", dir);
            rc |= test_gelu(sub);
        }

        printf("\n=== Residual Add Tests ===\n");
        {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s/residual_n32768", dir);
            rc |= test_residual(sub);
        }

        printf("\n%s\n", rc == 0 ? "ALL DAY 1 TESTS PASSED" : "SOME TESTS FAILED");
        return rc;

    } else if (strcmp(argv[1], "bench") == 0) {
        bench_decode();
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
