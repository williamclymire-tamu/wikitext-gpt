#include "attention.cuh"
#include <cstdlib>
#include <cstring>
#include <cstdio>

int run_test(const char* data_dir) {
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
    all_pass &= check(h_O_out, h_O_ref, qkv_size, 1e-4f, "naive vs reference");

    CUDA_CHECK(cudaMemset(d_O, 0, bytes));
    fused_attention_cuda(d_Q, d_K, d_V, d_O, B, H, N, d, causal);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_out, d_O, bytes, cudaMemcpyDeviceToHost));
    all_pass &= check(h_O_out, h_O_ref, qkv_size, 1e-4f, "fused vs reference");

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_out);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return all_pass ? 0 : 1;
}

void run_bench() {
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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s test [dir] | bench\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "test") == 0) {
        const char* dir = (argc > 2) ? argv[2] : "test_data";
        int rc = run_test(dir);
        if (rc == 0) {
            int extra_ns[] = {37, 127, 200};
            for (int i = 0; i < 3; i++) {
                char subdir[256];
                snprintf(subdir, sizeof(subdir), "%s/N%d", dir, extra_ns[i]);
                char meta_path[300];
                snprintf(meta_path, sizeof(meta_path), "%s/meta.json", subdir);
                FILE* f = fopen(meta_path, "r");
                if (f) { fclose(f); rc |= run_test(subdir); }
            }
        }
        printf("\n%s\n", rc == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
        return rc;
    } else if (strcmp(argv[1], "bench") == 0) {
        run_bench();
        return 0;
    }
    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}
