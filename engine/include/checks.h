#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>

// ── Binary tensor I/O (used by kernel tests) ────────────────────────────────
// Inline so the kernel test targets don't need to link weights_host.cpp.

inline float* load_bin(const char* path, size_t num_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* buf = (float*)malloc(num_floats * sizeof(float));
    size_t nread = fread(buf, sizeof(float), num_floats, f);
    if (nread != num_floats) {
        fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, num_floats, nread);
        exit(1);
    }
    fclose(f);
    return buf;
}

// numpy-style allclose: |a-b| <= atol + rtol*|b|
// CUDA-free so both the CPU harness and the GPU harness can share one verdict.
inline bool check_close(const float* got, const float* want, size_t n,
                        const char* label,
                        float atol = 1e-5f, float rtol = 1e-4f)
{
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t worst = 0;
    size_t num_fail = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(got[i] - want[i]);
        if (diff > max_abs) { max_abs = diff; worst = i; }
        float denom = fmaxf(fabsf(want[i]), 1e-8f);
        float rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        if (diff > atol + rtol * fabsf(want[i])) num_fail++;
    }

    bool pass = (num_fail == 0);
    printf("  %-16s max_abs=%.3e  max_rel=%.3e  fail=%zu/%zu  [%s]\n",
           label, max_abs, max_rel, num_fail, n, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("      worst i=%zu: got %.6f want %.6f\n",
               worst, got[worst], want[worst]);
    }
    return pass;
}

// Logits can drift more than activations (one extra V-wide GEMM, and --use_fast_math
// changes expf/tanhf), so what actually matters downstream is whether sampling
// would pick the same token. Checked separately from raw closeness.
inline bool check_argmax(const float* got, const float* want, int rows, int cols,
                         const char* label)
{
    int mismatch = 0;
    for (int r = 0; r < rows; r++) {
        int ag = 0, aw = 0;
        for (int c = 1; c < cols; c++) {
            if (got[r * cols + c]  > got[r * cols + ag])  ag = c;
            if (want[r * cols + c] > want[r * cols + aw]) aw = c;
        }
        if (ag != aw) mismatch++;
    }
    bool pass = (mismatch == 0);
    printf("  %-16s argmax mismatch=%d/%d  [%s]\n",
           label, mismatch, rows, pass ? "PASS" : "FAIL");
    return pass;
}
