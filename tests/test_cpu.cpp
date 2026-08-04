// CPU-only test harness. Builds with plain g++ -- no CUDA toolkit, no GPU.
//
//   make test_cpu
//   ./test_cpu ../wikitext-gpt/export
//
// Two independent things get checked:
//
//   1. reference vs exported fixtures
//      Confirms the scalar C++ forward agrees with what PyTorch (or the NumPy
//      generator) actually computed, stage by stage. Establishes the oracle.
//
//   2. incremental decode vs full prefill
//      Feeding tokens one at a time through the KV cache must reproduce the
//      corresponding rows of a full prefill. This catches cache indexing and
//      position-embedding bugs on CPU, where they are cheap to find, instead of
//      inside a CUDA kernel where they are not.

#include "weights_host.h"
#include "reference.h"
#include "checks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    const char* dir = (argc > 1) ? argv[1] : "export";

    ModelWeights m = load_host_weights(dir);
    const ModelConfig& c = m.cfg;

    printf("loaded %s\n", dir);
    printf("  vocab=%d ctx=%d layers=%d heads=%d d_model=%d d_ff=%d head_dim=%d\n\n",
           c.vocab_size, c.context_len, c.n_layer, c.n_head,
           c.d_model, c.d_ff, c.head_dim);

    char path[1024];
    int T = 0;
    snprintf(path, sizeof(path), "%s/parity/prompt.bin", dir);
    int* tokens = read_i32(path, &T);
    printf("prompt_len = %d\n\n", T);

    RefTrace trace = ref_trace_alloc(c, T);
    ref_forward(m, tokens, T, trace);

    bool ok = true;

    printf("[1] scalar reference vs exported fixtures\n");
    {
        snprintf(path, sizeof(path), "%s/parity/embed.bin", dir);
        float* want = read_f32(path, (size_t)T * c.d_model);
        ok &= check_close(trace.embed, want, (size_t)T * c.d_model, "embed");
        free(want);

        for (int i = 0; i < c.n_layer; i++) {
            snprintf(path, sizeof(path), "%s/parity/layer%d.bin", dir, i);
            float* w = read_f32(path, (size_t)T * c.d_model);
            char label[32];
            snprintf(label, sizeof(label), "layer%d", i);
            ok &= check_close(trace.layer_out[i], w, (size_t)T * c.d_model, label);
            free(w);
        }

        snprintf(path, sizeof(path), "%s/parity/final.bin", dir);
        float* wf = read_f32(path, (size_t)T * c.d_model);
        ok &= check_close(trace.final_ln, wf, (size_t)T * c.d_model, "final_ln");
        free(wf);

        snprintf(path, sizeof(path), "%s/parity/logits.bin", dir);
        float* wl = read_f32(path, (size_t)T * c.vocab_size);
        ok &= check_close(trace.logits, wl, (size_t)T * c.vocab_size, "logits",
                          1e-4f, 1e-3f);
        ok &= check_argmax(trace.logits, wl, T, c.vocab_size, "logits");
        free(wl);
    }

    printf("\n[2] incremental decode (KV cache) vs full prefill\n");
    {
        RefState s = ref_state_alloc(c, c.context_len);
        float* logits = (float*)malloc(sizeof(float) * c.vocab_size);

        int bad_rows = 0;
        float worst = 0.0f;

        for (int t = 0; t < T; t++) {
            ref_decode_step(m, s, tokens[t], logits);

            const float* want = trace.logits + (size_t)t * c.vocab_size;
            float row_worst = 0.0f;
            for (int i = 0; i < c.vocab_size; i++) {
                float diff = fabsf(logits[i] - want[i]);
                if (diff > row_worst) row_worst = diff;
            }
            if (row_worst > worst) worst = row_worst;
            if (row_worst > 1e-4f) bad_rows++;
        }

        bool pass = (bad_rows == 0);
        printf("  %-16s max_abs=%.3e  bad_rows=%d/%d  [%s]\n",
               "decode==prefill", worst, bad_rows, T, pass ? "PASS" : "FAIL");
        ok &= pass;

        free(logits);
        ref_state_free(s);
    }

    printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");

    ref_trace_free(trace, c);
    free(tokens);
    free_host_weights(m);
    return ok ? 0 : 1;
}
