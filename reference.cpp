#include "reference.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

// ------------------------------------------------------------------ scalar ops
// Double accumulation throughout. The reference is allowed to be more accurate
// than the thing it is checking; it is not allowed to be less.

static void ln_rows(const float* x, const float* w, const float* b,
                    float* y, int N, int d, float eps = 1e-5f)
{
    for (int r = 0; r < N; r++) {
        const float* xr = x + (size_t)r * d;
        float* yr = y + (size_t)r * d;

        double mean = 0.0;
        for (int i = 0; i < d; i++) mean += xr[i];
        mean /= d;

        double var = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = xr[i] - mean;
            var += diff * diff;
        }
        var /= d;   // biased, matching torch.nn.LayerNorm

        double inv = 1.0 / sqrt(var + eps);
        for (int i = 0; i < d; i++)
            yr[i] = (float)((xr[i] - mean) * inv * w[i] + b[i]);
    }
}

// Y[M,N] = X[M,K] @ W^T + bias,  W stored [N,K] (nn.Linear layout).
// Same contract as linear_cuda, so a disagreement here is a real disagreement.
static void linear_rows(const float* X, const float* W, const float* bias,
                        float* Y, int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        const float* xm = X + (size_t)m * K;
        float* ym = Y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float* wn = W + (size_t)n * K;
            double acc = bias ? bias[n] : 0.0;
            for (int k = 0; k < K; k++) acc += (double)xm[k] * wn[k];
            ym[n] = (float)acc;
        }
    }
}

static void gelu_inplace(float* x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        x[i] = (float)(0.5 * v * (1.0 + tanh(inner)));
    }
}

// ------------------------------------------------------------------ trace

RefTrace ref_trace_alloc(const ModelConfig& c, int T)
{
    RefTrace t;
    t.T = T;
    t.embed     = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);
    t.final_ln  = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);
    t.logits    = (float*)malloc(sizeof(float) * (size_t)T * c.vocab_size);
    t.layer_out = (float**)malloc(sizeof(float*) * c.n_layer);
    for (int i = 0; i < c.n_layer; i++)
        t.layer_out[i] = (float*)malloc(sizeof(float) * (size_t)T * c.d_model);
    return t;
}

void ref_trace_free(RefTrace& t, const ModelConfig& c)
{
    free(t.embed);
    free(t.final_ln);
    free(t.logits);
    for (int i = 0; i < c.n_layer; i++) free(t.layer_out[i]);
    free(t.layer_out);
}

// ------------------------------------------------------------------ prefill

void ref_forward(const ModelWeights& m, const int* tokens, int T, RefTrace& trace)
{
    const ModelConfig& c = m.cfg;
    const int d  = c.d_model;
    const int H  = c.n_head;
    const int hd = c.head_dim;
    const int dff = c.d_ff;

    float* x    = (float*)malloc(sizeof(float) * (size_t)T * d);
    float* h    = (float*)malloc(sizeof(float) * (size_t)T * d);
    float* qkv  = (float*)malloc(sizeof(float) * (size_t)T * 3 * d);
    float* q    = (float*)malloc(sizeof(float) * (size_t)H * T * hd);
    float* k    = (float*)malloc(sizeof(float) * (size_t)H * T * hd);
    float* v    = (float*)malloc(sizeof(float) * (size_t)H * T * hd);
    float* oh   = (float*)malloc(sizeof(float) * (size_t)H * T * hd);
    float* omg  = (float*)malloc(sizeof(float) * (size_t)T * d);
    float* proj = (float*)malloc(sizeof(float) * (size_t)T * d);
    float* ff   = (float*)malloc(sizeof(float) * (size_t)T * dff);
    float* scores = (float*)malloc(sizeof(float) * T);

    // embed
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            x[(size_t)t * d + i] = m.tok_emb[(size_t)tokens[t] * d + i]
                                 + m.pos_emb[(size_t)t * d + i];
    memcpy(trace.embed, x, sizeof(float) * (size_t)T * d);

    for (int l = 0; l < c.n_layer; l++) {
        const LayerWeights& L = m.layers[l];

        ln_rows(x, L.ln1_weight, L.ln1_bias, h, T, d);
        linear_rows(h, L.qkv_weight, L.qkv_bias, qkv, T, 3 * d, d);

        // [T, 3d] -> three [H, T, hd] views. This split is the single most
        // error-prone step in the whole engine, so it is written out longhand.
        for (int t = 0; t < T; t++) {
            for (int hh = 0; hh < H; hh++) {
                for (int j = 0; j < hd; j++) {
                    size_t src = (size_t)t * 3 * d + (size_t)hh * hd + j;
                    size_t dst = ((size_t)hh * T + t) * hd + j;
                    q[dst] = qkv[src];
                    k[dst] = qkv[src + d];
                    v[dst] = qkv[src + 2 * d];
                }
            }
        }

        const double scale = 1.0 / sqrt((double)hd);

        for (int hh = 0; hh < H; hh++) {
            const float* qh = q + (size_t)hh * T * hd;
            const float* kh = k + (size_t)hh * T * hd;
            const float* vh = v + (size_t)hh * T * hd;
            float* outh = oh + (size_t)hh * T * hd;

            for (int i = 0; i < T; i++) {
                // causal: attend to j <= i only
                double row_max = -INFINITY;
                for (int j = 0; j <= i; j++) {
                    double dot = 0.0;
                    for (int e = 0; e < hd; e++)
                        dot += (double)qh[(size_t)i * hd + e] * kh[(size_t)j * hd + e];
                    double s = dot * scale;
                    scores[j] = (float)s;
                    if (s > row_max) row_max = s;
                }

                double sum = 0.0;
                for (int j = 0; j <= i; j++) {
                    double e = exp((double)scores[j] - row_max);
                    scores[j] = (float)e;
                    sum += e;
                }

                for (int e = 0; e < hd; e++) {
                    double acc = 0.0;
                    for (int j = 0; j <= i; j++)
                        acc += (double)scores[j] * vh[(size_t)j * hd + e];
                    outh[(size_t)i * hd + e] = (float)(acc / sum);
                }
            }
        }

        // [H, T, hd] -> [T, d]
        for (int t = 0; t < T; t++)
            for (int hh = 0; hh < H; hh++)
                for (int j = 0; j < hd; j++)
                    omg[(size_t)t * d + hh * hd + j] = oh[((size_t)hh * T + t) * hd + j];

        linear_rows(omg, L.out_weight, L.out_bias, proj, T, d, d);
        for (size_t i = 0; i < (size_t)T * d; i++) x[i] += proj[i];

        ln_rows(x, L.ln2_weight, L.ln2_bias, h, T, d);
        linear_rows(h, L.fc1_weight, L.fc1_bias, ff, T, dff, d);
        gelu_inplace(ff, (size_t)T * dff);
        linear_rows(ff, L.fc2_weight, L.fc2_bias, proj, T, d, dff);
        for (size_t i = 0; i < (size_t)T * d; i++) x[i] += proj[i];

        memcpy(trace.layer_out[l], x, sizeof(float) * (size_t)T * d);
    }

    ln_rows(x, m.ln_f_weight, m.ln_f_bias, trace.final_ln, T, d);
    linear_rows(trace.final_ln, m.lm_head, nullptr, trace.logits, T, c.vocab_size, d);

    free(x); free(h); free(qkv); free(q); free(k); free(v);
    free(oh); free(omg); free(proj); free(ff); free(scores);
}

// ------------------------------------------------------------------ decode

RefState ref_state_alloc(const ModelConfig& c, int max_seq)
{
    RefState s;
    s.cfg = c;
    s.max_seq = max_seq;
    s.pos = 0;
    s.k_cache = (float**)malloc(sizeof(float*) * c.n_layer);
    s.v_cache = (float**)malloc(sizeof(float*) * c.n_layer);
    size_t cache_elems = (size_t)c.n_head * max_seq * c.head_dim;
    for (int i = 0; i < c.n_layer; i++) {
        s.k_cache[i] = (float*)calloc(cache_elems, sizeof(float));
        s.v_cache[i] = (float*)calloc(cache_elems, sizeof(float));
    }
    s.x = (float*)malloc(sizeof(float) * c.d_model);
    // scratch is carved up in ref_decode_step as: h[d] qkv[3d] ff[d_ff] tmp[d] scores[max_seq]
    size_t scratch_elems = (size_t)5 * c.d_model + c.d_ff + max_seq;
    s.scratch = (float*)malloc(sizeof(float) * scratch_elems);
    return s;
}

void ref_state_free(RefState& s)
{
    for (int i = 0; i < s.cfg.n_layer; i++) {
        free(s.k_cache[i]);
        free(s.v_cache[i]);
    }
    free(s.k_cache);
    free(s.v_cache);
    free(s.x);
    free(s.scratch);
}

void ref_decode_step(const ModelWeights& m, RefState& s, int token, float* logits_out)
{
    const ModelConfig& c = m.cfg;
    const int d = c.d_model, H = c.n_head, hd = c.head_dim, dff = c.d_ff;
    const int pos = s.pos;
    const int seq = pos + 1;   // cache length after appending this token

    float* x   = s.x;
    float* h   = s.scratch;
    float* qkv = h + d;
    float* ff  = qkv + 3 * d;
    float* tmp = ff + dff;
    float* scores = tmp + d;

    for (int i = 0; i < d; i++)
        x[i] = m.tok_emb[(size_t)token * d + i] + m.pos_emb[(size_t)pos * d + i];

    for (int l = 0; l < c.n_layer; l++) {
        const LayerWeights& L = m.layers[l];

        ln_rows(x, L.ln1_weight, L.ln1_bias, h, 1, d);
        linear_rows(h, L.qkv_weight, L.qkv_bias, qkv, 1, 3 * d, d);

        // append K,V for this position into the cache
        for (int hh = 0; hh < H; hh++) {
            for (int j = 0; j < hd; j++) {
                size_t slot = ((size_t)hh * s.max_seq + pos) * hd + j;
                s.k_cache[l][slot] = qkv[d + hh * hd + j];
                s.v_cache[l][slot] = qkv[2 * d + hh * hd + j];
            }
        }

        const double scale = 1.0 / sqrt((double)hd);

        // single-query attention over the whole cache; no mask needed because
        // the query is always the newest position
        for (int hh = 0; hh < H; hh++) {
            const float* qh = qkv + hh * hd;
            const float* kc = s.k_cache[l] + (size_t)hh * s.max_seq * hd;
            const float* vc = s.v_cache[l] + (size_t)hh * s.max_seq * hd;

            double row_max = -INFINITY;
            for (int j = 0; j < seq; j++) {
                double dot = 0.0;
                for (int e = 0; e < hd; e++)
                    dot += (double)qh[e] * kc[(size_t)j * hd + e];
                double sc = dot * scale;
                scores[j] = (float)sc;
                if (sc > row_max) row_max = sc;
            }

            double sum = 0.0;
            for (int j = 0; j < seq; j++) {
                double e = exp((double)scores[j] - row_max);
                scores[j] = (float)e;
                sum += e;
            }

            for (int e = 0; e < hd; e++) {
                double acc = 0.0;
                for (int j = 0; j < seq; j++)
                    acc += (double)scores[j] * vc[(size_t)j * hd + e];
                tmp[hh * hd + e] = (float)(acc / sum);
            }
        }

        linear_rows(tmp, L.out_weight, L.out_bias, h, 1, d, d);
        for (int i = 0; i < d; i++) x[i] += h[i];

        ln_rows(x, L.ln2_weight, L.ln2_bias, h, 1, d);
        linear_rows(h, L.fc1_weight, L.fc1_bias, ff, 1, dff, d);
        gelu_inplace(ff, dff);
        linear_rows(ff, L.fc2_weight, L.fc2_bias, h, 1, d, dff);
        for (int i = 0; i < d; i++) x[i] += h[i];
    }

    ln_rows(x, m.ln_f_weight, m.ln_f_bias, h, 1, d);
    linear_rows(h, m.lm_head, nullptr, logits_out, 1, c.vocab_size, d);

    s.pos++;
}
