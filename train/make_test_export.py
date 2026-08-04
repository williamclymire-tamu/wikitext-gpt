"""Generate a random-weight export in the engine's on-disk format, using NumPy only.

Purpose: bring up and regression-test the C++/CUDA engine without needing a GPU,
a trained checkpoint, or even PyTorch installed. The activations it writes into
parity/ are computed by the NumPy forward pass below, which is a line-for-line
transcription of model.py.

    python tools/make_test_export.py --out export_test

The engine's scalar CPU reference is checked against these fixtures on any machine;
the CUDA path is then checked against the same fixtures on a GPU box. Same oracle
for both, so a CUDA-only failure is unambiguously a kernel bug.
"""

import argparse
import json
import os
import struct

import numpy as np


# ---------------------------------------------------------------- ops
# Each of these mirrors exactly one thing the engine does.

def layernorm(x, w, b, eps=1e-5):
    """Biased variance (divide by d), matching torch.nn.LayerNorm."""
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    """tanh approximation -- matches elementwise.cu and F.gelu(approximate='tanh')."""
    inner = 0.7978845608 * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def linear(x, w, b):
    """nn.Linear semantics: w is [out, in], so y = x @ w.T + b."""
    y = x @ w.T
    return y + b if b is not None else y


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------- forward

def forward(w, cfg, ids, trace):
    """NumPy transcription of GPT.forward, dropout removed (eval mode)."""
    T = len(ids)
    d = cfg["d_model"]
    H = cfg["n_head"]
    hd = cfg["head_dim"]

    x = w["tok_emb"][ids] + w["pos_emb"][:T]
    trace["embed"] = x.copy()

    for i in range(cfg["n_layer"]):
        L = w["layers"][i]

        h = layernorm(x, L["ln1.weight"], L["ln1.bias"])
        qkv = linear(h, L["qkv.weight"], L["qkv.bias"])          # [T, 3d]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # [T, d] -> [H, T, hd]; this reshape+transpose is the layout the
        # attention kernels expect, and the most common place to get it wrong
        q = q.reshape(T, H, hd).transpose(1, 0, 2)
        k = k.reshape(T, H, hd).transpose(1, 0, 2)
        v = v.reshape(T, H, hd).transpose(1, 0, 2)

        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)          # [H, T, T]
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal[None], -np.inf, scores)
        attn = softmax(scores, axis=-1)

        o = attn @ v                                             # [H, T, hd]
        o = o.transpose(1, 0, 2).reshape(T, d)
        x = x + linear(o, L["out.weight"], L["out.bias"])

        h = layernorm(x, L["ln2.weight"], L["ln2.bias"])
        f = gelu(linear(h, L["fc1.weight"], L["fc1.bias"]))
        x = x + linear(f, L["fc2.weight"], L["fc2.bias"])

        trace[f"layer{i}"] = x.copy()

    x = layernorm(x, w["ln_f.weight"], w["ln_f.bias"])
    trace["final"] = x.copy()

    logits = x @ w["lm_head"].T
    trace["logits"] = logits
    return logits


# ---------------------------------------------------------------- io

def w32(path, a):
    a = np.ascontiguousarray(a, dtype="<f4")
    with open(path, "wb") as f:
        f.write(a.tobytes(order="C"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="export_test")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--context-len", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)   # head_dim 64, matches HEAD_DIM
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = {
        "vocab_size": args.vocab_size,
        "context_len": args.context_len,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_model": args.d_model,
        "d_ff": 4 * args.d_model,
        "head_dim": args.d_model // args.n_head,
    }
    assert cfg["head_dim"] * cfg["n_head"] == cfg["d_model"]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    rng = np.random.default_rng(args.seed)
    d, dff, V = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]

    def rnd(*shape):
        # 0.02 std matches model.py's _init_weights, so activations stay in a
        # realistic range instead of saturating tanh/softmax
        return rng.normal(0, 0.02, shape).astype(np.float32)

    w = {
        "tok_emb": rnd(V, d),
        "pos_emb": rnd(cfg["context_len"], d),
        "ln_f.weight": rng.normal(1, 0.05, d).astype(np.float32),
        "ln_f.bias": rnd(d),
        "lm_head": rnd(V, d),
        "layers": [],
    }
    for _ in range(cfg["n_layer"]):
        w["layers"].append({
            "ln1.weight": rng.normal(1, 0.05, d).astype(np.float32),
            "ln1.bias": rnd(d),
            "qkv.weight": rnd(3 * d, d), "qkv.bias": rnd(3 * d),
            "out.weight": rnd(d, d),     "out.bias": rnd(d),
            "ln2.weight": rng.normal(1, 0.05, d).astype(np.float32),
            "ln2.bias": rnd(d),
            "fc1.weight": rnd(dff, d),   "fc1.bias": rnd(dff),
            "fc2.weight": rnd(d, dff),   "fc2.bias": rnd(d),
        })

    w32(os.path.join(args.out, "tok_emb.bin"), w["tok_emb"])
    w32(os.path.join(args.out, "pos_emb.bin"), w["pos_emb"])
    w32(os.path.join(args.out, "ln_f.weight.bin"), w["ln_f.weight"])
    w32(os.path.join(args.out, "ln_f.bias.bin"), w["ln_f.bias"])
    w32(os.path.join(args.out, "lm_head.bin"), w["lm_head"])
    for i, L in enumerate(w["layers"]):
        for key, arr in L.items():
            w32(os.path.join(args.out, f"layer{i}.{key}.bin"), arr)

    # a trivial byte-per-token table so the detokenizer has something to read
    tok_dir = os.path.join(args.out, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    blob, offsets = bytearray(), [0]
    for i in range(V):
        s = f"<{i}>".encode()
        blob += s
        offsets.append(len(blob))
    with open(os.path.join(tok_dir, "token_blob.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(tok_dir, "token_offsets.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(offsets)}I", *offsets))

    # parity fixtures
    T = args.prompt_len
    ids = rng.integers(0, V, T).astype(np.int32)
    trace = {}
    forward(w, cfg, ids, trace)

    par = os.path.join(args.out, "parity")
    os.makedirs(par, exist_ok=True)
    with open(os.path.join(par, "prompt.bin"), "wb") as f:
        f.write(ids.astype("<i4").tobytes())
    with open(os.path.join(par, "meta.json"), "w") as f:
        json.dump({"prompt_len": int(T)}, f)
    w32(os.path.join(par, "embed.bin"), trace["embed"])
    for i in range(cfg["n_layer"]):
        w32(os.path.join(par, f"layer{i}.bin"), trace[f"layer{i}"])
    w32(os.path.join(par, "final.bin"), trace["final"])
    w32(os.path.join(par, "logits.bin"), trace["logits"])

    print(f"wrote {args.out}/  (V={V} L={cfg['n_layer']} d={d} hd={cfg['head_dim']} T={T})")
    print(f"  logits range [{trace['logits'].min():.4f}, {trace['logits'].max():.4f}]")
    print(f"  argmax(last) = {int(trace['logits'][-1].argmax())}")


if __name__ == "__main__":
    main()
