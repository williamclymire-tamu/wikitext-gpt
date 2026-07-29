"""Generate PyTorch reference data for decode attention, layernorm, gelu, residual.
Outputs raw fp32 binary files loadable with load_bin() in C++.
"""

import argparse, json, os, math
import torch
import torch.nn.functional as F


def save(t, path):
    t = t.contiguous().float()
    with open(path, 'wb') as f:
        f.write(t.numpy().tobytes())


def generate_decode_ref(outdir, B, H, d, seq_len, seed):
    """Single-query attention over a KV cache (no causal mask needed)."""
    sub = f'{outdir}/decode_B{B}_H{H}_S{seq_len}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    Q = torch.randn(B * H, d)                       # single query per head
    K_cache = torch.randn(B * H, seq_len, d)
    V_cache = torch.randn(B * H, seq_len, d)

    # Reference: softmax(Q @ K^T / sqrt(d)) @ V
    scale = d ** -0.5
    scores = (Q.unsqueeze(1) @ K_cache.transpose(-2, -1)).squeeze(1) * scale  # [BH, seq_len]
    attn = torch.softmax(scores, dim=-1)                                       # [BH, seq_len]
    O_ref = (attn.unsqueeze(1) @ V_cache).squeeze(1)                           # [BH, d]

    save(Q, f'{sub}/Q.bin')
    save(K_cache, f'{sub}/K_cache.bin')
    save(V_cache, f'{sub}/V_cache.bin')
    save(O_ref, f'{sub}/O_ref.bin')

    meta = {'B': B, 'H': H, 'd': d, 'seq_len': seq_len,
            'max_seq': seq_len + 64, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  decode: B={B} H={H} seq_len={seq_len} d={d} -> {sub}/')


def generate_layernorm_ref(outdir, N, d, seed):
    sub = f'{outdir}/layernorm_N{N}_d{d}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    x = torch.randn(N, d)
    weight = torch.randn(d)
    bias = torch.randn(d)

    ln = F.layer_norm(x, (d,), weight=weight, bias=bias, eps=1e-5)

    save(x, f'{sub}/input.bin')
    save(weight, f'{sub}/weight.bin')
    save(bias, f'{sub}/bias.bin')
    save(ln, f'{sub}/output_ref.bin')

    meta = {'N': N, 'd': d, 'eps': 1e-5, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  layernorm: N={N} d={d} -> {sub}/')


def generate_gelu_ref(outdir, n, seed):
    sub = f'{outdir}/gelu_n{n}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    x = torch.randn(n)
    y = F.gelu(x, approximate='tanh')

    save(x, f'{sub}/input.bin')
    save(y, f'{sub}/output_ref.bin')

    meta = {'n': n, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  gelu: n={n} -> {sub}/')


def generate_residual_ref(outdir, n, seed):
    sub = f'{outdir}/residual_n{n}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    a = torch.randn(n)
    b = torch.randn(n)
    c = a + b

    save(a, f'{sub}/a.bin')
    save(b, f'{sub}/b.bin')
    save(c, f'{sub}/output_ref.bin')

    meta = {'n': n, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  residual: n={n} -> {sub}/')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='test_day1')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Generating Day 1 reference data to {args.outdir}/')

    # Decode attention tests
    generate_decode_ref(args.outdir, B=1, H=4, d=64, seq_len=128, seed=args.seed)
    generate_decode_ref(args.outdir, B=1, H=4, d=64, seq_len=37,  seed=args.seed)
    generate_decode_ref(args.outdir, B=2, H=4, d=64, seq_len=256, seed=args.seed)

    # LayerNorm tests (d=256 matches the GPT model)
    generate_layernorm_ref(args.outdir, N=128, d=256, seed=args.seed)
    generate_layernorm_ref(args.outdir, N=37,  d=256, seed=args.seed)
    generate_layernorm_ref(args.outdir, N=128, d=64,  seed=args.seed)

    # GELU tests
    generate_gelu_ref(args.outdir, n=32768, seed=args.seed)
    generate_gelu_ref(args.outdir, n=1000,  seed=args.seed)

    # Residual add tests
    generate_residual_ref(args.outdir, n=32768, seed=args.seed)

    print('\nDone.')


if __name__ == '__main__':
    main()
