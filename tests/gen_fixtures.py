"""Generate PyTorch reference fixtures for all CUDA kernel tests.

Subcommands:
    python gen_fixtures.py attention --outdir test_data
    python gen_fixtures.py decode    --outdir test_decode_data
    python gen_fixtures.py all       (generates both with default dirs)
"""

import argparse, json, os
import torch
import torch.nn.functional as F


# ── Shared helpers ────────────────────────────────────────────────────────────

def save_tensor(t, path):
    """Write a contiguous float32 tensor to a raw binary file."""
    t = t.contiguous().float()
    with open(path, 'wb') as f:
        f.write(t.numpy().tobytes())


# ── Prefill attention ─────────────────────────────────────────────────────────

def naive_attention(Q, K, V, causal=True):
    B, H, N, d = Q.shape
    scale = d ** -0.5
    S = (Q @ K.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.softmax(S, dim=-1) @ V


def gen_attention(args):
    """Prefill attention: naive vs fused, multiple sequence lengths."""
    torch.manual_seed(args.seed)
    Q = torch.randn(args.B, args.H, args.N, args.d)
    K = torch.randn(args.B, args.H, args.N, args.d)
    V = torch.randn(args.B, args.H, args.N, args.d)
    O = naive_attention(Q, K, V, causal=args.causal)

    os.makedirs(args.outdir, exist_ok=True)
    save_tensor(Q, f'{args.outdir}/Q.bin')
    save_tensor(K, f'{args.outdir}/K.bin')
    save_tensor(V, f'{args.outdir}/V.bin')
    save_tensor(O, f'{args.outdir}/O_ref.bin')

    meta = {'B': args.B, 'H': args.H, 'N': args.N, 'd': args.d,
            'causal': args.causal, 'seed': args.seed, 'dtype': 'float32'}
    with open(f'{args.outdir}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved ({args.B},{args.H},{args.N},{args.d}) to {args.outdir}/')

    for test_N in [37, 127, 200]:
        sub = f'{args.outdir}/N{test_N}'
        os.makedirs(sub, exist_ok=True)
        torch.manual_seed(args.seed)
        Qq = torch.randn(args.B, args.H, test_N, args.d)
        Kk = torch.randn(args.B, args.H, test_N, args.d)
        Vv = torch.randn(args.B, args.H, test_N, args.d)
        Oo = naive_attention(Qq, Kk, Vv, causal=args.causal)
        save_tensor(Qq, f'{sub}/Q.bin')
        save_tensor(Kk, f'{sub}/K.bin')
        save_tensor(Vv, f'{sub}/V.bin')
        save_tensor(Oo, f'{sub}/O_ref.bin')
        with open(f'{sub}/meta.json', 'w') as f:
            json.dump({**meta, 'N': test_N}, f, indent=2)
        print(f'  + N={test_N}')


# ── Decode attention + elementwise kernels ────────────────────────────────────

def gen_decode_ref(outdir, B, H, d, seq_len, seed):
    sub = f'{outdir}/decode_B{B}_H{H}_S{seq_len}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    Q = torch.randn(B * H, d)
    K_cache = torch.randn(B * H, seq_len, d)
    V_cache = torch.randn(B * H, seq_len, d)

    scale = d ** -0.5
    scores = (Q.unsqueeze(1) @ K_cache.transpose(-2, -1)).squeeze(1) * scale
    attn = torch.softmax(scores, dim=-1)
    O_ref = (attn.unsqueeze(1) @ V_cache).squeeze(1)

    save_tensor(Q, f'{sub}/Q.bin')
    save_tensor(K_cache, f'{sub}/K_cache.bin')
    save_tensor(V_cache, f'{sub}/V_cache.bin')
    save_tensor(O_ref, f'{sub}/O_ref.bin')

    meta = {'B': B, 'H': H, 'd': d, 'seq_len': seq_len,
            'max_seq': seq_len + 64, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  decode: B={B} H={H} seq_len={seq_len} d={d}')


def gen_layernorm_ref(outdir, N, d, seed):
    sub = f'{outdir}/layernorm_N{N}_d{d}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    x = torch.randn(N, d)
    weight = torch.randn(d)
    bias = torch.randn(d)
    ln = F.layer_norm(x, (d,), weight=weight, bias=bias, eps=1e-5)

    save_tensor(x, f'{sub}/input.bin')
    save_tensor(weight, f'{sub}/weight.bin')
    save_tensor(bias, f'{sub}/bias.bin')
    save_tensor(ln, f'{sub}/output_ref.bin')

    meta = {'N': N, 'd': d, 'eps': 1e-5, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  layernorm: N={N} d={d}')


def gen_gelu_ref(outdir, n, seed):
    sub = f'{outdir}/gelu_n{n}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    x = torch.randn(n)
    y = F.gelu(x, approximate='tanh')

    save_tensor(x, f'{sub}/input.bin')
    save_tensor(y, f'{sub}/output_ref.bin')

    meta = {'n': n, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  gelu: n={n}')


def gen_residual_ref(outdir, n, seed):
    sub = f'{outdir}/residual_n{n}'
    os.makedirs(sub, exist_ok=True)

    torch.manual_seed(seed)
    a = torch.randn(n)
    b = torch.randn(n)
    c = a + b

    save_tensor(a, f'{sub}/a.bin')
    save_tensor(b, f'{sub}/b.bin')
    save_tensor(c, f'{sub}/output_ref.bin')

    meta = {'n': n, 'seed': seed}
    with open(f'{sub}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  residual: n={n}')


def gen_decode(args):
    """Decode attention, layernorm, gelu, residual fixtures."""
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Generating decode/elementwise fixtures to {args.outdir}/')

    gen_decode_ref(args.outdir, B=1, H=4, d=64, seq_len=128, seed=args.seed)
    gen_decode_ref(args.outdir, B=1, H=4, d=64, seq_len=37,  seed=args.seed)
    gen_decode_ref(args.outdir, B=2, H=4, d=64, seq_len=256, seed=args.seed)

    gen_layernorm_ref(args.outdir, N=128, d=256, seed=args.seed)
    gen_layernorm_ref(args.outdir, N=37,  d=256, seed=args.seed)
    gen_layernorm_ref(args.outdir, N=128, d=64,  seed=args.seed)

    gen_gelu_ref(args.outdir, n=32768, seed=args.seed)
    gen_gelu_ref(args.outdir, n=1000,  seed=args.seed)

    gen_residual_ref(args.outdir, n=32768, seed=args.seed)
    print('Done.')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    top = argparse.ArgumentParser(description='Generate CUDA kernel test fixtures')
    sub = top.add_subparsers(dest='command', required=True)

    # attention subcommand
    ap = sub.add_parser('attention', help='Prefill attention fixtures')
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--H', type=int, default=1)
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--causal', action='store_true', default=True)
    ap.add_argument('--outdir', default='test_data')
    ap.add_argument('--seed', type=int, default=42)

    # decode subcommand
    dp = sub.add_parser('decode', help='Decode attention + elementwise fixtures')
    dp.add_argument('--outdir', default='test_decode_data')
    dp.add_argument('--seed', type=int, default=42)

    # all subcommand
    allp = sub.add_parser('all', help='Generate all fixtures')
    allp.add_argument('--seed', type=int, default=42)

    args = top.parse_args()

    if args.command == 'attention':
        gen_attention(args)
    elif args.command == 'decode':
        gen_decode(args)
    elif args.command == 'all':
        # attention with defaults
        class AttnArgs:
            B = 1; H = 1; N = 128; d = 64; causal = True
            outdir = 'test_data'; seed = args.seed
        gen_attention(AttnArgs())
        # decode with defaults
        class DecodeArgs:
            outdir = 'test_decode_data'; seed = args.seed
        gen_decode(DecodeArgs())


if __name__ == '__main__':
    main()
