import argparse, json, os
import torch
import torch.nn.functional as F

def naive_attention(Q, K, V, causal=True):
    B, H, N, d = Q.shape
    scale = d ** -0.5
    S = (Q @ K.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.softmax(S, dim=-1) @ V

def save_tensor(t, path):
    t = t.contiguous().float()
    with open(path, 'wb') as f:
        f.write(t.numpy().tobytes())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--H', type=int, default=1)
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--causal', action='store_true', default=True)
    p.add_argument('--outdir', type=str, default='test_data')
    args = p.parse_args()

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

if __name__ == '__main__':
    main()
