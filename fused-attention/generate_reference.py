#!/usr/bin/env python3
"""Generate reference attention inputs/outputs for CUDA kernel validation.

Dumps raw fp32 binary files: Q.bin, K.bin, V.bin, O.bin
Also runs PyTorch's F.scaled_dot_product_attention for a second reference.

Usage:
    python generate_reference.py                          # defaults
    python generate_reference.py --N 512 --d 64 --causal  # custom
"""

import argparse
import json
import os
import struct

import torch
import torch.nn.functional as F


def naive_attention(Q, K, V, causal=True):
    """Explicit Q @ K^T, mask, softmax, @ V. No fused ops."""
    B, H, N, d = Q.shape
    scale = d ** -0.5
    S = (Q @ K.transpose(-2, -1)) * scale          # (B, H, N, N)
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    A = torch.softmax(S, dim=-1)
    O = A @ V
    return O


def save_tensor(t, path):
    """Save a contiguous fp32 tensor as raw bytes."""
    t = t.contiguous().float()
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="batch size")
    parser.add_argument("--H", type=int, default=1, help="number of heads")
    parser.add_argument("--N", type=int, default=128, help="sequence length")
    parser.add_argument("--d", type=int, default=64, help="head dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--outdir", type=str, default="test_data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    Q = torch.randn(args.B, args.H, args.N, args.d)
    K = torch.randn(args.B, args.H, args.N, args.d)
    V = torch.randn(args.B, args.H, args.N, args.d)

    O_naive = naive_attention(Q, K, V, causal=args.causal)

    # Also compute with PyTorch's optimized SDPA for comparison
    O_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=args.causal)

    diff = (O_naive - O_sdpa).abs()
    print(f"Naive vs SDPA max abs diff: {diff.max().item():.2e}")

    os.makedirs(args.outdir, exist_ok=True)
    save_tensor(Q, os.path.join(args.outdir, "Q.bin"))
    save_tensor(K, os.path.join(args.outdir, "K.bin"))
    save_tensor(V, os.path.join(args.outdir, "V.bin"))
    save_tensor(O_naive, os.path.join(args.outdir, "O_ref.bin"))

    meta = {
        "B": args.B, "H": args.H, "N": args.N, "d": args.d,
        "causal": args.causal, "seed": args.seed, "dtype": "float32",
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {args.outdir}/: Q.bin K.bin V.bin O_ref.bin meta.json")
    print(f"  Shape: ({args.B}, {args.H}, {args.N}, {args.d})")
    print(f"  Causal: {args.causal}")

    # Generate additional test cases for non-power-of-two
    for test_N in [37, 127, 200]:
        subdir = os.path.join(args.outdir, f"N{test_N}")
        os.makedirs(subdir, exist_ok=True)
        torch.manual_seed(args.seed)
        Qq = torch.randn(args.B, args.H, test_N, args.d)
        Kk = torch.randn(args.B, args.H, test_N, args.d)
        Vv = torch.randn(args.B, args.H, test_N, args.d)
        Oo = naive_attention(Qq, Kk, Vv, causal=args.causal)
        save_tensor(Qq, os.path.join(subdir, "Q.bin"))
        save_tensor(Kk, os.path.join(subdir, "K.bin"))
        save_tensor(Vv, os.path.join(subdir, "V.bin"))
        save_tensor(Oo, os.path.join(subdir, "O_ref.bin"))
        m = {**meta, "N": test_N}
        with open(os.path.join(subdir, "meta.json"), "w") as f:
            json.dump(m, f, indent=2)
        print(f"  + non-power-of-two test: N={test_N}")


if __name__ == "__main__":
    main()
