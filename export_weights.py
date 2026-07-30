"""Export a trained checkpoint to flat binaries for the C++/CUDA inference engine.

    python export_weights.py                                # checkpoints/best.pt -> export/
    python export_weights.py --checkpoint checkpoints/last.pt --out export
    python export_weights.py --random                       # untrained weights, for engine bring-up

Everything is fp32, little-endian, C-order. No framework, no pickle, no schema --
the engine reads sizes out of config.json and fread()s the rest.

Layout produced:

    export/config.json              model dims
    export/tok_emb.bin              [vocab_size, d_model]
    export/pos_emb.bin              [context_len, d_model]
    export/ln_f.weight.bin          [d_model]
    export/ln_f.bias.bin            [d_model]
    export/lm_head.bin              [vocab_size, d_model]
    export/layer{i}.qkv.weight.bin  [3*d_model, d_model]      (nn.Linear layout: [out, in])
    export/layer{i}.qkv.bias.bin    [3*d_model]
    ... etc per layer

    export/tokenizer/token_blob.bin     concatenated raw UTF-8 bytes of every token
    export/tokenizer/token_offsets.bin  [vocab_size+1] uint32 offsets into the blob

    export/parity/meta.json         prompt length
    export/parity/prompt.bin        [T] int32 token ids
    export/parity/embed.bin         [T, d_model]  after tok+pos embedding
    export/parity/layer{i}.bin      [T, d_model]  after each transformer block
    export/parity/final.bin         [T, d_model]  after ln_f
    export/parity/logits.bin        [T, vocab_size]

The parity fixtures are per-stage on purpose. A single end-to-end logit check tells
you that something is wrong; per-stage tells you which layer.
"""

import argparse
import json
import os
import struct

import torch

from model import GPT, GPTConfig


# ---------------------------------------------------------------- tensor io

def write_tensor(path, t):
    """fp32, little-endian, C-order. Detached and forced contiguous."""
    a = t.detach().to(torch.float32).contiguous().cpu().numpy()
    if a.dtype.byteorder == ">":
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    with open(path, "wb") as f:
        f.write(a.tobytes(order="C"))
    return a.size


def export_model(model, out_dir):
    cfg = model.config
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "vocab_size": cfg.vocab_size,
        "context_len": cfg.context_len,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.d_model,
        "d_ff": 4 * cfg.d_model,
        "head_dim": cfg.d_model // cfg.n_head,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = 0
    total += write_tensor(os.path.join(out_dir, "tok_emb.bin"), model.tok_emb.weight)
    total += write_tensor(os.path.join(out_dir, "pos_emb.bin"), model.pos_emb.weight)
    total += write_tensor(os.path.join(out_dir, "ln_f.weight.bin"), model.ln_f.weight)
    total += write_tensor(os.path.join(out_dir, "ln_f.bias.bin"), model.ln_f.bias)

    # lm_head is tied to tok_emb in model.py. Exported separately anyway so the
    # engine does not have to know or care about tying.
    total += write_tensor(os.path.join(out_dir, "lm_head.bin"), model.lm_head.weight)

    for i, block in enumerate(model.blocks):
        p = os.path.join(out_dir, f"layer{i}")
        total += write_tensor(f"{p}.ln1.weight.bin", block.ln1.weight)
        total += write_tensor(f"{p}.ln1.bias.bin", block.ln1.bias)
        total += write_tensor(f"{p}.qkv.weight.bin", block.attn.qkv.weight)
        total += write_tensor(f"{p}.qkv.bias.bin", block.attn.qkv.bias)
        total += write_tensor(f"{p}.out.weight.bin", block.attn.out_proj.weight)
        total += write_tensor(f"{p}.out.bias.bin", block.attn.out_proj.bias)
        total += write_tensor(f"{p}.ln2.weight.bin", block.ln2.weight)
        total += write_tensor(f"{p}.ln2.bias.bin", block.ln2.bias)
        total += write_tensor(f"{p}.fc1.weight.bin", block.ffn.fc1.weight)
        total += write_tensor(f"{p}.fc1.bias.bin", block.ffn.fc1.bias)
        total += write_tensor(f"{p}.fc2.weight.bin", block.ffn.fc2.weight)
        total += write_tensor(f"{p}.fc2.bias.bin", block.ffn.fc2.bias)

    print(f"  weights: {total:,} floats ({total * 4 / 1e6:.1f} MB)")
    return meta


# ---------------------------------------------------------------- tokenizer

def bytes_to_unicode():
    """GPT-2's reversible byte <-> printable-unicode map (Radford et al. 2019).

    ByteLevelBPE stores token strings in this remapped space so that every byte
    is a printable character. The engine wants raw bytes, so we invert it here
    and ship a plain byte table -- no unicode handling in C++ at all.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def export_tokenizer(tokenizer_dir, out_dir, vocab_size):
    """Flatten the vocab into a byte blob + offset table."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"  ! no tokenizer at {vocab_path}, skipping (engine will print raw ids)")
        return

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    unicode_to_byte = {v: k for k, v in bytes_to_unicode().items()}

    table = [b""] * vocab_size
    for tok_str, tok_id in vocab.items():
        if tok_id >= vocab_size:
            continue
        try:
            table[tok_id] = bytes(unicode_to_byte[ch] for ch in tok_str)
        except KeyError:
            # special tokens (<|endoftext|>) are not in the byte map; keep literal
            table[tok_id] = tok_str.encode("utf-8")

    tok_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    offsets = [0]
    with open(os.path.join(tok_dir, "token_blob.bin"), "wb") as f:
        for b in table:
            f.write(b)
            offsets.append(offsets[-1] + len(b))

    with open(os.path.join(tok_dir, "token_offsets.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(offsets)}I", *offsets))

    print(f"  tokenizer: {vocab_size:,} tokens, {offsets[-1]:,} bytes")


# ---------------------------------------------------------------- parity

@torch.no_grad()
def export_parity(model, prompt_ids, out_dir):
    """Re-walk model.forward() by hand, dumping every intermediate.

    Mirrors GPT.forward exactly. If this drifts from model.py the parity test
    will fail loudly, which is the point.
    """
    par = os.path.join(out_dir, "parity")
    os.makedirs(par, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    T = idx.shape[1]

    with open(os.path.join(par, "prompt.bin"), "wb") as f:
        f.write(struct.pack(f"<{T}i", *prompt_ids))
    with open(os.path.join(par, "meta.json"), "w") as f:
        json.dump({"prompt_len": T}, f)

    # dropout is identity in eval(), so this matches the training-time graph
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=device))
    write_tensor(os.path.join(par, "embed.bin"), x[0])

    for i, block in enumerate(model.blocks):
        x = block(x)
        write_tensor(os.path.join(par, f"layer{i}.bin"), x[0])

    x = model.ln_f(x)
    write_tensor(os.path.join(par, "final.bin"), x[0])

    logits = model.lm_head(x)
    write_tensor(os.path.join(par, "logits.bin"), logits[0])

    print(f"  parity: prompt_len={T}, {len(model.blocks)} stage dumps + logits")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--tokenizer-dir", default="data/tokenizer")
    p.add_argument("--out", default="export")
    p.add_argument("--random", action="store_true",
                   help="skip the checkpoint and export freshly initialized weights "
                        "(lets you bring the engine up before training finishes)")
    p.add_argument("--parity-len", type=int, default=32)
    args = p.parse_args()

    if args.random:
        print("exporting RANDOM weights (engine bring-up mode)")
        config = GPTConfig()
        model = GPT(config)
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        model = GPT(config)
        model.load_state_dict(ckpt["model_state"])
        print(f"loaded {args.checkpoint}: epoch {ckpt['epoch']}, val ppl {ckpt['val_ppl']:.2f}")

    model.eval()
    print(f"exporting to {args.out}/ ...")

    export_model(model, args.out)
    export_tokenizer(args.tokenizer_dir, args.out, config.vocab_size)

    T = min(args.parity_len, config.context_len)
    g = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(0, config.vocab_size, (T,), generator=g).tolist()
    export_parity(model, prompt_ids, args.out)

    print("done.")


if __name__ == "__main__":
    main()
