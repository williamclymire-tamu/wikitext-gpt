"""Download WikiText-103 and prepare tokenized splits for training."""

import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

DATA_DIR = "data"
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
VOCAB_SIZE = 16384


def download_wikitext():
    print("downloading WikiText-103...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    print(f"  train: {len(ds['train']):,} | val: {len(ds['validation']):,} | test: {len(ds['test']):,}")
    return ds


def train_tokenizer(ds):
    print(f"training tokenizer (vocab={VOCAB_SIZE})...")

    tmp_path = os.path.join(DATA_DIR, "_train_text.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in ds["train"]:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[tmp_path],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|padding|>"],
    )

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)
    os.remove(tmp_path)
    print(f"  saved to {TOKENIZER_DIR}/")
    return tokenizer


def tokenize_split(tokenizer, texts, name):
    import torch

    print(f"tokenizing {name}...")
    all_ids = []
    eos = tokenizer.token_to_id("<|endoftext|>")

    for row in texts:
        text = row["text"].strip()
        if not text:
            continue
        all_ids.extend(tokenizer.encode(text).ids)
        all_ids.append(eos)

    t = torch.tensor(all_ids, dtype=torch.long)
    out_path = os.path.join(DATA_DIR, f"{name}.pt")
    torch.save(t, out_path)
    print(f"  {name}: {len(all_ids):,} tokens -> {out_path}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = download_wikitext()
    tok = train_tokenizer(ds)
    for split, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        tokenize_split(tok, ds[split], name)
    print("\ndone.")
