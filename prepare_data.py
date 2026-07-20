"""prepare_data.py -- Download WikiText-103 and prepare it for training.

Steps:
  1. Download WikiText-103-raw via HuggingFace datasets
  2. Train a BPE tokenizer on the training split
  3. Tokenize all splits (train / validation / test)
  4. Save as .pt files for fast loading during training

Usage:
    pip install datasets tokenizers
    python prepare_data.py

Output:
    data/tokenizer/vocab.json, merges.txt
    data/train.pt, data/val.pt, data/test.pt
"""

import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


DATA_DIR = "data"
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
VOCAB_SIZE = 16384  # reasonable for WikiText-103 (~100M tokens)


def download_wikitext():
    """Download WikiText-103-raw-v1 from HuggingFace."""
    print("downloading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    print(f"  train: {len(ds['train']):,} lines")
    print(f"  val:   {len(ds['validation']):,} lines")
    print(f"  test:  {len(ds['test']):,} lines")
    return ds


def train_tokenizer(ds):
    """Train a BPE tokenizer on the training split."""
    print(f"\ntraining BPE tokenizer (vocab_size={VOCAB_SIZE})...")

    # write training text to a temp file (tokenizers library wants a file path)
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
    print(f"  saved to {TOKENIZER_DIR}/")

    os.remove(tmp_path)
    return tokenizer


def tokenize_split(tokenizer, texts, name):
    """Tokenize a dataset split into a flat list of token IDs."""
    import torch

    print(f"tokenizing {name}...")
    all_ids = []
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    for row in texts:
        text = row["text"].strip()
        if not text:
            continue
        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)
        all_ids.append(eos_id)  # document boundary

    tensor = torch.tensor(all_ids, dtype=torch.long)
    out_path = os.path.join(DATA_DIR, f"{name}.pt")
    torch.save(tensor, out_path)
    print(f"  {name}: {len(all_ids):,} tokens -> {out_path}")
    return tensor


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    ds = download_wikitext()
    tokenizer = train_tokenizer(ds)

    tokenize_split(tokenizer, ds["train"], "train")
    tokenize_split(tokenizer, ds["validation"], "val")
    tokenize_split(tokenizer, ds["test"], "test")

    print("\ndone. run `python train.py` next.")


if __name__ == "__main__":
    main()
