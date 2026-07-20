from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tokenizers import ByteLevelBPETokenizer

from cslm_model import CSLMConfig, CSLMModel, find_latest_checkpoint, load_checkpoint


def build_tokenizer(tokenizer_dir: Path) -> ByteLevelBPETokenizer:
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        raise FileNotFoundError(
            f"missing tokenizer files in {tokenizer_dir}; expected vocab.json and merges.txt"
        )
    return ByteLevelBPETokenizer(str(vocab_file), str(merges_file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained CSLM checkpoint.")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt to continue.")
    parser.add_argument(
        "--checkpoint",
        default="cslm/checkpoint",
        help="Checkpoint file or directory. If a directory is given, the newest .pt/.pth file is used.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="cslm/tokenizer",
        help="Directory containing vocab.json and merges.txt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    tokenizer_dir = Path(args.tokenizer_dir)
    checkpoint_path = find_latest_checkpoint(Path(args.checkpoint))

    if checkpoint_path is None:
        print(
            "No checkpoint found. Train the model first, then place the saved .pt file in cslm/checkpoint/ "
            "or pass --checkpoint /path/to/checkpoint.pt.",
            file=sys.stderr,
        )
        return 1

    tokenizer = build_tokenizer(tokenizer_dir)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = CSLMConfig.from_checkpoint(checkpoint)
    model = CSLMModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    prompt = args.prompt
    encoded = tokenizer.encode(prompt)
    input_ids = encoded.ids
    if not input_ids:
        end_token = tokenizer.token_to_id("<|endoftext|>")
        if end_token is None:
            raise RuntimeError("tokenizer is missing <|endoftext|>")
        input_ids = [end_token]

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tokenizer.decode(generated[0].tolist())
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())