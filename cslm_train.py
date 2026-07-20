from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["cs_corpus.txt"],
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)

os.makedirs("cslm/tokenizer", exist_ok=True)
tokenizer.save_model("cslm/tokenizer")
print("tokenizer trained and saved to cslm/tokenizer")

samples = [
    "Dijkstra's algorithm finds the shortest path between nodes in a weighted graph.",
    "def dijkstra(graph, start, end):\n    \"\"\"Return the cost of the shortest path.\"\"\"",
]
for sample in samples:
    enc = tokenizer.encode(sample)
    decoded = tokenizer.decode(enc.ids)
    print(f"\nsample: {sample!r}")
    print(f"tokens ({len(enc.ids)}): {enc.tokens[:12]}...")
    print(f"round-trip exact match: {decoded == sample}")