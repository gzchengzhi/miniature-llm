"""
model/tokenizer.py
==================
Character-level tokenizer for the MCS corpus.
"""

import json


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text=None, stoi=None, itos=None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(self.stoi)
        elif text is not None:
            chars = sorted(list(set(text)))
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.itos = {i: c for i, c in enumerate(chars)}
            self.vocab_size = len(chars)
        else:
            raise ValueError("Either text or (stoi, itos) must be provided.")

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": {str(k): v for k, v in self.itos.items()},
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            stoi=data["stoi"],
            itos={int(k): v for k, v in data["itos"].items()},
        )