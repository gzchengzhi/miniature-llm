"""
experiments/exp_a_sentence_strength.py
======================================
Verification Experiment A: Sentence Strength vs. Perplexity.

Compares the conceptual triangle model's sentence strength with the miniature
LLM's perplexity on the same set of sentences.

Run:
    python experiments/exp_a_sentence_strength.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM
from model.glossary import gloss_text

# Sentence strengths computed from the conceptual triangle model
TEST_SENTENCES = [
    # (sentence, strength)
    ("虎吃羊。", 1.21),
    ("虎吃狗。", 1.21),
    ("狐吃鸡。", 1.16),
    ("鸡吃虫。", 1.17),
    ("羊吃草。", 1.05),
    ("牛吃草。", 0.99),
    ("虎追牛。", 1.21),
    ("狗追鸡。", 1.14),
    ("羊抓虎。", 0.30),
    ("肉追羊。", 0.30),
    ("草赶虫。", 0.30),
    ("麦抓牛。", 0.30),
    ("肉赶虎。", 0.30),
]


def load_model(device):
    checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)
    saved_cfg = checkpoint["config"]
    tokenizer = CharTokenizer.load("checkpoints/tokenizer.json")
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=saved_cfg["n_embd"],
        n_head=saved_cfg["n_head"],
        n_layer=saved_cfg["n_layer"],
        block_size=saved_cfg["block_size"],
        dropout=saved_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def perplexity(model, tokenizer, sentence, device):
    ids = tokenizer.encode(sentence)
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    _, loss = model(x, y)
    return math.exp(loss.item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    print(f"{'Sentence':<10} {'English':<30} {'Strength':>10} {'PPL':>10}")
    print("-" * 64)

    strengths, ppls = [], []
    for sent, strength in TEST_SENTENCES:
        ppl = perplexity(model, tokenizer, sent, device)
        english = gloss_text(sent)
        print(f"{sent:<10} {english:<30} {strength:>10.3f} {ppl:>10.3f}")
        strengths.append(strength)
        ppls.append(ppl)

    if len(strengths) > 1:
        import numpy as np
        corr = np.corrcoef(strengths, ppls)[0, 1]
        print(f"\nPearson correlation (strength vs. PPL): {corr:.3f}")


if __name__ == "__main__":
    main()