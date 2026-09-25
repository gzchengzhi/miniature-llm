"""
experiments/exp_b_semantic_distance.py
======================================
Verification Experiment B: Semantic Distance vs. Representation Similarity.

Compares spreading-activation distances with cosine similarity of the
miniature LLM's hidden-state representations for word pairs.

Run:
    python experiments/exp_b_semantic_distance.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM
from model.glossary import gloss_text

# Semantic distances computed from the spreading activation model
WORD_PAIRS = [
    # (word_a, word_b, distance)
    ("虎", "羊", 0.15),
    ("虎", "狗", 0.20),
    ("虎", "鸡", 0.35),
    ("狐", "牛", 0.40),
    ("虎", "草", 0.75),
    ("狐", "麦", 0.80),
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
def embedding_of(model, tokenizer, word, device):
    ids = tokenizer.encode(word)
    if not ids:
        return None
    idx = torch.tensor([[ids[0]]], dtype=torch.long, device=device)
    emb = model.token_emb(idx)
    return emb[0, 0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    print(f"{'Pair (Chinese)':<14} {'Pair (English)':<24} "
          f"{'Distance':>10} {'CosSim':>10}")
    print("-" * 62)

    distances, sims = [], []
    for a, b, dist in WORD_PAIRS:
        ea = embedding_of(model, tokenizer, a, device)
        eb = embedding_of(model, tokenizer, b, device)
        if ea is None or eb is None:
            continue
        sim = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        pair_zh = f"{a}-{b}"
        pair_en = f"{gloss_text(a)}-{gloss_text(b)}"
        print(f"{pair_zh:<14} {pair_en:<24} {dist:>10.3f} {sim:>10.3f}")

        distances.append(dist)
        sims.append(sim)

    if len(distances) > 1:
        import numpy as np
        corr = np.corrcoef(distances, sims)[0, 1]
        print(f"\nPearson correlation (distance vs. cos-sim): {corr:.3f}")


if __name__ == "__main__":
    main()