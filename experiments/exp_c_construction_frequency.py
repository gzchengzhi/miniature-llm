"""
experiments/exp_c_construction_frequency.py
===========================================
Verification Experiment C: Construction Frequency vs. Perplexity.

Compares construction frequencies in the corpus with the miniature LLM's
average perplexity on sentences of each construction type.

Run:
    python experiments/exp_c_construction_frequency.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM


# Construction frequency and sample sentences
CONSTRUCTIONS = {
    "simple_transitive": {
        "frequency": 0.60,
        "sentences": ["虎吃羊。", "狗追鸡。", "鸡吃虫。", "狐抓兔。"],
    },
    "adj_subject": {
        "frequency": 0.25,
        "sentences": ["饿虎吃羊。", "肥狗追鸡。"],
    },
    "adj_object": {
        "frequency": 0.10,
        "sentences": ["虎吃肥羊。", "狗追饿鸡。"],
    },
    "double_adj": {
        "frequency": 0.05,
        "sentences": ["饿虎吃肥羊。"],
    },
}


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
def avg_perplexity(model, tokenizer, sentences, device):
    vals = []
    for s in sentences:
        ids = tokenizer.encode(s)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        _, loss = model(x, y)
        vals.append(math.exp(loss.item()))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    print(f"{'Construction':<22} {'Freq':>8} {'Avg PPL':>10}")
    print("-" * 42)

    freqs, ppls = [], []
    for name, spec in CONSTRUCTIONS.items():
        ppl = avg_perplexity(model, tokenizer, spec["sentences"], device)
        print(f"{name:<22} {spec['frequency']:>8.2f} {ppl:>10.3f}")
        freqs.append(spec["frequency"])
        ppls.append(ppl)

    if len(freqs) > 1:
        import numpy as np
        corr = np.corrcoef(freqs, ppls)[0, 1]
        print(f"\nPearson correlation (frequency vs. PPL): {corr:.3f}")


if __name__ == "__main__":
    main()