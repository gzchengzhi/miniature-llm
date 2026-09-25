"""
experiments/exp_d_learning_curve.py
===================================
Verification Experiment D: Learning Curve Comparison.

Trains the miniature LLM on subsets of the corpus of increasing size and
records validation perplexity for each subset.

Run:
    python experiments/exp_d_learning_curve.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM
from train import prepare_data, make_get_batch, estimate_loss


SUBSET_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
TRAIN_ITERS_PER_SUBSET = 500


def train_on_subset(text, tokenizer, cfg, fraction, device):
    # Subset the corpus
    lines = text.split("\n")
    n_lines = max(1, int(len(lines) * fraction))
    subset_text = "\n".join(lines[:n_lines])

    train_data, val_data = prepare_data(subset_text, tokenizer, cfg.block_size)
    get_batch = make_get_batch(cfg)

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for it in range(TRAIN_ITERS_PER_SUBSET):
        x, y = get_batch(train_data)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    losses = estimate_loss(model, get_batch, train_data, val_data, cfg)
    return losses["val"]


def main():
    cfg = Config()
    torch