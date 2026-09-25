"""
train.py
========
Training script for the miniature LLM.

Run:
    python train.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import os
import time

import torch

from config import Config
from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM


def prepare_data(text, tokenizer, block_size, split=0.9):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * split)
    return data[:n], data[n:]


def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.learning_rate * 0.1
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.learning_rate * 0.1 + coeff * (
        cfg.learning_rate - cfg.learning_rate * 0.1
    )


def make_get_batch(cfg):
    def get_batch(data):
        ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
        return x, y
    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, train_data, val_data, cfg):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data)
            x, y = x.to(cfg.device), y.to(cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # Load corpus
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Corpus not found at {cfg.data_path}. "
            "Run 'python data/build_corpus.py' first."
        )
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    train_data, val_data = prepare_data(text, tokenizer, cfg.block_size)
    get_batch = make_get_batch(cfg)

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(cfg.device)

    print(f"Device: {cfg.device}")
    print(f"Parameters: {model.num_params():,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(train_data):,}")
    print("-" * 50)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    start = time.time()
    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, get_batch, train_data, val_data, cfg)
            print(
                f"iter {it:4d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.2f} | "
                f"lr {lr:.2e} | {time.time()-start:.0f}s"
            )

        x, y = get_batch(train_data)
        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    # Save the model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
    }, "checkpoints/mini_llm.pt")
    tokenizer.save("checkpoints/tokenizer.json")
    print("Saved model to checkpoints/mini_llm.pt")


if __name__ == "__main__":
    main()