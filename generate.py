"""
generate.py
===========
Generate text from a trained miniature LLM.

Output is shown in both Chinese and English (via character-level gloss).

Run:
    python generate.py --prompt "虎" --max_new_tokens 30
"""

import argparse
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model.tokenizer import CharTokenizer
from model.mini_llm import MiniLLM
from model.glossary import gloss_text


def load_model(device):
    cfg = Config()
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
    return model, tokenizer, saved_cfg


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50,
             temperature=0.8, top_k=20, block_size=64, device="cpu"):
    if prompt:
        idx = torch.tensor([tokenizer.encode(prompt)],
                           dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="虎")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, saved_cfg = load_model(device)

    print(f"Prompt (Chinese): {args.prompt}")
    print(f"Prompt (English): {gloss_text(args.prompt)}")
    print("-" * 60)

    for i in range(5):
        output = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            block_size=saved_cfg["block_size"],
            device=device,
        )
        print(f"Sample {i + 1}:")
        print(f"Chinese:")
        print(f"  {output}")
        print(f"English:")
        print(f"  {gloss_text(output)}")
        print("=" * 60)

if __name__ == "__main__":
    main()