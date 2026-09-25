markdown
# Miniature LLM

A complete, transparent implementation of a small decoder-only Transformer
language model, built from scratch with PyTorch. Companion code for
Chapter 3 of *The Principles of Large Language Models and Their Construction*.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Build the corpus:

bash
python data/build_corpus.py
Train the model:

bash
python train.py
Generate samples:

bash
python generate.py --prompt "虎" --max_new_tokens 30
Run a verification experiment:

bash
python experiments/exp_a_sentence_strength.py