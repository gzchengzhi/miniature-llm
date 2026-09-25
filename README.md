# Miniature LLM

Companion code for Chapter 3 of *The Principles of Large Language Models and 
Their Construction: A Comparative Study Based on Linguistics and Cognitive 
Psychology*.

A complete, transparent implementation of a small decoder-only Transformer 
language model, built from scratch with PyTorch. The model trains in minutes 
on an ordinary personal computer and is designed to be read line by line.

---

## Overview

This repository accompanies Chapter 3 of the book. It provides:

- A fully transparent implementation of a decoder-only Transformer
- A generated training corpus based on the Mini Chinese Language System (MCS)
- Training and generation scripts that run on a personal computer
- Four verification experiments that compare cognitive models of language 
  with the miniature LLM's behavior

The model has approximately **1.5 million parameters** in its minimal 
configuration. It trains in **5–10 minutes on CPU** or **1–3 minutes on a GPU**.

---

## Repository Structure
miniature-llm/
├── README.md # This file
├── requirements.txt # Python dependencies
├── config.py # Central configuration
├── train.py # Training script
├── generate.py # Text generation
├── data/
│ └── build_corpus.py # Generates the MCS training corpus
├── model/
│ ├── init.py
│ ├── tokenizer.py # Character-level tokenizer
│ ├── attention.py # Causal self-attention
│ ├── transformer.py # Transformer block and feed-forward network
│ ├── mini_llm.py # Full model
│ └── glossary.py # Chinese-to-English character gloss
├── experiments/
│ ├── init.py
│ ├── exp_a_sentence_strength.py
│ ├── exp_b_semantic_distance.py
│ ├── exp_c_construction_frequency.py
│ └── exp_d_learning_curve.py
└── notebooks/
└── walkthrough.ipynb # Interactive tutorial (optional)

text

---

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/gzchengzhi/miniature-llm.git
cd miniature-llm
Step 2: Install dependencies
bash
pip install -r requirements.txt
Windows users: if you see the following error when running any script:

text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll 
already initialized.
add the following two lines at the top of train.py and generate.py,
before any other imports:

python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
This is a known issue on Windows caused by multiple OpenMP runtimes being
loaded. The workaround is safe for the purposes of this repository.

Step 3: Build the training corpus
bash
python data/build_corpus.py
This generates data/mcs_corpus.txt, containing approximately 4,500
sentences from the Mini Chinese Language System.

Step 4: Train the model
bash
python train.py
Training takes 5–10 minutes on CPU, or 1–3 minutes on a GPU. The trained
model is saved to checkpoints/mini_llm.pt, and the tokenizer is saved to
checkpoints/tokenizer.json.

You will see progress output like:

text
Device: cpu
Parameters: 1,517,440
Vocabulary size: 15
Training tokens: 27,000
--------------------------------------------------
iter    0 | train 4.1523 | val 4.1489 | ppl 63.345 | lr 3.00e-05 | 0s
iter  200 | train 1.6547 | val 1.7012 | ppl 5.480  | lr 2.71e-04 | 8s
...
Step 5: Generate text
bash
python generate.py --prompt "虎" --max_new_tokens 30
The output shows both Chinese and English (via character-level gloss):

text
Prompt (Chinese): 虎
Prompt (English): tiger
============================================================
Sample 1:
Chinese:
  虎吃羊。狐追狗。鸡吃虫。
English:
  tiger eat sheep . fox chase dog . chicken eat worm .
============================================================
Step 6: Run verification experiments
bash
python experiments/exp_a_sentence_strength.py
python experiments/exp_b_semantic_distance.py
python experiments/exp_c_construction_frequency.py
python experiments/exp_d_learning_curve.py
Each experiment prints a table comparing a cognitive model's predictions
with the miniature LLM's behavior.

What the Experiments Show
Experiment	Cognitive Model	What It Tests
A	Conceptual triangle	Sentence strength vs. model perplexity
B	Spreading activation	Semantic distance vs. representation similarity
C	Construction grammar	Construction frequency vs. perplexity
D	Language acquisition	Learning curve shape vs. child development
Experiment A: Sentence Strength
The conceptual triangle model assigns each sentence a sentence strength
based on how well its constituents are connected through attribute transfer.
The experiment correlates sentence strength with the miniature LLM's
perplexity on the same sentences. A negative correlation indicates that
sentences judged as more meaningful by the cognitive model are easier for the
LLM to predict.

Experiment B: Semantic Distance
The spreading activation model assigns a semantic distance to each pair
of words based on their distance in a semantic network. The experiment
correlates this distance with the cosine similarity of the LLM's internal
representations for the same word pairs.

Experiment C: Construction Frequency
Usage-based construction grammar predicts that high-frequency constructions
are processed more fluently. The experiment correlates construction frequency
in the training corpus with the LLM's average perplexity on sentences
instantiating each construction.

Experiment D: Learning Curve
Child language acquisition proceeds through distinct developmental stages.
The experiment trains the LLM on subsets of the corpus of increasing size
and records validation perplexity for each subset, revealing the shape of the
learning curve.

Configuration
All hyperparameters are defined in config.py. The key parameters are:

Parameter	Default	Description
block_size	64	Context length in tokens
n_layer	4	Number of Transformer blocks
n_head	4	Number of attention heads
n_embd	128	Embedding dimension
batch_size	32	Training batch size
learning_rate	3e-4	Peak learning rate
max_iters	2000	Number of training iterations
temperature	0.8	Sampling temperature for generation
top_k	20	Top-k sampling for generation
To increase the model's capacity, increase n_layer, n_head, and n_embd.
A standard configuration (n_layer=6, n_head=6, n_embd=384,
block_size=256) yields approximately 10.6 million parameters and trains in
under an hour on a GPU.

Hardware Requirements
Component	Minimum	Recommended
Python	3.8+	3.10+
RAM	8 GB	16 GB
GPU	Not required	2 GB+ VRAM
Disk space	500 MB	1 GB
Operating system	Windows, macOS, Linux	Any
The model runs on CPU without any specialized hardware. A GPU speeds up
training but is not required.

Troubleshooting
OMP: Error #15 on Windows
Add os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' at the top of train.py
and generate.py, before importing PyTorch. See Step 2 above.

SyntaxError: invalid syntax in config.py
If line 2 of any file contains ---, remove it. This is a Markdown
separator that should not appear in Python source files.

ModuleNotFoundError: No module named 'torch'
Install PyTorch:

bash
pip install torch numpy
FileNotFoundError: Corpus not found
Run the corpus generator first:

bash
python data/build_corpus.py
FileNotFoundError: checkpoints/mini_llm.pt
Train the model first:

bash
python train.py
git branch shows error: cannot spawn less
Disable Git's pager:

bash
git config --global core.pager cat
Reproducing Book Results
To reproduce the exact results reported in Chapter 3:

Use the default configuration in config.py.

Set seed = 1337 (already the default).

Run python train.py to completion.

Run each experiment script in experiments/.

Small variations in results may occur due to hardware differences and
non-determinism in GPU operations. The qualitative patterns reported in the
book should be robust across runs.

Extending the Repository
The code is designed to be modified. Common extensions:

Change the vocabulary. Modify data/build_corpus.py to generate a
different corpus. The tokenizer adapts automatically to any character set.

Add experiments. Create a new script in experiments/. Import
MiniLLM, CharTokenizer, and the relevant helpers from model/ and
train.py.

Train a larger model. Increase n_layer, n_head, and n_embd in
config.py. Expect longer training times and larger memory requirements.

Add tools or agents. The generate.py script can be extended into a
simple agent by adding a tool-calling loop, as described in Chapter 6.

Related Chapters
This repository is part of a larger book project. Related repositories:

Chapter 4: Qwen3-4B local deployment (see the book for details)

Chapter 6: Minimal reasoning agent

Chapter 7: Multi-agent collaboration

Chapter 8: Multi-agent mathematical problem solving

Citation
If you use this code in academic work, please cite:

Cheng, Z. (2026). The Principles of Large Language Models and Their
Construction: A Comparative Study Based on Linguistics and Cognitive
Psychology.

License
This project is released under the MIT License. You are free to use, modify,
and distribute the code for any purpose, including commercial use, provided
that the original copyright notice is retained.

Contact
For questions, corrections, or suggestions, please open an issue on GitHub:

https://github.com/gzchengzhi/miniature-llm/issues

Acknowledgments
The architecture of this miniature LLM is inspired by Andrej Karpathy's
NanoGPT and by the original Transformer paper (Vaswani et al., 2017). The
Mini Chinese Language System corpus is based on the conceptual triangle model
proposed in the author's earlier work on computational linguistics.
