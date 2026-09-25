"""
data/build_corpus.py
====================
Generates the Mini Chinese Language System (MCS) corpus for training the
miniature LLM. Writes the result to data/mcs_corpus.txt.

Run directly:
    python data/build_corpus.py
"""

import os
import random


# Vocabulary of the MCS ecosystem
PREDATORS = ["虎", "狐", "狗"]
PREY = ["鸡", "羊", "牛", "虫", "狐", "狗"]
HERBIVORES = ["牛", "羊", "虫", "鸡"]
PLANTS = ["草", "麦"]
FOOD = ["肉"]
VERBS = ["吃", "追", "赶", "抓"]
ADJECTIVES = ["肥", "饿"]


def generate_corpus(n_sentences=4500, seed=1337):
    """Generate a list of sentences describing the MCS ecosystem."""
    random.seed(seed)
    sentences = []

    # 1. Predator-prey sentences
    for _ in range(int(n_sentences * 0.7)):
        s = random.choice(PREDATORS)
        v = random.choice(VERBS)
        o = random.choice(PREY)
        if random.random() < 0.3:
            s = random.choice(ADJECTIVES) + s
        if random.random() < 0.3:
            o = random.choice(ADJECTIVES) + o
        sentences.append(s + v + o + "。")

    # 2. Herbivore-plant sentences
    for _ in range(int(n_sentences * 0.25)):
        s = random.choice(HERBIVORES)
        v = "吃"
        o = random.choice(PLANTS)
        if random.random() < 0.3:
            s = random.choice(ADJECTIVES) + s
        if random.random() < 0.3:
            o = random.choice(ADJECTIVES) + o
        sentences.append(s + v + o + "。")

    # 3. Simple descriptive sentences
    for _ in range(int(n_sentences * 0.05)):
        s = random.choice(PREDATORS + PREY + PLANTS + FOOD)
        adj = random.choice(ADJECTIVES)
        sentences.append(s + adj + "。")

    random.shuffle(sentences)
    return sentences


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "mcs_corpus.txt")
    sentences = generate_corpus()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    print(f"Wrote {len(sentences)} sentences to {out_path}")


if __name__ == "__main__":
    main()