## 文件 3：config.py

"""
config.py
=========
Central configuration for the miniature LLM.
"""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Data
    data_path: str = "data/mcs_corpus.txt"
    block_size: int = 64

    # Model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 100

    # Generation
    temperature: float = 0.8
    top_k: int = 20

    # System
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 1337

    def to_dict(self):
        return asdict(self)