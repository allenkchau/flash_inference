from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    # primary tensor dimensions
    hidden_dim: int
    vocab_size: int
    intermediate_dim: int
    n_layers: int
    n_heads: int

    # attention and architecture variants
    n_kv_heads: Optional[int]
    mlp_bias: bool

    # numerical and positional hyperparameters
    max_seq_len: int

    def __post_init__(self):
        if self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @classmethod
    def from_json(cls, ):
        return cls()

