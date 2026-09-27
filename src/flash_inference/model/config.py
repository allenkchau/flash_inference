from dataclasses import dataclass


@dataclass
class ModelArgs:
    hidden_dim: int
    vocab_size: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    mlp_bias: bool

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @classmethod
    def from_json(cls, ):
        return cls()

