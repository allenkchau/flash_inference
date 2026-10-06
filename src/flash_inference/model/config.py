from dataclasses import dataclass
from typing import Optional


from transformers import AutoConfig


@dataclass
class ModelConfig:
    # primary tensor dimensions
    hidden_dim: int
    vocab_size: int
    intermediate_dim: int
    n_layers: int
    n_heads: int

    # numerical and positional hyperparameters
    max_seq_len: int
    rms_norm_eps: float
    rope_theta: float

    # attention and architecture variants
    n_kv_heads: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None

    # biases usually default to false in most modern models
    attn_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        # fallback to stardard MHAttention if GQAttention keys are absent
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        else:
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
                )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @classmethod
    def from_huggingface(cls, model_name: str):
        hf_config = AutoConfig.from_pretrained(model_name)

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            intermediate_dim=hf_config.intermediate_size,
            max_seq_len=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            attn_bias=getattr(hf_config, "attention_bias", False),
            mlp_bias=getattr(hf_config, "mlp_bias", False),
        )
