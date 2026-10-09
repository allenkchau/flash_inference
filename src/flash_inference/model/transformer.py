import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig
from flash_inference.model.rope import RotaryEmbedding
from flash_inference.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            
        cos, sin = self.rotary_emb(seq_len)

        x = self.token_embeddings(input_ids)  # shape: (batch_size, seq_len, hidden_dim)
        for layer in self.blocks:
            x = layer(x, cos, sin)
        out = self.output_proj()    # shape: (batch_size, seq_len, )
        return out
