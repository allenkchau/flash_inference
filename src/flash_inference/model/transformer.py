import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList([])

        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        x = self.token_embeddings(input_ids)  # shape: ()
        for layer in self.blocks:
            x = self.layer
        out = self.output_proj()
        return out
