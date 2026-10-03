from flash_inference.model.attention import GQAttention
from flash_inference.model.config import ModelConfig
import torch
import torch.nn as nn

from flash_inference.model.mlp import MLP
from flash_inference.model.rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = GQAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.input_norm(x)
        attn_out = self.attn(norm_x)
        h = attn_out + x

        norm_h = self.post_attn_norm(h)
        mlp_out = self.mlp(norm_h)
        out = mlp_out + h
        return out
