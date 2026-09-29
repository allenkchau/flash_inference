from flash_inference.model.config import ModelConfig
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Llama family uses a SwiGLU MLP
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # projection matrices
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=config.mlp_bias)
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=config.mlp_bias)

        # SiLU activation
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
