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
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

        # SiLU activation
        self.activation = 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))
