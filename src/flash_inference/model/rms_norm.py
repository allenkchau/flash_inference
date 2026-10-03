import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, rms_norm_eps: float):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dim = hidden_dim

        # gamma parameter
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_x = torch.sqrt(
            ((x**2).sum(dim=-1, keepdim=True) / self.hidden_dim + self.rms_norm_eps)
        )
        out = (x / rms_x) * self.weight
        return out
