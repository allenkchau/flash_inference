import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        self.eps = config.rms_norm_eps
        self.gamma = nn.Parameter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqr( + self.eps)
        torch.()
        return 
