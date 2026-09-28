from flash_inference.model.config import ModelConfig
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        

    def forward(x: torch.Tensor) -> torch.Tensor:

