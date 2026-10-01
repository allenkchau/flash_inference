import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig

class GQAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # learned key, query, value matrices
        self.Wk = 
        self.Wv = 
        self.Wq = 

        # 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.Wk()
        
