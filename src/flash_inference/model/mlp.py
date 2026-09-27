from flash_inference.model.config import ModelArgs
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # mlp weights
        self.W1 = 
        self.activation = nn.GeLU()
        self.W2 = 

        # intermediate activation
        self.activation = 

    def forward(x: torch.Tensor) -> torch.Tensor:
        pass
