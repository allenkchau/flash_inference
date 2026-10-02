import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig
from flash_inference.model.rope import Rope

class GQAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # learned key, query, value matrices (shape: hidden_dim, )
        self.Wq = nn.Linear(config.hidden_dim, bias=config.attn_bias)
        self.Wk = nn.Linear(config.hidden_dim, bias=config.attn_bias)
        self.Wv = nn.Linear(config.hidden_dim, bias=config.attn_bias)

        # dimension of the  key vectors
        self.d_k = config.hidden_dim / config.n_heads



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get query, key, value projections of our input
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # calculate attention scores
        attn_scores = torch.matmul(Q, K.T)

        # divide by d_k for stability
        scaled_attn_scores = attn_scores / torch.sqrt(self.d_k)

        # softmax
        normalized_attn_scores = torch.softmax(scaled_attn_scores, dim=)

        normalized_attn_scores





