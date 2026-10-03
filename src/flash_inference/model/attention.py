import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig
from flash_inference.model.rope import Rope

class GQAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # learned key, query, value matrices
        self.Wq = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=config.attn_bias)
        self.Wk = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=config.attn_bias)
        self.Wv = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=config.attn_bias)

        # dimension of the key vectors
        self.d_k = config.hidden_dim / config.n_heads

        # causal attention mask
        # even though we aren't training a full LLM from scratch, this is required during the prefill phase of inference
        self.causal_mask = 

        # final output matrix that captures info from all the attn heads
        self.Wo = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=config.attn_bias)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get query, key, value projections of our input
        # input shape: (batch, seq_len, hidden_dim)
        Q = self.Wq(x)  # shape: (batch_size, seq_len, n_heads * head_dim)
        K = self.Wk(x)  # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        V = self.Wv(x)  # shape: (batch_size, seq_len, n_kv_heads * head_dim)

        # calculate attention scores
        attn_scores = torch.matmul(Q, K.T)  # shape: (batch_size, )

        # divide by d_k for stability
        scaled_attn_scores = attn_scores / torch.sqrt(self.d_k)

        # softmax
        normalized_attn_scores = torch.softmax(scaled_attn_scores, dim=)

        normalized_attn_scores

        self.Wo
        





