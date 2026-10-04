import math
import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig
#from flash_inference.model.rope import Rope

class GQAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # learned key, query, value matrices
        self.q_proj = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=config.attn_bias)
        self.k_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=config.attn_bias)
        self.v_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=config.attn_bias)

        # dimensions we need in the forward pass
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads

        # causal attention mask
        # even though we aren't training a full LLM from scratch, this is required during the prefill phase of inference
        mask = torch.tril(torch.ones(1, 1, 1, config.max_seq_len, config.max_seq_len))      # shape: (batch_size, n_kv_heads, group_size, seq_len, seq_len)
        # register buffer is a tensor attached to a module that does not reveive gradient updates; it os just saved and loaded inside the model's state_dict()
        self.register_buffer("causal_mask", mask)

        # final output matrix that captures info from all the attn heads
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=config.attn_bias)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get query, key, value projections of our input
        # input shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)  # shape: (batch_size, seq_len, n_heads * head_dim)
        K = self.k_proj(x)  # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        V = self.v_proj(x)  # shape: (batch_size, seq_len, n_kv_heads * head_dim)

        # at this point for a single token, all heads are stuck together side by side in one flat n_heads * head_dim dimensional vector
        # we want to separate the n_heads * head_dim numbers into n_head distinct heads of head_dim numbers each
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # now we have separate data for each head
        # tokens are in dim 1 and heads are in dim 2
        # we want within each individual head for every token to look at every other token; we want an attn matrix of size (seq_len, seq_len)
        # if we multiply Q and K.T as is we would be multiplying across heads not across tokens so we have to swap dims 1 and 2
        Q = Q.transpose(1, 2)   # shape: (batch_size, n_heads, seq_len, head_dim)
        K = K.transpose(1, 2)   # shape: (batch_size, n_kv_heads, seq_len, head_dim)
        V = V.transpose(1, 2)   # shape: (batch_size, n_kv_heads, seq_len, head_dim)

        # now to multiply K and Q we need to split the query heads into groups
        Q = Q.reshape(batch_size, self.n_kv_heads, self.group_size, seq_len, self.head_dim)

        # add group dimension to K and V
        K = K.unsqueeze(2)   # shape: (batch_size, n_kv_heads, group_size(1), seq_len, head_dim)
        V = V.unsqueeze(2)   # shape: (batch_size, n_kv_heads, group_size(1), seq_len, head_dim)

        # calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # shape: (batch_size, n_kv_heads, group_size, seq_len, seq_len)

        # divide by d_k for stability
        scaled_attn_scores = attn_scores / math.sqrt(self.head_dim)    # shape: (batch_size, n_kv_heads, group_size, seq_len, seq_len)

        # build the mask for the sequence
        curr_mask = self.causal_mask[:, :, :, :seq_len, :seq_len]

        # apply mask to the scores, and then softmax
        scores = scaled_attn_scores.masked_fill(curr_mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)    # we normalize the values across the columns of our attn score matrix

        out = torch.matmul(attn_weights, V)     # shape: (batch_size, n_kv_heads, group_size, seq_len, head_dim)

        # move seq_len to position 1 after batch_size
        out = out.permute(0, 3, 1, 2, 4)    # shape: (batch_size, seq_len, n_kv_heads, group_size, head_dim)
        # tranpose and reshape output back to 
        out = out.reshape(batch_size, seq_len, self.n_heads * self.head_dim)   

        # concatenate results from all attn heads and feed it into final linear layer
        out = self.o_proj(out)      # shape: (batch_size, seq_len, hidden_dim)
        return out






