import torch
import torch.nn as nn

from flash_inference.model.config import ModelConfig


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # here we generate cos and sin angles for the max_seq_len once at the beginning of the transformer
        positions = torch.arange(config.max_seq_len)    # shape: (max_seq_len)

        base = torch.tensor(config.rope_theta)
        dim_indices = torch.arange(0, config.head_dim, 2)   # shape: (head_dim / 2)
        freqs = 1.0 / torch.pow(base, dim_indices / config.head_dim)    # shape: (head_dim / 2)

        # we want to combine the positions with the freqs
        angles = torch.matmul(positions.unsqueeze(-1), freqs.unsqueeze(0))    # shape: (max_seq_len, head_dim / 2)

        # now we have each freq accounted for but there should be 2 coords per freq
        # in the original paper, x and y are interleaved with each other so (0, 1) are fastest and paired together and (62, 63) are slowest
        # however, llama puts all the x's in the first half and all the y's in the second half
        angles = torch.cat([angles, angles], dim=-1)    # shape: (max_seq_len, head_dim)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # save the sin and cos tables so they move to the same device as our model
        self.register_buffer("sin", sin, persistent=False)  # set persistent to false because the tables can be recomputed and don't need to bloat the checkpoint weights file
        self.register_buffer("cos", cos, persistent=False)

    # the forward pass gets a slice of the cos and sin tables that we generated in __init__
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids has shape: (batch_size, seq_len)
        # when indexing Tensor[IndexTensor], the dimensions of IndexTensor replace the indexed dimension; the remaining unindexed dimensions are preserved at the end
        # new shape is shape: (batch_size, seq_len, head_dim)
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]
        return cos.unsqueeze(1), sin.unsqueeze(1)   # shape: (batch_size, 1(heads broadcasting), seq_len, head_dim)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Takes cos and sin angles and applies them to actual Q and K vectors
    """

    return 

