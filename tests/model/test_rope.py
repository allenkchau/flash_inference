import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from flash_inference.model.config import ModelConfig
from flash_inference.model.rope import (
    RotaryEmbedding,
    apply_rotary_emb,
)


@pytest.fixture
def smollm_config():
    return ModelConfig(
        hidden_dim=576,
        n_layers=30,
        n_heads=9,
        n_kv_heads=3,
        vocab_size=49152,
        intermediate_dim=1536,
        max_seq_len=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_rope_output_shape(smollm_config):
    """Verify that applying RoPE does not alter the input tensor shapes."""
    rope = RotaryEmbedding(smollm_config)
    batch_size = 2
    n_heads = smollm_config.n_heads
    seq_len = 16
    head_dim = smollm_config.head_dim

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    positions = torch.arange(seq_len)

    cos, sin = rope(positions)
    q_rot = apply_rotary_emb(q, cos, sin)

    assert q_rot.shape == q.shape, f"Expected shape {q.shape}, but got {q_rot.shape}"


def test_rope_position_zero_identity(smollm_config):
    """At position 0, angle is 0.

    Rotation by 0 radians is an identity transformation: R_0(x) == x.
    """
    rope = RotaryEmbedding(smollm_config)
    batch_size = 1
    n_heads = smollm_config.n_heads
    head_dim = smollm_config.head_dim

    x = torch.randn(batch_size, n_heads, 1, head_dim)
    positions = torch.tensor([0])

    cos, sin = rope(positions)
    x_rot = apply_rotary_emb(x, cos, sin)

    torch.testing.assert_close(
        x_rot,
        x,
        atol=1e-6,
        rtol=1e-6,
        msg="RoPE at position 0 must act as an identity transformation!",
    )


def test_rope_norm_preservation(smollm_config):
    """Because RoPE applies orthogonal 2D rotations,

    it must preserve vector L2 norms for every token and head.
    """
    rope = RotaryEmbedding(smollm_config)
    batch_size = 2
    n_heads = smollm_config.n_heads
    seq_len = 32
    head_dim = smollm_config.head_dim

    x = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float64)
    positions = torch.arange(seq_len)

    cos, sin = rope(positions)
    # Ensure cos/sin match float64 for precision testing
    x_rot = apply_rotary_emb(x, cos.to(torch.float64), sin.to(torch.float64))

    norm_before = torch.linalg.norm(x, dim=-1)
    norm_after = torch.linalg.norm(x_rot, dim=-1)

    torch.testing.assert_close(
        norm_after,
        norm_before,
        atol=1e-5,
        rtol=1e-5,
        msg="RoPE violated norm preservation! Rotations must preserve length.",
    )


def test_rope_relative_shift_invariance(smollm_config):
    """Core RoPE mathematical identity:

    <R_m(q), R_n(k)> == <R_{m+s}(q), R_{n+s}(k)>.
    """
    rope = RotaryEmbedding(smollm_config)
    head_dim = smollm_config.head_dim

    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)

    pos_m, pos_n, shift = 5, 2, 10

    # Pair 1: positions (5, 2)
    cos_m, sin_m = rope(torch.tensor([pos_m]))
    cos_n, sin_n = rope(torch.tensor([pos_n]))
    q_m = apply_rotary_emb(q, cos_m.to(torch.float64), sin_m.to(torch.float64))
    k_n = apply_rotary_emb(k, cos_n.to(torch.float64), sin_n.to(torch.float64))
    dot_product_1 = (q_m * k_n).sum()

    # Pair 2: shifted positions (15, 12)
    cos_ms, sin_ms = rope(torch.tensor([pos_m + shift]))
    cos_ns, sin_ns = rope(torch.tensor([pos_n + shift]))
    q_shifted = apply_rotary_emb(q, cos_ms.to(torch.float64), sin_ms.to(torch.float64))
    k_shifted = apply_rotary_emb(k, cos_ns.to(torch.float64), sin_ns.to(torch.float64))
    dot_product_2 = (q_shifted * k_shifted).sum()

    torch.testing.assert_close(
        dot_product_1,
        dot_product_2,
        atol=1e-5,
        rtol=1e-5,
        msg="RoPE dot product is not invariant under equal translation shifts!",
    )


def test_rope_hf_numerical_parity(smollm_config):
    """Golden Reference Test:

    Verify custom RoPE output against Hugging Face's official apply_rotary_pos_emb.
    """
    torch.manual_seed(42)

    batch_size = 2
    n_heads = smollm_config.n_heads
    seq_len = 16
    head_dim = smollm_config.head_dim

    # 1. Hugging Face LLaMA RoPE setup
    hf_config = LlamaConfig(
        hidden_size=smollm_config.hidden_dim,
        num_attention_heads=smollm_config.n_heads,
        num_key_value_heads=smollm_config.n_kv_heads,
        max_position_embeddings=smollm_config.max_seq_len,
        rope_theta=smollm_config.rope_theta,
    )
    hf_rope = LlamaRotaryEmbedding(config=hf_config)

    # Synthetic Q and K tensors: (batch, n_heads, seq_len, head_dim)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(
        batch_size,
        smollm_config.n_kv_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
    )

    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    print(f"Position Ids shape: {position_ids.shape}")

    # 2. Hugging Face reference forward
    cos_hf, sin_hf = hf_rope(q, position_ids)
    hf_q_rot, hf_k_rot = apply_rotary_pos_emb(q, k, cos_hf, sin_hf)

    # 3. Custom implementation forward
    custom_rope = RotaryEmbedding(smollm_config)
    cos_custom, sin_custom = custom_rope(position_ids[0])
    custom_q_rot = apply_rotary_emb(q, cos_custom, sin_custom)
    custom_k_rot = apply_rotary_emb(k, cos_custom, sin_custom)

    # 4. Assert parity
    torch.testing.assert_close(
        custom_q_rot,
        hf_q_rot,
        atol=1e-5,
        rtol=1e-5,
        msg="Custom rotated Queries do not match Hugging Face reference!",
    )
    torch.testing.assert_close(
        custom_k_rot,
        hf_k_rot,
        atol=1e-5,
        rtol=1e-5,
        msg="Custom rotated Keys do not match Hugging Face reference!",
    )
