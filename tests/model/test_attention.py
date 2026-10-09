import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)

from flash_inference.model.attention import GQAttention
from flash_inference.model.config import ModelConfig


@pytest.fixture
def smollm_config():
    """ModelConfig fixture matching SmolLM-135M dimensions:

    hidden_dim = 576, n_heads = 9, n_kv_heads = 3, head_dim = 64
    """
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
        attn_bias=False,
    )


@pytest.fixture
def hf_llama_config(smollm_config):
    """Matching Hugging Face LlamaConfig for reference validation."""
    return LlamaConfig(
        hidden_size=smollm_config.hidden_dim,
        num_attention_heads=smollm_config.n_heads,
        num_key_value_heads=smollm_config.n_kv_heads,
        max_position_embeddings=smollm_config.max_seq_len,
        rms_norm_eps=smollm_config.rms_norm_eps,
        rope_theta=smollm_config.rope_theta,
        attention_bias=False,
        _attn_implementation="eager",
    )


def build_rotary_embeddings(hf_config, x: torch.Tensor):
    """Helper to compute real (cos, sin) tensors matching the input shape."""
    rotary_emb = LlamaRotaryEmbedding(config=hf_config)
    seq_len = x.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
    cos, sin = rotary_emb(x, position_ids)
    return cos, sin


def test_attention_projections_and_shapes(smollm_config):
    """Verify that GQA linear projections have correct weight matrix dimensions.

    - q_proj: (hidden_dim, n_heads * head_dim) = (576, 9 * 64 = 576)
    - k_proj: (hidden_dim, n_kv_heads * head_dim) = (576, 3 * 64 = 192)
    - v_proj: (hidden_dim, n_kv_heads * head_dim) = (576, 3 * 64 = 192)
    - o_proj: (n_heads * head_dim, hidden_dim) = (576, 576)
    """
    attn = GQAttention(smollm_config)
    attn.eval()

    h_dim = smollm_config.hidden_dim
    q_dim = smollm_config.n_heads * smollm_config.head_dim
    kv_dim = smollm_config.n_kv_heads * smollm_config.head_dim

    assert attn.q_proj.weight.shape == (q_dim, h_dim)
    assert attn.k_proj.weight.shape == (kv_dim, h_dim)
    assert attn.v_proj.weight.shape == (kv_dim, h_dim)
    assert attn.o_proj.weight.shape == (h_dim, q_dim)

    if hasattr(attn.q_proj, "bias") and attn.q_proj.bias is not None:
        pytest.fail("Attention projections should not have bias terms for SmolLM.")


def test_attention_forward_output_shape(smollm_config, hf_llama_config):
    """Verify output tensor maintains (batch_size, seq_len, hidden_dim) with RoPE applied."""
    attn = GQAttention(smollm_config)
    attn.eval()

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, smollm_config.hidden_dim)

    cos, sin = build_rotary_embeddings(hf_llama_config, x)

    with torch.no_grad():
        out = attn(x, cos=cos, sin=sin)

    assert out.shape == (
        batch_size,
        seq_len,
        smollm_config.hidden_dim,
    ), (
        f"Expected shape {(batch_size, seq_len, smollm_config.hidden_dim)}, got {out.shape}"
    )


def test_attention_causal_masking(smollm_config, hf_llama_config):
    """Causality test with RoPE active:

    Changes to future tokens in the input must NOT alter the outputs
    of preceding tokens.
    """
    attn = GQAttention(smollm_config)
    attn.eval()

    batch_size = 1
    seq_len = 8
    x = torch.randn(batch_size, seq_len, smollm_config.hidden_dim)

    cos, sin = build_rotary_embeddings(hf_llama_config, x)

    with torch.no_grad():
        out1 = attn(x, cos=cos, sin=sin)

        # Perturb the last token position
        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += torch.randn_like(x_perturbed[:, -1, :]) * 10.0

        out2 = attn(x_perturbed, cos=cos, sin=sin)

    # Check that tokens at indices [0, seq_len - 2] are completely unchanged
    torch.testing.assert_close(
        out1[:, :-1, :],
        out2[:, :-1, :],
        atol=1e-6,
        rtol=1e-6,
        msg="Attention violated causal property! Future tokens affected past outputs.",
    )

    # The last token output SHOULD be different
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :]), (
        "Perturbed token output remained identical!"
    )


def test_attention_hf_numerical_parity(smollm_config, hf_llama_config):
    """Golden Reference Test:

    Verify end-to-end numerical parity against Hugging Face's LlamaAttention
    with real RoPE positions, causal masking, and GQA grouping.
    """
    torch.manual_seed(42)

    # 1. Custom Attention
    custom_attn = GQAttention(smollm_config)
    custom_attn.eval()

    # 2. Hugging Face LlamaAttention setup
    hf_attn = LlamaAttention(config=hf_llama_config, layer_idx=0)
    hf_attn.eval()

    # 3. Synchronize projection weights
    with torch.no_grad():
        hf_attn.q_proj.weight.copy_(custom_attn.q_proj.weight)
        hf_attn.k_proj.weight.copy_(custom_attn.k_proj.weight)
        hf_attn.v_proj.weight.copy_(custom_attn.v_proj.weight)
        hf_attn.o_proj.weight.copy_(custom_attn.o_proj.weight)

    # 4. Input setup
    batch_size = 2
    seq_len = 12
    x = torch.randn(batch_size, seq_len, smollm_config.hidden_dim, dtype=torch.float32)

    # 5. Real rotary position embeddings (cos, sin)
    cos, sin = build_rotary_embeddings(hf_llama_config, x)

    # 6. Build standard HF 4D Causal Attention Mask
    # Shape: (batch_size, 1, seq_len, seq_len)
    min_val = torch.finfo(torch.float32).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=min_val)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len).expand(
        batch_size, 1, seq_len, seq_len
    )

    # 7. Forward passes
    with torch.no_grad():
        custom_out = custom_attn(x, cos=cos, sin=sin)
        hf_out = hf_attn(
            x,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
        )[0]

    # 8. Assert exact numerical equivalence
    torch.testing.assert_close(
        custom_out,
        hf_out,
        atol=1e-5,
        rtol=1e-5,
        msg="Custom Attention with RoPE output deviates from Hugging Face reference!",
    )
