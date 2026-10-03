import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from flash_inference.model.config import ModelConfig
from flash_inference.model.mlp import MLP


@pytest.fixture
def smollm_args():
    """Provides a ModelConfig fixture matching SmolLM-135M dimensions."""
    return ModelConfig(
        hidden_dim=576,
        n_layers=30,
        n_heads=9,
        vocab_size=49152,
        intermediate_dim=1536,
        max_seq_len=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_mlp_output_shape(smollm_args):
    """Verify input-to-output shape consistency for arbitrary batch sizes and sequence lengths."""
    mlp = MLP(smollm_args)
    # change to inference mode - this affects behavior of some layers like dropout and batchnorm
    mlp.eval()

    batch_size = 2
    seq_len = 8

    # Shape: (batch_size, seq_len, hidden_dim)
    x = torch.randn(batch_size, seq_len, smollm_args.hidden_dim)

    with torch.no_grad():
        out = mlp(x)

    assert out.shape == (
        batch_size,
        seq_len,
        smollm_args.hidden_dim,
    ), (
        f"Expected shape {(batch_size, seq_len, smollm_args.hidden_dim)}, got"
        f" {out.shape}"
    )


def test_mlp_has_no_bias(smollm_args):
    """Verify that all linear projections in the MLP omit bias parameters."""
    mlp = MLP(smollm_args)

    assert mlp.gate_proj.bias is None, "gate_proj should not have a bias term"
    assert mlp.up_proj.bias is None, "up_proj should not have a bias term"
    assert mlp.down_proj.bias is None, "down_proj should not have a bias term"


def test_mlp_hf_numerical_parity(smollm_args):
    """Golden Reference Test: Verify that custom MLP output matches Hugging Face's

    LlamaMLP when initialized with identical weights and inputs.
    """
    torch.manual_seed(42)

    # 1. Instantiate custom MLP
    custom_mlp = MLP(smollm_args)
    custom_mlp.eval()

    # 2. Instantiate Hugging Face reference LlamaMLP with identical dimensions
    hf_config = LlamaConfig(
        hidden_size=smollm_args.hidden_dim,
        intermediate_size=smollm_args.intermediate_dim,
        hidden_act="silu",
        mlp_bias=False,
    )
    hf_mlp = LlamaMLP(hf_config)
    hf_mlp.eval()

    # 3. Synchronize weights: copy custom MLP weights into HF MLP
    with torch.no_grad():
        hf_mlp.gate_proj.weight.copy_(custom_mlp.gate_proj.weight)
        hf_mlp.up_proj.weight.copy_(custom_mlp.up_proj.weight)
        hf_mlp.down_proj.weight.copy_(custom_mlp.down_proj.weight)

    # 4. Create a deterministic synthetic input tensor
    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, smollm_args.hidden_dim, dtype=torch.float32)

    # 5. Run forward passes
    with torch.no_grad():
        custom_out = custom_mlp(x)
        hf_out = hf_mlp(x)

    # 6. Assert numerical equivalence down to standard float32 precision
    torch.testing.assert_close(
        custom_out,
        hf_out,
        atol=1e-5,
        rtol=1e-5,
        msg="Custom MLP outputs do not match Hugging Face reference outputs!",
    )
