import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from flash_inference.model.config import ModelConfig
from flash_inference.model.rms_norm import RMSNorm


@pytest.fixture
def smollm_config():
  """ModelConfig fixture matching SmolLM-135M dimensions."""
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


def test_rms_norm_output_shape(smollm_config):
  """Verify that RMSNorm preserves arbitrary tensor shapes and ranks."""
  dim = smollm_config.hidden_dim
  eps = smollm_config.rms_norm_eps
  norm = RMSNorm(hidden_dim=dim, rms_norm_eps=eps)

  # 2D: (batch_size, dim)
  x_2d = torch.randn(4, dim)
  assert norm(x_2d).shape == (4, dim)

  # 3D: (batch_size, seq_len, dim)
  x_3d = torch.randn(2, 8, dim)
  assert norm(x_3d).shape == (2, 8, dim)


def test_rms_norm_mathematical_identity(smollm_config):
  """Verify implementation matches the exact formula:

  y = (x / sqrt(mean(x^2) + eps)) * weight
  """
  torch.manual_seed(42)
  dim = smollm_config.hidden_dim
  eps = smollm_config.rms_norm_eps

  norm = RMSNorm(hidden_dim=dim, rms_norm_eps=eps)
  norm.eval()

  x = torch.randn(2, 4, dim, dtype=torch.float32)

  with torch.no_grad():
    actual_out = norm(x)

    # Reference calculation directly from first principles
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected_out = x * torch.rsqrt(variance + eps) * norm.weight
  print(f"Actual: {actual_out}")
  print(f"Expected: {expected_out}")

  torch.testing.assert_close(
      actual_out,
      expected_out,
      atol=1e-6,
      rtol=1e-6,
      msg="RMSNorm output does not match first-principles formula!",
  )


def test_rms_norm_scale_invariance(smollm_config):
  """RMSNorm is scale-invariant (modulo eps).

  Scaling the input by a positive constant c should produce an identical normalized
  output when epsilon is negligible.
  """
  dim = smollm_config.hidden_dim
  norm = RMSNorm(hidden_dim=dim, rms_norm_eps=1e-8)
  norm.eval()

  # Set weights to all ones to isolate the normalization factor
  with torch.no_grad():
    norm.weight.fill_(1.0)

  x = torch.randn(2, 4, dim) * 10.0
  c = 3.5

  with torch.no_grad():
    out_original = norm(x)
    out_scaled = norm(x * c)

  torch.testing.assert_close(
      out_original,
      out_scaled,
      atol=1e-5,
      rtol=1e-5,
      msg="RMSNorm must be scale-invariant for positive scalars!",
  )


def test_rms_norm_hf_numerical_parity(smollm_config):
  """Golden Reference Test:

  Verify numerical parity against Hugging Face's official LlamaRMSNorm.
  """
  torch.manual_seed(42)
  dim = smollm_config.hidden_dim
  eps = smollm_config.rms_norm_eps

  # 1. Custom RMSNorm
  custom_norm = RMSNorm(hidden_dim=dim, rms_norm_eps=eps)
  custom_norm.eval()

  # 2. Hugging Face LlamaRMSNorm
  hf_norm = LlamaRMSNorm(hidden_size=dim, eps=eps)
  hf_norm.eval()

  # 3. Synchronize weights
  with torch.no_grad():
    hf_norm.weight.copy_(custom_norm.weight)

  # 4. Input tensor
  x = torch.randn(2, 16, dim, dtype=torch.float32)

  # 5. Forward passes
  with torch.no_grad():
    custom_out = custom_norm(x)
    hf_out = hf_norm(x)

  # 6. Assert parity
  torch.testing.assert_close(
      custom_out,
      hf_out,
      atol=1e-6,
      rtol=1e-6,
      msg="Custom RMSNorm outputs deviate from Hugging Face reference!",
  )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_upcast_precision(smollm_config, dtype):
  """Verify that the layer handles half-precision inputs safely.

  In production inference, RMSNorm computes variance in float32 to avoid
  overflow before downcasting back to the target dtype.
  """
  dim = smollm_config.hidden_dim
  eps = smollm_config.rms_norm_eps

  norm = RMSNorm(hidden_dim=dim, rms_norm_eps=eps).to(dtype=dtype)
  norm.eval()

  x = torch.randn(2, 4, dim, dtype=dtype)

  with torch.no_grad():
    out = norm(x)

  assert out.dtype == dtype, (
      f"Expected output dtype {dtype}, but got {out.dtype}"
  )
  assert not torch.isnan(out).any(), "Output contains NaNs!"
  assert not torch.isinf(out).any(), "Output contains Infs!"
