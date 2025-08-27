"""Minimal reference implementation scaffolding for Σ-Möbius (toy prototype).
This is intentionally tiny and deterministic so it can be used for unit tests and prototypes.
"""
from typing import Tuple

import torch


def initialize_H(seed: int, shape: Tuple[int, ...] = (2, 2, 2, 2, 4, 8, 2)) -> torch.Tensor:
    """Create a deterministic initial Hyper-Möbius tensor for testing.

    The last dimension encodes a small complex-like vector (real, imag) in this toy impl.
    """
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32)


def mobius_map(tau: float) -> torch.Tensor:
    """Return a small 2x2 contractive matrix (spectral radius < 1).

    This toy map uses simple trigonometric components to vary with `tau`.
    """
    c = torch.cos(torch.tensor(tau, dtype=torch.float32))
    s = torch.sin(torch.tensor(tau, dtype=torch.float32))
    # Small rotation + contraction
    M = torch.tensor([[0.8 * c, -0.2 * s], [0.2 * s, 0.8 * c]], dtype=torch.float32)
    return M


def mobius_transform(q: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply a 2x2 linear transform to last dimension vectors.

    q: (..., 2)
    M: (2,2)
    Returns: (..., 2)
    """
    return torch.matmul(q, M.t())


def mobius_diffuse(H: torch.Tensor, M: torch.Tensor, tau: float) -> torch.Tensor:
    """Apply the toy Möbius diffusion kernel to the tensor H.

    This implementation reshapes the tensor to apply the 2x2 transform to the
    last dimension and a pointwise nonlinearity.
    """
    orig_shape = H.shape
    flat = H.reshape(-1, orig_shape[-1])  # (N, 2)
    transformed = mobius_transform(flat, M)
    # simple nonlinearity to emulate diffusion/renormalization
    sigma = torch.tanh(transformed)
    out = sigma.reshape(orig_shape)
    return out


def decode_to_modality(H: torch.Tensor, modality: str = "audio") -> torch.Tensor:
    """Decode the tensor to a toy modality output.

    For 'audio' we return a 1-D waveform by mean-reduction and normalization.
    """
    if modality != "audio":
        raise NotImplementedError("Only 'audio' modality is implemented in the reference scaffold.")
    # mean over all axes except last (the small complex vector), then flatten
    reduced = H.mean(dim=list(range(H.dim() - 1)))
    wav = reduced.flatten()
    # normalize to [-1, 1]
    denom = wav.abs().max().clamp(min=1e-6)
    wav = (wav / denom).to(torch.float32)
    return wav


def stable_check(H_prev: torch.Tensor, H_next: torch.Tensor, tol: float = 1e-4) -> bool:
    """Simple stability check: small change between iterations."""
    return torch.norm(H_next - H_prev) < tol
