"""Minimal reference implementation scaffolding for a tiny deterministic toy.

This module is intentionally small and dependency-free so it can be used in
unit tests and CI without heavy ML dependencies.
"""
from typing import Tuple

import torch


def initialize_h(
    seed: int, shape: Tuple[int, ...] = (2, 2, 2, 2, 4, 8, 2)
) -> torch.Tensor:
    """Create a deterministic initial tensor for testing.

    The returned tensor values live roughly in [-1, 1].
    """
    torch.manual_seed(seed)
    return (torch.rand(*shape) * 2.0 - 1.0).to(torch.float32)


def mobius_map(tau: float) -> torch.Tensor:
    """Return a small 2x2 contractive matrix parametrized by `tau`."""
    c = torch.cos(torch.tensor(tau, dtype=torch.float32))
    s = torch.sin(torch.tensor(tau, dtype=torch.float32))
    return torch.tensor([[0.8 * c, -0.2 * s], [0.2 * s, 0.8 * c]], dtype=torch.float32)


def mobius_transform(q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Apply a 2x2 linear transform to vectors stored in the last dimension."""
    return torch.einsum("...k,kj->...j", q, m)


def mobius_diffuse(h: torch.Tensor, m: torch.Tensor, tau: float) -> torch.Tensor:
    """Apply a tiny diffusion kernel across the penultimate dimension.

    This toy kernel mixes neighboring slots and applies a tanh nonlinearity.
    """
    out = h.clone()
    alpha = torch.tanh(torch.tensor(tau, dtype=torch.float32))
    # iterate over interior positions and mix neighbors
    for idx in range(1, h.shape[-2] - 1):
        left = mobius_transform(h[..., idx - 1, :], m)
        center = mobius_transform(h[..., idx, :], m)
        right = mobius_transform(h[..., idx + 1, :], m)
        out[..., idx, :] = torch.tanh((left + center + right) * (0.33 + 0.1 * alpha))
    return out


def decode_to_modality(h: torch.Tensor, modality: str = "audio") -> torch.Tensor:
    """Decode the tensor to a toy modality output.

    For audio we collapse spatial dims into a 1-D waveform and normalize to [-1,1].
    """
    if modality != "audio":
        raise NotImplementedError(
            "Only 'audio' modality is implemented in the reference scaffold."
        )

    # mean over all axes except the final small vector, then flatten
    reduced = h.mean(dim=list(range(h.ndim - 1))).squeeze()
    wav = reduced.flatten()
    denom = wav.abs().max().clamp(min=1e-6)
    return (wav / denom).to(torch.float32)


def stable_check(h_prev: torch.Tensor, h_next: torch.Tensor, tol: float = 1e-4) -> bool:
    """Simple stability check: small change between iterations."""
    return torch.norm(h_next - h_prev) < tol
