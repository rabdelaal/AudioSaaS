import torch
from ref_impl.core import initialize_H, mobius_map, mobius_diffuse, decode_to_modality


def test_initialize_shape():
    H = initialize_H(42)
    assert isinstance(H, torch.Tensor)
    assert H.shape == (2, 2, 2, 2, 4, 8, 2)


def test_mobius_diffuse_contracts():
    H = initialize_H(0)
    M = mobius_map(0.5)
    next_H = mobius_diffuse(H, M, 0.5)
    assert torch.isfinite(next_H).all()
    # In this toy kernel the tanh nonlinearity tends to contract values
    assert next_H.abs().mean() <= H.abs().mean() + 1e-3


def test_decode_audio_length():
    H = initialize_H(123)
    wav = decode_to_modality(H, 'audio')
    assert wav.ndim == 1
    assert wav.abs().max() <= 1.0 + 1e-6
