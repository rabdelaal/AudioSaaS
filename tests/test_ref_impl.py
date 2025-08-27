import torch
from ref_impl.core import initialize_h, mobius_map, mobius_diffuse, decode_to_modality


def test_initialize_shape():
    h = initialize_h(42)
    assert isinstance(h, torch.Tensor)
    assert h.shape == (2, 2, 2, 2, 4, 8, 2)


def test_mobius_diffuse_contracts():
    h = initialize_h(0)
    m = mobius_map(0.5)
    next_h = mobius_diffuse(h, m, 0.5)
    assert torch.isfinite(next_h).all()
    # In this toy kernel the tanh nonlinearity tends to contract values
    assert next_h.abs().mean() <= h.abs().mean() + 1e-3


def test_decode_audio_length():
    h = initialize_h(123)
    wav = decode_to_modality(h, "audio")
    assert wav.ndim == 1
    assert wav.abs().max() <= 1.0 + 1e-6
