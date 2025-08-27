Minimal Σ‑Möbius reference implementation

This folder contains a tiny, deterministic PyTorch scaffold used for unit tests and early prototyping.

Quick start (PowerShell):

```powershell
# Create a short-path virtualenv (recommended)
python -m venv C:\audiovenv
C:\audiovenv\Scripts\python.exe -m pip install -U pip setuptools wheel
C:\audiovenv\Scripts\python.exe -m pip install -r ref_impl/requirements.txt

# Run tests
C:\audiovenv\Scripts\python.exe -m pytest -q
```

Files:
- `ref_impl/core.py` — toy implementations of `initialize_h`, `mobius_map`, `mobius_diffuse`, `decode_to_modality`.
- `tests/test_ref_impl.py` — pytest unit tests (happy path + simple edge checks).

Notes:
- This is a minimal scaffold for development. Replace kernels with production implementations later.
