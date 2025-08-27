# MMAudio -> Flask wrapper

This small Flask app lets you upload a video, optionally upload an audio file (or auto-generate one), and produces a downloadable video with the audio merged.

Prerequisites
- Python 3.8+
- ffmpeg installed and available on PATH. On Windows, download from https://ffmpeg.org/ and add to PATH.

Install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run

```powershell
python app.py
```

Then open http://localhost:5000 in your browser. Upload a video and optionally an audio file or choose to generate audio.

Notes
- The app uses ffmpeg subprocess calls to extract/generate/merge audio. Ensure ffmpeg is installed.
- Uploaded files are stored in `uploads/` inside the project folder.
- This is a minimal example intended for local testing. For production, secure uploads, sanitize filenames more strictly, and run behind a proper web server.

Submodules and CI
-----------------

This repository uses git submodules to include upstream projects (MMAudio, bark, Wav2Lip). After cloning, initialise submodules with:

```bash
git submodule update --init --recursive
```

A GitHub Actions workflow is included at `.github/workflows/ci.yml` which initializes submodules and runs the lightweight reference tests on push.

If you want me to add more tests to CI (end-to-end pipeline smoke test or linting), tell me which tests to include and I will wire them into the workflow.
