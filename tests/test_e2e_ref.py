import os
import subprocess
import tempfile
import time
from ref_impl import core

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def has_audio(path):
    try:
        out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path])
        return b"audio" in out
    except Exception:
        return False


def test_e2e_ref_pipeline():
    # create tiny silent video using ffmpeg
    video = os.path.join(UPLOAD_DIR, 'test_silent.mp4')
    wav = os.path.join(UPLOAD_DIR, 'test_ref.wav')
    out = os.path.join(UPLOAD_DIR, 'test_with_audio.mp4')

    # make video
    subprocess.check_call(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=1', '-f', 'lavfi', '-i', 'anullsrc', '-c:v', 'libx264', '-t', '1', '-pix_fmt', 'yuv420p', video])

    # generate ref_impl wav
    H = core.initialize_H(42)
    M = core.mobius_map(0.5)
    cur = core.mobius_diffuse(H, M, 0.5)
    wav_tensor = core.decode_to_modality(cur, 'audio')

    import wave, numpy as _np
    arr = wav_tensor.detach().cpu().numpy().astype('float32')
    arr = arr / max(1e-6, abs(arr).max())
    data = (_np.clip(arr, -1, 1) * 32767).astype(_np.int16)
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(data.tobytes())

    # merge
    subprocess.check_call(['ffmpeg', '-y', '-i', video, '-i', wav, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', out])

    assert os.path.exists(out)
    assert has_audio(out)
