import os
import uuid
import subprocess
import logging
import threading
import sys
import time
import importlib
from typing import List
from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

# Optional reference implementation backend (minimal PyTorch scaffold)
try:
    from ref_impl import core as ref_core

    REF_IMPL_OK = True
except Exception:
    ref_core = None
    REF_IMPL_OK = False

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB max upload
app.static_folder = os.path.join(BASE_DIR, "static")

# Basic logging for production visibility
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


@app.route("/health")
def health():
    return {"status": "ok"}, 200


def allowed_file(filename, allowed_set):
    _, ext = os.path.splitext(filename.lower())
    return ext in allowed_set


def run_ffmpeg(args):
    # args is a list for subprocess
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {proc.stderr.decode('utf8', errors='ignore')}"
        )
    return proc


def has_audio_stream(path: str) -> bool:
    """Return True if the media file has at least one audio stream (uses ffprobe)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out = proc.stdout.decode("utf8", errors="ignore").strip()
        return len(out) > 0
    except Exception:
        # If ffprobe is not available or fails, be conservative and assume no audio
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=True)


# Simple in-memory job store. For production, use a persistent queue/db.
JOBS: dict = {}


def run_mmaudio_job(
    job_id: str, video_path: str, prompt: str, duration: float, variant: str
):
    """Run the MMAudio demo.py as a subprocess and update JOBS with progress.

    Invokes the demo script in the cloned MMAudio repo. Assumes dependencies
    (PyTorch, torchaudio, etc.) are installed and models may be downloaded.
    """
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = time.time()
    output_dir = os.path.join(BASE_DIR, "MMAudio", "output")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "MMAudio", "demo.py"),
        "--video",
        video_path,
        "--prompt",
        prompt,
        "--duration",
        str(duration),
        "--output",
        output_dir,
    ]
    if variant:
        cmd += ["--variant", variant]

    # Launch subprocess and stream stderr/stdout for basic progress capture
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        return

    # Read lines and store the latest message
    latest = ""
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            latest = line.strip()
            JOBS[job_id]["progress_msg"] = latest

    ret = proc.wait()
    JOBS[job_id]["finished_at"] = time.time()
    if ret != 0:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"demo.py exited with code {ret}"
        return

    # find generated outputs in output_dir (matching video stem)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    generated = []
    for fn in os.listdir(output_dir):
        if fn.startswith(stem):
            generated.append(fn)

    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["outputs"] = generated


def _ffprobe_duration(path: str) -> float:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out = proc.stdout.decode("utf8", errors="ignore").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def check_requirements(tts_engine: str, sync_method: str):
    """Check optional dependencies and return a tuple:
    (missing: List[str], bark_ok: bool, parler_ok: bool, pyttsx3_ok: bool)
    """
    missing: List[str] = []
    try:
        importlib.import_module("whisperx")
        whisperx_ok = True
    except Exception:
        whisperx_ok = False
    if sync_method == "whisperx" and not whisperx_ok:
        missing.append("whisperx")

    bark_ok = True
    parler_ok = True
    pyttsx3_ok = True
    try:
        importlib.import_module("bark")
    except Exception:
        bark_ok = False
    try:
        importlib.import_module("parler")
    except Exception:
        parler_ok = False
    try:
        importlib.import_module("pyttsx3")
    except Exception:
        pyttsx3_ok = False

    if tts_engine == "bark" and not bark_ok:
        missing.append("bark")
    if tts_engine == "parler" and not parler_ok:
        missing.append("parler-tts")
    if tts_engine in ("bark", "parler") and not (bark_ok or parler_ok or pyttsx3_ok):
        missing.append("pyttsx3 (fallback)")

    return missing, bark_ok, parler_ok, pyttsx3_ok


def determine_duration(video_path: str, options: dict) -> float:
    """Return duration in seconds, using ffprobe fallback to options."""
    duration = _ffprobe_duration(video_path)
    if duration <= 0:
        return float(options.get("duration") or 8.0)
    return duration


def build_amix_cmd(mix_inputs: list, final_audio: str) -> list:
    """Construct an ffmpeg amix command list for the given inputs."""
    inputs = []
    for _, path in mix_inputs:
        inputs += ["-i", path]
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            f"amix=inputs={len(mix_inputs)}:normalize=1",
            "-ar",
            "44100",
            "-ac",
            "2",
            final_audio,
        ]
    )
    return cmd


def find_wav2lip_checkpoint(wav2lip_dir: str):
    """Return first existing wav2lip checkpoint path or None."""
    checkpoint_paths = [
        os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth"),
        os.path.join(wav2lip_dir, "checkpoints", "wav2lip.pth"),
    ]
    for p in checkpoint_paths:
        if os.path.isfile(p):
            return p
    return None


def build_extract_copy_cmd(video_path: str, extracted_audio: str) -> list:
    """Return ffmpeg command list to copy audio stream from video."""
    return ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", extracted_audio]


def build_extract_reencode_cmd(video_path: str, extracted_audio: str) -> list:
    """Return ffmpeg command list to re-encode audio if copy fails."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        extracted_audio,
    ]


def build_foley_cmd(duration: float, foley_path: str) -> list:
    """Return ffmpeg command list to generate ambience using anoisesrc."""
    return [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=color=pink:duration={duration}",
        "-ar",
        "44100",
        "-ac",
        "2",
        foley_path,
    ]


def build_merge_cmd(video_path: str, audio_path: str, out_video: str) -> list:
    """Return ffmpeg command list to merge audio into video with aac encoding."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_video,
    ]


def build_wav2lip_cmd(
    wav2lip_dir: str,
    ckpt: str,
    video_path: str,
    final_audio: str,
    lipsync_out: str,
) -> list:
    """Return the command list to run Wav2Lip inference script."""
    return [
        sys.executable,
        os.path.join(wav2lip_dir, "inference.py"),
        "--checkpoint_path",
        ckpt,
        "--face",
        video_path,
        "--audio",
        final_audio,
        "--outfile",
        lipsync_out,
    ]


def run_pipeline_job(job_id: str, video_path: str, options: dict):  # noqa: C901
    """Run a practical pipeline: Foley/Ambience + TTS + Sync.

    Refactored to call helper functions for smaller testable units.
    """
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = time.time()

    tts_engine = options.get("tts", "parler")
    sync_method = options.get("sync", "whisperx")
    transcript = options.get("transcript") or ""
    backend = options.get("backend") or ""

    missing, bark_ok, parler_ok, pyttsx3_ok = (
        check_requirements(tts_engine, sync_method)
    )
    if missing:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = (
            "Missing required packages: " + ", ".join(missing) + "."
            + " Install them and retry. Example: pip install whisperx pyttsx3."
        )
        return

    if backend == "ref" and not REF_IMPL_OK:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = (
            "Reference backend (ref_impl) not available. Ensure ref_impl is included."
        )
        return

    duration = determine_duration(video_path, options)

    # Step 1: extract existing audio if present
    extracted_audio = None
    if has_audio_stream(video_path):
        extracted_audio = os.path.join(UPLOAD_DIR, f"{job_id}_extracted.m4a")
        try:
            run_ffmpeg(build_extract_copy_cmd(video_path, extracted_audio))
        except RuntimeError:
            run_ffmpeg(build_extract_reencode_cmd(video_path, extracted_audio))

    # Step 2: Foley/Ambience generation (fallback: ffmpeg noise)
    foley_path = os.path.join(UPLOAD_DIR, f"{job_id}_foley.wav")
    try:
        run_ffmpeg(build_foley_cmd(duration, foley_path))
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"Failed to generate ambience: {e}"
        return

    # Step 3: TTS generation
    tts_path = None
    if backend == "ref":
        JOBS[job_id]["progress_msg"] = "Generating audio with reference backend..."
        try:
            seed = int(job_id[:8], 16)
            h = ref_core.initialize_h(seed)
            m = ref_core.mobius_map(0.5)
            max_it = 64
            prev = h
            cur = None
            for _ in range(max_it):
                cur = ref_core.mobius_diffuse(prev, m, 0.5)
                if ref_core.stable_check(prev, cur, tol=1e-3):
                    break
                prev = cur

            wav = ref_core.decode_to_modality(cur, "audio")
            import wave
            import numpy as _np

            tts_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.wav")
            arr = wav.detach().cpu().numpy().astype("float32")
            arr = arr / max(1e-6, abs(arr).max())
            data = (_np.clip(arr, -1, 1) * 32767).astype(_np.int16)
            with wave.open(tts_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(data.tobytes())
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"Reference backend generation failed: {e}"
            return
    else:
        if transcript:
            # prefer Bark if available
            if bark_ok:
                try:
                    bark_api = importlib.import_module("bark.api")
                    bark_gen = importlib.import_module("bark.generation")
                    audio_arr = bark_api.generate_audio(transcript)
                    try:
                        from scipy.io.wavfile import write as write_wav

                        write_wav(tts_path, bark_gen.SAMPLE_RATE, audio_arr)
                    except Exception:
                        import wave
                        import numpy as _np

                        data = (_np.clip(audio_arr, -1, 1) * 32767).astype(_np.int16)
                        with wave.open(tts_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(bark_gen.SAMPLE_RATE)
                            wf.writeframes(data.tobytes())
                except Exception as e:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = f"Bark TTS failed: {e}"
                    return
            elif pyttsx3_ok and not (bark_ok or parler_ok):
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.save_to_file(transcript, tts_path)
                engine.runAndWait()
            else:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = "No available TTS engine (bark/parler/pyttsx3)."
                return

    if transcript:
        tts_path = os.path.join(UPLOAD_DIR, f"{job_id}_tts.wav")
        if bark_ok:
            try:
                bark_api = importlib.import_module("bark.api")
                bark_gen = importlib.import_module("bark.generation")
                audio_arr = bark_api.generate_audio(transcript)
                try:
                    from scipy.io.wavfile import write as write_wav

                    write_wav(tts_path, bark_gen.SAMPLE_RATE, audio_arr)
                except Exception:
                    import wave
                    import numpy as _np

                    data = (_np.clip(audio_arr, -1, 1) * 32767).astype(_np.int16)
                    with wave.open(tts_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(bark_gen.SAMPLE_RATE)
                        wf.writeframes(data.tobytes())
            except Exception as e:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = f"Bark TTS failed: {e}"
                return
        elif pyttsx3_ok and not (bark_ok or parler_ok):
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.save_to_file(transcript, tts_path)
            engine.runAndWait()
        else:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = "No available TTS engine (bark/parler/pyttsx3)."
            return

    # Step 4: Mix tracks: prioritize tts > foley > extracted
    mix_inputs = []
    if tts_path:
        mix_inputs.append(("a", tts_path))
    if foley_path:
        mix_inputs.append(("b", foley_path))
    if extracted_audio:
        mix_inputs.append(("c", extracted_audio))

    final_audio = os.path.join(UPLOAD_DIR, f"{job_id}_final.wav")
    try:
        amix_cmd = build_amix_cmd(mix_inputs, final_audio)
        run_ffmpeg(amix_cmd)
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"Failed to mix audio: {e}"
        return

    # Step 5: Merge into video
    out_video = os.path.join(UPLOAD_DIR, f"{job_id}_with_pipeline.mp4")
    try:
        run_ffmpeg(build_merge_cmd(video_path, final_audio, out_video))
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"Failed to merge final audio: {e}"
        return

    # Optional: attempt Wav2Lip lip-sync if repo and checkpoint are available
    wav2lip_dir = os.path.join(BASE_DIR, "Wav2Lip")
    available_ckpt = find_wav2lip_checkpoint(wav2lip_dir)

    if available_ckpt and os.path.isdir(wav2lip_dir):
        JOBS[job_id]["progress_msg"] = "Running Wav2Lip lip-sync..."
        lipsync_out = os.path.join(UPLOAD_DIR, f"{job_id}_wav2lip.mp4")
        cmd = build_wav2lip_cmd(
            wav2lip_dir, available_ckpt, video_path, final_audio, lipsync_out
        )
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                JOBS[job_id]["progress_msg"] = "Wav2Lip failed"
                JOBS[job_id]["error"] = proc.stderr.decode("utf8", errors="ignore")
            else:
                JOBS[job_id]["outputs"] = [
                    os.path.basename(final_audio),
                    os.path.basename(lipsync_out),
                ]
                JOBS[job_id]["status"] = "done"
                return
        except Exception as e:
            JOBS[job_id]["progress_msg"] = "Wav2Lip invocation error"
            JOBS[job_id]["error"] = str(e)
            return
    else:
        if os.path.isdir(wav2lip_dir):
            JOBS[job_id]["progress_msg"] = (
                "Wav2Lip checkpoint missing. Download the checkpoint "
                "(wav2lip_gan.pth or wav2lip.pth) and place it in Wav2Lip/checkpoints/"
            )

    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["outputs"] = [
        os.path.basename(final_audio),
        os.path.basename(out_video),
    ]


@app.route("/start_pipeline", methods=["POST"])
def start_pipeline():
    if "video" not in request.files:
        return {"error": "no video uploaded"}, 400
    video = request.files["video"]
    _prompt = request.form.get("prompt", "")
    transcript = request.form.get("transcript", "")
    foley = request.form.get("foley") or "foleycrafter"
    tts = request.form.get("tts") or "parler"
    sync = request.form.get("sync") or "whisperx"

    vid_name = secure_filename(video.filename)
    uid = uuid.uuid4().hex
    saved_video = f"{uid}_{vid_name}"
    video_path = os.path.join(UPLOAD_DIR, saved_video)
    video.save(video_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "video": saved_video, "created_at": time.time()}

    opts = {
        "foley": foley,
        "tts": tts,
        "sync": sync,
        "transcript": transcript,
        "duration": request.form.get("duration"),
    }
    t = threading.Thread(
        target=run_pipeline_job, args=(job_id, video_path, opts), daemon=True
    )
    t.start()

    return {"job_id": job_id}


@app.route("/MMAudio/output/<path:filename>")
def mmaudio_output(filename):
    out_dir = os.path.join(BASE_DIR, "MMAudio", "output")
    return send_from_directory(out_dir, filename, as_attachment=True)


@app.route("/process", methods=["POST"])
def process():
    # Checklist: receive files, save them, extract or generate audio,
    # merge, and provide a download link
    if "video" not in request.files:
        flash("No video uploaded")
        return redirect(url_for("index"))

    video = request.files["video"]
    audio = request.files.get("audio")
    generate_flag = request.form.get("generate_audio") == "on"
    gen_duration = float(request.form.get("generate_duration") or 10)

    if video.filename == "":
        flash("No selected video")
        return redirect(url_for("index"))

    if not allowed_file(video.filename, ALLOWED_VIDEO):
        flash("Video file type not allowed")
        return redirect(url_for("index"))

    vid_name = secure_filename(video.filename)
    uid = uuid.uuid4().hex
    saved_video = f"{uid}_{vid_name}"
    video_path = os.path.join(UPLOAD_DIR, saved_video)
    video.save(video_path)

    audio_path = None
    saved_audio = None

    try:
        if audio and audio.filename != "":
            if not allowed_file(audio.filename, ALLOWED_AUDIO):
                flash("Audio file type not allowed")
                return redirect(url_for("index"))
            aud_name = secure_filename(audio.filename)
            saved_audio = f"{uid}_{aud_name}"
            audio_path = os.path.join(UPLOAD_DIR, saved_audio)
            audio.save(audio_path)
        elif generate_flag:
            # generate a sine tone using ffmpeg lavfi
            saved_audio = f"{uid}_generated.wav"
            audio_path = os.path.join(UPLOAD_DIR, saved_audio)
            args = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={gen_duration}",
                "-ar",
                "44100",
                "-ac",
                "2",
                audio_path,
            ]
            run_ffmpeg(args)
        else:
            # try to extract audio from video if it has an audio stream
            if has_audio_stream(video_path):
                saved_audio = f"{uid}_extracted.m4a"
                audio_path = os.path.join(UPLOAD_DIR, saved_audio)
                # prefer copying stream (fast) but fall back to re-encode if copy fails
                try:
                    args = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-vn",
                        "-acodec",
                        "copy",
                        audio_path,
                    ]
                    run_ffmpeg(args)
                except RuntimeError:
                    args = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-vn",
                        "-ac",
                        "2",
                        "-ar",
                        "44100",
                        audio_path,
                    ]
                    run_ffmpeg(args)
            else:
                # No audio stream present. Fall back to generating a short
                # sine tone so merging succeeds.
                saved_audio = f"{uid}_generated_from_noaudio.wav"
                audio_path = os.path.join(UPLOAD_DIR, saved_audio)
                gen_dur = max(1.0, float(request.form.get("generate_duration") or 10))
                args = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency=440:duration={gen_dur}",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    audio_path,
                ]
                run_ffmpeg(args)
                flash("Input video had no audio stream — a generated tone was used.")

        # Merge audio into video (re-mux video stream and encode audio to aac if needed)
        output_video_name = f"{uid}_with_audio.mp4"
        output_video_path = os.path.join(UPLOAD_DIR, output_video_name)
        merge_args = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_video_path,
        ]
        run_ffmpeg(merge_args)

    except Exception as e:
        flash(str(e))
        return redirect(url_for("index"))

    # Provide links to download
    return render_template(
        "result.html", audio_file=saved_audio, video_file=output_video_name
    )


@app.route("/start_job", methods=["POST"])
def start_job():
    """Start a MMAudio generation job using the cloned repo's demo.py.

    Expects multipart form with `video` and `prompt`. Optional `duration` and
    `variant`. Returns job id as JSON.
    """
    if "video" not in request.files:
        return {"error": "no video uploaded"}, 400
    video = request.files["video"]
    prompt = request.form.get("prompt", "")
    duration = float(request.form.get("duration") or 8.0)
    variant = request.form.get("variant") or ""

    vid_name = secure_filename(video.filename)
    uid = uuid.uuid4().hex
    saved_video = f"{uid}_{vid_name}"
    video_path = os.path.join(UPLOAD_DIR, saved_video)
    video.save(video_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "video": saved_video,
        "prompt": prompt,
        "created_at": time.time(),
    }

    t = threading.Thread(
        target=run_mmaudio_job,
        args=(job_id, video_path, prompt, duration, variant),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job not found"}, 404
    return job


if __name__ == "__main__":
    # Disable the auto-reloader on Windows to avoid socket errors when running
    # the development server inside some environments. For development you can
    # set debug=True but keep use_reloader=False to avoid the reloader thread.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
