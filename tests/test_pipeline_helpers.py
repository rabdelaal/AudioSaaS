import os

from app import (
    build_amix_cmd,
    check_requirements,
    determine_duration,
    find_wav2lip_checkpoint,
)


def test_build_amix_cmd_single():
    final = "out.wav"
    cmd = build_amix_cmd([("a", "one.wav")], final)
    assert isinstance(cmd, list)
    assert "-i" in cmd
    assert final in cmd


def test_build_amix_cmd_multiple():
    final = "out.wav"
    cmd = build_amix_cmd([("a", "one.wav"), ("b", "two.wav")], final)
    assert "amix=inputs=2" in " ".join(cmd)


def test_find_wav2lip_checkpoint(tmp_path):
    wav2 = tmp_path / "Wav2Lip"
    ck = wav2 / "checkpoints"
    ck.mkdir(parents=True)
    p = ck / "wav2lip.pth"
    p.write_bytes(b"x")
    found = find_wav2lip_checkpoint(str(wav2))
    assert found is not None
    assert os.path.basename(found) in ("wav2lip.pth", "wav2lip_gan.pth")


def test_check_requirements_type():
    missing, bark_ok, parler_ok, pyttsx3_ok = check_requirements("parler", "timed")
    assert isinstance(missing, list)
    assert isinstance(bark_ok, bool)
    assert isinstance(parler_ok, bool)
    assert isinstance(pyttsx3_ok, bool)


def test_determine_duration_tmp_file(tmp_path):
    # create a tiny file and ensure function returns fallback when ffprobe
    # is not present/doesn't return a duration
    f = tmp_path / "video.mp4"
    f.write_bytes(b"\0")
    dur = determine_duration(str(f), {"duration": "3"})
    assert isinstance(dur, float)
