import os
import pytest

# Integration test: requires the Flask server to be running or
# set RUN_INTEGRATION=1 to enable
if not os.environ.get("RUN_INTEGRATION"):
    pytest.skip(
        "Integration tests disabled. Set RUN_INTEGRATION=1 to enable.",
        allow_module_level=True,
    )

import time
import requests

VIDEO = "d:/AudioSaaS/MMAudio/training/example_videos/example.mp4"
URL = "http://127.0.0.1:5000/start_pipeline"

with open(VIDEO, "rb") as f:
    files = {"video": ("test.mp4", f, "video/mp4")}
    data = {
        "prompt": "gentle ambient ocean",
        "transcript": "Hello world",
        "foley": "foleycrafter",
        "tts": "pyttsx3",
        "sync": "timed",
        "duration": "4",
        "backend": "ref",
    }
    r = requests.post(URL, files=files, data=data)
    print("start response:", r.status_code, r.text)
    if r.status_code != 200:
        raise SystemExit("start failed")
    job_id = r.json()["job_id"]

STATUS_URL = f"http://127.0.0.1:5000/job_status/{job_id}"
for _i in range(60):
    r = requests.get(STATUS_URL)
    print("status", r.status_code, r.text)
    if r.status_code != 200:
        break
    j = r.json()
    if j.get("status") in ("done", "error"):
        break
    time.sleep(2)
print("final:", r.text)
