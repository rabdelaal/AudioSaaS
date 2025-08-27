import time
import requests

VIDEO = 'd:/AudioSaaS/MMAudio/training/example_videos/0B4dYTMsgHA_000130.mp4'
URL = 'http://127.0.0.1:5000/start_pipeline'

with open(VIDEO, 'rb') as f:
    files = {'video': ('test.mp4', f, 'video/mp4')}
    data = {'prompt': 'ocean waves and seagulls', 'transcript': '', 'foley': 'foleycrafter', 'tts': 'pyttsx3', 'sync': 'timed', 'duration': '8'}
    r = requests.post(URL, files=files, data=data)
    print('start response:', r.status_code, r.text)
    if r.status_code != 200:
        raise SystemExit('start failed')
    job_id = r.json()['job_id']

STATUS_URL = f'http://127.0.0.1:5000/job_status/{job_id}'
for i in range(60):
    r = requests.get(STATUS_URL)
    print('status', r.status_code, r.text)
    if r.status_code != 200:
        break
    j = r.json()
    if j.get('status') in ('done','error'):
        break
    time.sleep(2)
print('final:', r.text)
