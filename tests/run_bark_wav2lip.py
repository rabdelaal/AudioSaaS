import requests
import time

VIDEO = 'd:/AudioSaaS/uploads/84de91f22ed4406a814cf91547660382_17aa5283-d71a-4c3b-a7c4-a0de8d009a04.mp4'
URL = 'http://127.0.0.1:5000/start_pipeline'
with open(VIDEO,'rb') as f:
    files = {'video':('video.mp4', f, 'video/mp4')}
    data = {'prompt':'', 'transcript':'Hello, this is a bark TTS test.','foley':'foleycrafter','tts':'bark','sync':'timed','duration':'5'}
    r = requests.post(URL, files=files, data=data)
    print('start:', r.status_code, r.text)
    job_id = r.json()['job_id']

for i in range(60):
    r = requests.get(f'http://127.0.0.1:5000/job_status/{job_id}')
    print('status', r.status_code, r.text)
    j = r.json()
    if j.get('status') in ('done','error'):
        break
    time.sleep(2)
print('final', r.text)
