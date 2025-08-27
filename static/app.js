const videoInput = document.getElementById('video-input');
const videoPreview = document.getElementById('video-preview');
const startBtn = document.getElementById('start-btn');
const mergeBtn = document.getElementById('merge-btn');
const statusEl = document.getElementById('status');
const outputsEl = document.getElementById('outputs');
const startPipelineBtn = document.getElementById('start-pipeline');
const transcriptEl = document.getElementById('transcript');
const pipelineFoley = document.getElementById('pipeline-foley');
const pipelineTts = document.getElementById('pipeline-tts');
const pipelineSync = document.getElementById('pipeline-sync');

let currentJob = null;

videoInput.addEventListener('change', ()=>{
  const file = videoInput.files[0];
  if(!file) return;
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
});

startPipelineBtn.addEventListener('click', async ()=>{
  const file = videoInput.files[0];
  if(!file){ alert('Select a video first'); return; }
  const prompt = document.getElementById('prompt').value;
  const transcript = transcriptEl.value;
  const foley = pipelineFoley.value;
  const tts = pipelineTts.value;
  const sync = pipelineSync.value;
    const backend = document.getElementById('pipeline-backend').value;

  statusEl.textContent = 'Uploading and starting pipeline...';
  outputsEl.innerHTML = '';

  const fd = new FormData();
  fd.append('video', file);
  fd.append('prompt', prompt);
  fd.append('transcript', transcript);
  fd.append('foley', foley);
  fd.append('tts', tts);
  fd.append('sync', sync);
  fd.append('backend', backend);

  const res = await fetch('/start_pipeline', { method: 'POST', body: fd });
  if(!res.ok){ statusEl.textContent = 'Failed to start pipeline'; return; }
  const data = await res.json();
  currentJob = data.job_id;
  pollJob(currentJob);
});

startBtn.addEventListener('click', async ()=>{
  const file = videoInput.files[0];
  if(!file){ alert('Select a video first'); return; }
  const prompt = document.getElementById('prompt').value;
  const duration = document.getElementById('duration').value || 8;
  const variant = document.getElementById('variant').value;

  statusEl.textContent = 'Uploading and queuing job...';
  outputsEl.innerHTML = '';

  const fd = new FormData();
  fd.append('video', file);
  fd.append('prompt', prompt);
  fd.append('duration', duration);
  fd.append('variant', variant);

  const res = await fetch('/start_job', { method: 'POST', body: fd });
  if(!res.ok){ statusEl.textContent = 'Failed to start job'; return; }
  const data = await res.json();
  currentJob = data.job_id;
  pollJob(currentJob);
});

mergeBtn.addEventListener('click', async ()=>{
  // quick merge: use existing /process form endpoint by creating a form submit
  const file = videoInput.files[0];
  if(!file){ alert('Select a video first'); return; }
  const form = new FormData();
  form.append('video', file);
  // use fetch to /process
  statusEl.textContent = 'Uploading for quick merge...';
  const res = await fetch('/process', { method: 'POST', body: form });
  if(res.redirected){ window.location = res.url; return; }
  const text = await res.text();
  statusEl.textContent = 'Response received';
  // show raw response in new tab
  const w = window.open();
  w.document.write(text);
});

async function pollJob(jobId){
  statusEl.textContent = 'Job queued...';
  while(true){
    const res = await fetch('/job_status/' + jobId);
    if(!res.ok){ statusEl.textContent = 'Job not found'; return; }
    const json = await res.json();
    statusEl.textContent = `${json.status}` + (json.progress_msg ? ` — ${json.progress_msg}` : '');
    if(json.status === 'done'){
      renderOutputs(json.outputs || []);
      return;
    }
    if(json.status === 'error'){
      statusEl.textContent = 'Error: ' + (json.error || 'unknown');
      return;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
}

function renderOutputs(outputs){
  outputsEl.innerHTML = '';
  if(outputs.length === 0) outputsEl.textContent = 'No outputs found';
  outputs.forEach(fn => {
    const a = document.createElement('a');
    a.href = '/MMAudio/output/' + fn; // we will add a route later if needed
    a.textContent = fn;
    outputsEl.appendChild(a);
  });
  statusEl.textContent = 'Done';
}
