/* ==========================================================================
   Theme Analytics — frontend
   Vanilla ES2020, no build step. Everything goes through the same REST API
   documented at /docs, so the UI is never a privileged client.

   Live progress uses the SSE endpoint and falls back to polling if the
   EventSource fails (some corporate proxies buffer text/event-stream).
   ========================================================================== */
'use strict';

const API = '/api/v1';

const state = {
  files: [],          // File[] staged for upload
  jobs: [],           // JobSummary[]
  jobId: null,        // selected job
  job: null,          // JobDetail of the selected job
  rows: [],           // ResultRow[]
  stream: null,       // EventSource
  pollTimer: null,
  maxFiles: 200,
  maxMb: 200,
  allowed: [],
};

/* ---------------------------------------------------------------- helpers */
const $ = (id) => document.getElementById(id);

/** Build an element. Text is always set via textContent — never innerHTML,
 *  because filenames and transcripts are untrusted input. */
function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  if (opts.class) node.className = opts.class;
  if (opts.text !== undefined) node.textContent = opts.text;
  if (opts.title) node.title = opts.title;
  if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, v);
  if (opts.on) for (const [k, v] of Object.entries(opts.on)) node.addEventListener(k, v);
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(API + path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.error?.message) message = body.error.message;
    } catch { /* non-JSON error body */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function toast(message, kind = '') {
  const node = el('div', { class: `toast ${kind}`, text: message });
  $('toasts').append(node);
  setTimeout(() => {
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 250);
  }, kind === 'err' ? 7000 : 3800);
}

const STATUS_PILL = {
  queued: 'pill-muted', running: 'pill-run', completed: 'pill-ok',
  completed_with_errors: 'pill-warn', failed: 'pill-err', cancelled: 'pill-muted',
};
const STATUS_LABEL = {
  queued: 'queued', running: 'running', completed: 'completed',
  completed_with_errors: 'completed with errors', failed: 'failed', cancelled: 'cancelled',
};

function pill(status) {
  return el('span', { class: `pill ${STATUS_PILL[status] || 'pill-muted'}` }, [
    el('span', { class: 'dot' }),
    el('span', { text: STATUS_LABEL[status] || status }),
  ]);
}

const fmtBytes = (n) => n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
const fmtSecs = (s) => !s ? '—' : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
const fmtTime = (iso) => iso ? new Date(iso).toLocaleString() : '—';

/* ------------------------------------------------------------- readiness */
async function loadReadiness() {
  const pillNode = $('readiness');
  const text = $('readinessText');
  try {
    const r = await api('/health/ready');
    const ready = r.status === 'ready';
    pillNode.className = `pill ${ready ? 'pill-ok' : 'pill-warn'}`;
    const bits = [`STT: ${r.stt_backend}`, `LLM: ${r.llm_reachable ? (r.llm_model || 'up') : 'unreachable'}`, `${r.workers} workers`];
    text.textContent = ready ? `ready · ${bits.join(' · ')}` : `degraded · ${bits.join(' · ')}`;
    pillNode.title = Object.entries(r.details || {}).map(([k, v]) => `${k}: ${v}`).join('\n') || 'All dependencies healthy';
  } catch (err) {
    pillNode.className = 'pill pill-err';
    text.textContent = 'backend unreachable';
    pillNode.title = String(err.message || err);
  }
}

/* ------------------------------------------------------------ file upload */
function renderFileList() {
  const list = $('fileList');
  list.replaceChildren();

  state.files.forEach((file, index) => {
    list.append(el('li', {}, [
      el('span', { class: 'name', text: file.name, title: file.name }),
      el('span', { class: 'row' }, [
        el('span', { class: 'muted tiny', text: fmtBytes(file.size) }),
        el('button', {
          attrs: { type: 'button', 'aria-label': `Remove ${file.name}` }, text: '✕',
          on: { click: () => { state.files.splice(index, 1); renderFileList(); } },
        }),
      ]),
    ]));
  });

  const any = state.files.length > 0;
  $('uploadBtn').disabled = !any;
  $('clearBtn').disabled = !any;
  $('uploadBtn').textContent = any
    ? `Process ${state.files.length} call${state.files.length > 1 ? 's' : ''}`
    : 'Process calls';
}

function addFiles(fileList) {
  const allowed = state.allowed;
  const rejected = [];
  for (const file of fileList) {
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (allowed.length && !allowed.includes(ext)) { rejected.push(`${file.name} (${ext})`); continue; }
    if (file.size > state.maxMb * 1048576) { rejected.push(`${file.name} (too large)`); continue; }
    if (state.files.some((f) => f.name === file.name && f.size === file.size)) continue;
    if (state.files.length >= state.maxFiles) { rejected.push(`${file.name} (batch limit)`); continue; }
    state.files.push(file);
  }
  if (rejected.length) toast(`Skipped: ${rejected.slice(0, 3).join(', ')}${rejected.length > 3 ? `, +${rejected.length - 3} more` : ''}`, 'err');
  renderFileList();
}

function uploadFiles() {
  if (!state.files.length) return;

  const form = new FormData();
  form.append('name', $('batchName').value);
  for (const file of state.files) form.append('files', file, file.name);

  const progress = $('uploadProgress');
  const bar = progress.querySelector('.bar');
  progress.classList.remove('hidden');
  $('uploadBtn').disabled = true;

  // XHR rather than fetch: it reports upload progress, which matters when
  // someone drops 200 recordings at once.
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${API}/jobs`);
  xhr.upload.addEventListener('progress', (event) => {
    if (event.lengthComputable) bar.style.width = `${(event.loaded / event.total) * 100}%`;
  });
  xhr.addEventListener('load', () => {
    progress.classList.add('hidden');
    bar.style.width = '0';
    $('uploadBtn').disabled = false;

    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }

    if (xhr.status >= 200 && xhr.status < 300) {
      state.files = [];
      $('batchName').value = '';
      renderFileList();
      toast(`Batch queued — ${body.total} call${body.total > 1 ? 's' : ''} accepted`, 'ok');
      (body.rejected || []).forEach((r) => toast(`Rejected ${r.filename}: ${r.reason}`, 'err'));
      loadJobs().then(() => selectJob(body.job_id));
    } else {
      toast(body?.error?.message || `Upload failed (${xhr.status})`, 'err');
    }
  });
  xhr.addEventListener('error', () => {
    progress.classList.add('hidden');
    $('uploadBtn').disabled = false;
    toast('Upload failed — network error', 'err');
  });
  xhr.send(form);
}

/* --------------------------------------------------------------- job list */
async function loadJobs() {
  try {
    const data = await api('/jobs?limit=60');
    state.jobs = data.jobs;
    renderJobList();
  } catch (err) {
    toast(`Could not load batches: ${err.message}`, 'err');
  }
}

function renderJobList() {
  const list = $('jobList');
  list.replaceChildren();

  if (!state.jobs.length) {
    list.append(el('li', { class: 'muted tiny', text: 'No batches yet.' }));
    return;
  }

  for (const job of state.jobs) {
    const item = el('li', {
      class: job.id === state.jobId ? 'active' : '',
      on: { click: () => selectJob(job.id) },
    }, [
      el('div', { class: 'job-row' }, [
        el('span', { class: 'job-name', text: job.name || job.id, title: job.name }),
        pill(job.status),
      ]),
      el('div', { class: 'job-row muted tiny' }, [
        el('span', { text: `${job.completed}/${job.total} classified${job.failed ? ` · ${job.failed} failed` : ''}` }),
        el('span', { text: fmtTime(job.created_at) }),
      ]),
    ]);
    list.append(item);
  }
}

/* ---------------------------------------------------------- job selection */
async function selectJob(jobId) {
  state.jobId = jobId;
  stopStream();
  renderJobList();
  $('emptyState').classList.add('hidden');
  $('jobPanel').classList.remove('hidden');

  await refreshJob();
  if (state.job && !isTerminal(state.job.status)) startStream(jobId);
}

const isTerminal = (status) => ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(status);

async function refreshJob() {
  if (!state.jobId) return;
  try {
    state.job = await api(`/jobs/${state.jobId}`);
    renderJob();
    await loadResults();
  } catch (err) {
    toast(`Could not load batch: ${err.message}`, 'err');
    if (String(err.message).includes('not found')) clearSelection();
  }
}

function clearSelection() {
  stopStream();
  state.jobId = null; state.job = null; state.rows = [];
  $('jobPanel').classList.add('hidden');
  $('emptyState').classList.remove('hidden');
}

function renderJob() {
  const job = state.job;
  if (!job) return;

  $('jobTitle').textContent = job.name || job.id;
  $('jobMeta').textContent = `${job.id} · created ${fmtTime(job.created_at)}${job.finished_at ? ` · finished ${fmtTime(job.finished_at)}` : ''}`;

  $('jobBar').style.width = `${job.progress}%`;

  // replaceChildren rejects null, so the optional pieces are filtered out.
  $('jobCounts').replaceChildren(...[
    pill(job.status),
    el('span', { text: `  ${job.completed} of ${job.total} classified` }),
    job.failed ? el('span', { text: ` · ${job.failed} failed` }) : null,
    job.pending ? el('span', { text: ` · ${job.pending} pending` }) : null,
    job.error ? el('span', { text: ` · ${job.error}` }) : null,
  ].filter(Boolean));

  $('cancelBtn').classList.toggle('hidden', isTerminal(job.status));
  $('downloadBtn').disabled = job.total === 0;

  // Per-call progress is only interesting while work is in flight.
  const progressList = $('callProgress');
  progressList.replaceChildren();
  if (!isTerminal(job.status)) {
    for (const call of job.calls) {
      progressList.append(el('li', {}, [
        el('span', { class: 'sr', text: String(call.sr_no) }),
        el('span', { class: 'fname', text: call.filename, title: call.filename }),
        el('span', { class: 'stage', text: call.status === 'running' ? call.stage : call.status }),
        el('span', { class: 'stage', text: `${call.progress}%` }),
      ]));
    }
  }
}

/* ----------------------------------------------------------------- report */
async function loadResults() {
  if (!state.jobId) return;
  try {
    const data = await api(`/jobs/${state.jobId}/results?transcript_chars=600`);
    state.rows = data.rows;
    renderReportHead(data.columns);
    renderRows();
    renderDistribution();
    $('reportNote').textContent = data.transcripts_truncated
      ? 'Transcripts are shortened in this table — open a row for the full text, or download the Excel report.'
      : '';
  } catch (err) {
    toast(`Could not load the report: ${err.message}`, 'err');
  }
}

function renderReportHead(columns) {
  const head = $('reportHead');
  head.replaceChildren();
  for (const column of columns) head.append(el('th', { text: column }));
  head.append(el('th', { text: 'Status' }));
}

function currentFilters() {
  return {
    text: $('rowFilter').value.trim().toLowerCase(),
    theme: $('themeFilter').value,
  };
}

function renderRows() {
  const body = $('reportBody');
  body.replaceChildren();
  const { text, theme } = currentFilters();

  const rows = state.rows.filter((row) => {
    if (theme && row.theme !== theme) return false;
    if (!text) return true;
    return [row.file_name, row.theme, row.specific_issue, row.reason_for_issue]
      .join(' ').toLowerCase().includes(text);
  });

  // Rebuild the theme filter from what the batch actually produced.
  const select = $('themeFilter');
  const themes = [...new Set(state.rows.map((r) => r.theme))].sort();
  if (select.dataset.signature !== themes.join('|')) {
    select.dataset.signature = themes.join('|');
    const previous = select.value;
    select.replaceChildren(el('option', { attrs: { value: '' }, text: 'All themes' }));
    for (const name of themes) select.append(el('option', { attrs: { value: name }, text: name }));
    select.value = themes.includes(previous) ? previous : '';
  }

  if (!rows.length) {
    body.append(el('tr', { class: 'empty-row' }, el('td', { attrs: { colspan: '7' }, text: 'No rows match.' })));
    return;
  }

  for (const row of rows) {
    body.append(el('tr', {
      class: row.status === 'failed' ? 'failed' : '',
      on: { click: () => openDrawer(row.call_id) },
    }, [
      el('td', { class: 'sr', text: String(row.sr_no) }),
      el('td', { text: row.file_name, title: row.file_name }),
      el('td', { text: row.theme }),
      el('td', { text: row.specific_issue }),
      el('td', { class: 'clip', text: row.transcription || '—' }),
      el('td', { class: 'clip', text: row.ai_reasoning || '—' }),
      el('td', {}, pill(row.status)),
    ]));
  }
}

function renderDistribution() {
  const body = $('distributionBody');
  body.replaceChildren();

  const done = state.rows.filter((r) => r.status === 'completed');
  if (!done.length) {
    body.append(el('tr', { class: 'empty-row' }, el('td', { attrs: { colspan: '4' }, text: 'Nothing classified yet.' })));
    return;
  }

  // Key as JSON, not a delimiter join: theme and issue names both contain
  // spaces, so a plain string key cannot be split back apart reliably.
  const counts = new Map();
  for (const row of done) {
    const key = JSON.stringify([row.theme, row.specific_issue]);
    counts.set(key, (counts.get(key) || 0) + 1);
  }

  for (const [key, count] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
    const [theme, issue] = JSON.parse(key);
    body.append(el('tr', {}, [
      el('td', { text: theme }),
      el('td', { text: issue }),
      el('td', { class: 'num', text: String(count) }),
      el('td', { class: 'num', text: `${((count / done.length) * 100).toFixed(1)}%` }),
    ]));
  }
}

/* ----------------------------------------------------------------- drawer */
async function openDrawer(callId) {
  const drawer = $('drawer');
  const body = $('drawerBody');
  body.replaceChildren(el('p', { class: 'muted', text: 'Loading…' }));
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  $('scrim').hidden = false;

  try {
    const detail = await api(`/jobs/${state.jobId}/results/${callId}`);
    renderDrawer(detail);
  } catch (err) {
    body.replaceChildren(el('p', { class: 'muted', text: `Could not load detail: ${err.message}` }));
  }
}

function closeDrawer() {
  $('drawer').classList.remove('open');
  $('drawer').setAttribute('aria-hidden', 'true');
  $('scrim').hidden = true;
}

function renderDrawer(detail) {
  $('drawerTitle').textContent = detail.filename;
  $('drawerSub').textContent = `Sr. No ${detail.sr_no} · ${detail.id}`;

  const cls = detail.classification;
  const tr = detail.transcription;
  const tl = detail.translation;
  const parts = [];

  const facts = el('dl', { class: 'detail-grid' });
  const addFact = (label, value) => {
    if (value === null || value === undefined || value === '') return;
    facts.append(el('dt', { text: label }), el('dd', {}, typeof value === 'string' || typeof value === 'number'
      ? document.createTextNode(String(value)) : value));
  };

  addFact('Status', pill(detail.status));
  if (cls) {
    addFact('Theme', cls.theme);
    addFact('Specific issue', cls.issue);
    addFact('Reason for issue', cls.reason);
    addFact('Confidence', cls.confidence?.toFixed(2));
    addFact('Sentiment', cls.sentiment);
    if (!cls.theme_matched) addFact('⚠︎ Note', 'The model proposed a theme outside the taxonomy; it was remapped.');
    else if (!cls.issue_matched) addFact('⚠︎ Note', 'The issue is not one of the predefined issues for this theme.');
  }
  if (tr) {
    addFact('Language', tr.language);
    addFact('Duration', fmtSecs(tr.duration_seconds));
    addFact('Segments', tr.segment_count);
    const flags = Object.entries(tr.silence_flags || {}).filter(([, v]) => v).map(([k]) => k);
    if (flags.length) addFact('Silence flags', flags.join(', '));
  }
  if (tl) addFact('Translated', tl.translated ? 'yes — Arabic source' : 'no — already English');
  addFact('Timings', [
    detail.timings?.convert && `convert ${detail.timings.convert}s`,
    detail.timings?.transcribe && `STT ${detail.timings.transcribe}s`,
    detail.timings?.translate && `translate ${detail.timings.translate}s`,
    detail.timings?.classify && `classify ${detail.timings.classify}s`,
    detail.timings?.total && `total ${detail.timings.total}s`,
  ].filter(Boolean).join(' · '));
  if (detail.error) addFact('Error', `${detail.failed_stage || 'unknown'}: ${detail.error}`);
  parts.push(facts);

  if (cls?.reasoning) {
    parts.push(el('div', { class: 'section-title', text: 'AI reasoning' }));
    parts.push(el('p', { text: cls.reasoning }));
  }
  if (cls?.evidence?.length) {
    parts.push(el('div', { class: 'section-title', text: 'Evidence from the call' }));
    for (const quote of cls.evidence) parts.push(el('p', { class: 'quote', text: `“${quote}”` }));
  }
  if (tl?.text) {
    parts.push(el('div', { class: 'section-title', text: 'Final transcript (English)' }));
    parts.push(el('pre', { class: 'transcript', text: tl.text }));
  }
  if (tr?.text && tr.text !== tl?.text) {
    parts.push(el('div', { class: 'section-title', text: 'Original transcript' }));
    parts.push(el('pre', { class: `transcript ${tr.language === 'ar' ? 'rtl' : ''}`, text: tr.text }));
  }

  $('drawerBody').replaceChildren(...parts);
}

/* ------------------------------------------------------- live progress */
function startStream(jobId) {
  stopStream();
  try {
    const stream = new EventSource(`${API}/jobs/${jobId}/events`);
    state.stream = stream;

    stream.addEventListener('update', (event) => {
      if (state.jobId !== jobId) return;
      state.job = JSON.parse(event.data);
      renderJob();
      const summary = state.jobs.find((j) => j.id === jobId);
      if (summary) {
        Object.assign(summary, {
          status: state.job.status, completed: state.job.completed,
          failed: state.job.failed, total: state.job.total,
        });
        renderJobList();
      }
    });

    stream.addEventListener('done', () => {
      stopStream();
      refreshJob();
      loadJobs();
      toast('Batch finished', 'ok');
    });

    stream.addEventListener('deleted', () => { stopStream(); clearSelection(); loadJobs(); });

    stream.onerror = () => {
      // Fires both on a real failure and just after a normal close, so ignore
      // it once we have already torn the stream down ourselves.
      if (state.stream !== stream) return;
      // Proxies that buffer event streams land here - degrade to polling.
      stopStream();
      if (state.jobId === jobId) startPolling(jobId);
    };
  } catch {
    startPolling(jobId);
  }
}

function startPolling(jobId) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    if (state.jobId !== jobId) return stopPolling();
    await refreshJob();
    if (state.job && isTerminal(state.job.status)) { stopPolling(); loadJobs(); }
  }, 2500);
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

function stopStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
  stopPolling();
}

/* --------------------------------------------------------------- actions */
async function cancelJob() {
  if (!state.jobId) return;
  try {
    await api(`/jobs/${state.jobId}/cancel`, { method: 'POST' });
    toast('Cancellation requested — calls already running will finish.');
    await refreshJob();
    await loadJobs();
  } catch (err) { toast(err.message, 'err'); }
}

async function deleteJob() {
  if (!state.jobId) return;
  const job = state.job;
  if (!confirm(`Delete "${job?.name || state.jobId}" and its uploads, transcripts and report?`)) return;
  try {
    await api(`/jobs/${state.jobId}`, { method: 'DELETE' });
    toast('Batch deleted', 'ok');
    clearSelection();
    await loadJobs();
  } catch (err) { toast(err.message, 'err'); }
}

function downloadExcel() {
  if (!state.jobId) return;
  window.location.href = `${API}/jobs/${state.jobId}/export`;
}

/* ------------------------------------------------------------------- init */
function initTheme() {
  const stored = localStorage.getItem('ta-theme');
  const dark = stored ? stored === 'dark'
    : window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
}

async function initLimits() {
  try {
    // Take the limits from the server so client-side validation cannot drift
    // out of step with what the API will actually accept.
    const config = await api('/config');
    state.maxFiles = config.max_files_per_job;
    state.maxMb = config.max_upload_mb;
    state.allowed = config.allowed_extensions;
    $('fileInput').setAttribute('accept', config.allowed_extensions.join(','));
  } catch {
    /* keep the defaults in `state` - the server still enforces the real ones */
  }
  $('limitHint').textContent =
    `up to ${state.maxFiles} files per batch, ${state.maxMb} MB each`;
}

function wireEvents() {
  const dropzone = $('dropzone');
  const input = $('fileInput');

  dropzone.addEventListener('click', () => input.click());
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });

  for (const type of ['dragenter', 'dragover']) {
    dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
  }
  for (const type of ['dragleave', 'drop']) {
    dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); });
  }
  dropzone.addEventListener('drop', (event) => addFiles(event.dataTransfer.files));

  $('uploadForm').addEventListener('submit', (event) => { event.preventDefault(); uploadFiles(); });
  $('clearBtn').addEventListener('click', () => { state.files = []; renderFileList(); });
  $('refreshJobs').addEventListener('click', () => { loadJobs(); loadReadiness(); });
  $('cancelBtn').addEventListener('click', cancelJob);
  $('deleteBtn').addEventListener('click', deleteJob);
  $('downloadBtn').addEventListener('click', downloadExcel);
  $('drawerClose').addEventListener('click', closeDrawer);
  $('scrim').addEventListener('click', closeDrawer);
  $('rowFilter').addEventListener('input', renderRows);
  $('themeFilter').addEventListener('change', renderRows);

  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeDrawer(); });

  $('themeToggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    localStorage.setItem('ta-theme', dark ? 'light' : 'dark');
  });

  window.addEventListener('beforeunload', stopStream);
}

(async function main() {
  initTheme();
  wireEvents();
  renderFileList();
  await Promise.all([loadReadiness(), loadJobs(), initLimits()]);
  setInterval(loadReadiness, 30000);
})();
