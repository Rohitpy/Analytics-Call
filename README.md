# Theme Analytics

Call centre theme detection. Upload call recordings, and each one is
transcribed with Whisper large-v3, translated to English when it is not already
English, and classified by an LLM into a **theme**, a **specific issue**, and
the **reason for that issue** — then written to Excel.

Everything is an API. The Streamlit UI is a client of the same endpoints
documented at `/docs`; it has no privileged access and runs as its own
process, so it can sit on a different host from the GPU box entirely.

---

## The pipeline

```
upload ──▶ ffmpeg ──▶ Whisper large-v3 ──▶ LLM translate ──▶ LLM classify ──▶ Excel
           16 kHz     + silero VAD         (Arabic only)     theme / issue /
           mono PCM   chunked on silence                     reason + reasoning
```

Each stage is throttled where it actually costs something, so the GPU and the
inference server never fight each other:

| Stage | Bounded by | Default |
|---|---|---|
| ffmpeg | subprocess, naturally parallel | — |
| Whisper | `STT_CONCURRENCY` (thread pool) | 1 |
| translate + classify | `LLM_MAX_CONCURRENCY` (semaphore) | 8 |
| calls in flight | `PIPELINE_WORKERS` | 4 |

`PIPELINE_WORKERS` can safely exceed `STT_CONCURRENCY`: while one call sits on
the GPU, the others are waiting on the LLM.

---

## Layout

```
backend/
  main.py                 app factory, lifespan, static mount
  container.py            composition root — every service built once
  core/
    config.py             all settings (nothing else reads os.getenv)
    logging_config.py     rotating file + stdout, job/call/stage contextvars
    errors.py             typed errors -> JSON responses
    yaml_compat.py        works with PyYAML or ruamel.yaml
  schemas/
    common.py             JobStatus / CallStatus / Stage enums
    job.py                JobRecord, CallRecord, ResultRow, API responses
    theme.py              taxonomy models + stage outputs
  api/
    deps.py               FastAPI dependencies
    v1/router.py          route registration
    v1/routes/            health.py  jobs.py  results.py  themes.py
  services/
    audio.py              ffmpeg -> 16 kHz mono PCM (async subprocess)
    transcription.py      Whisper + silero VAD (ported from app.py) + mock backend
    translation.py        Arabic -> English, skipped when already English
    classification.py     theme/issue/reason + JSON parsing and validation
    prompt_builder.py     taxonomy -> prompt, {{placeholder}} rendering
    taxonomy.py           loads themes.yaml, hot-reloadable
    pipeline.py           the four stages for one call (stateless)
    job_manager.py        queue + worker pool + report finalisation
    job_store.py          job state, in memory + JSON mirror on disk
    excel.py              report writer (also builds the API's rows)
    upload.py             validation + streaming save
  prompts/
    translation.yaml      stage 2 prompt
    classification.yaml   stage 3 prompt
  data/
    themes.yaml           THE TAXONOMY — see "Tuning the classifier"
streamlit_app/                        the UI - a pure API client, own process
  app.py                  entry point, tabs, timed refresh
  config.py               API URL and poll interval (env-driven)
  api_client.py           thin wrapper over the REST API
  formatting.py           table shaping - no streamlit import, so it is testable
  views/
    sidebar.py            health, upload form, batch picker
    job_panel.py          status, counters, progress, download/cancel/delete
    report.py             the six report columns + theme rollup
    call_detail.py        classification, evidence, both transcripts
    taxonomy.py           browse themes, hot-reload themes.yaml
.streamlit/config.toml                headless mode, upload size cap, theme
storage/                              uploads, work, results, jobs, logs
app.py                                the original script, kept for reference
```

---

## Setup

Requires Python 3.13. Everything in `requirements.txt` is already present in
the target environment; the file exists to pin the floors.

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit
```

Point `.env` at your infrastructure:

```ini
WHISPER_MODEL_PATH=/data0/genaiadm_bkp/GenAi-LLM/whisper/large-v3.pt
WHISPER_DEVICE=cuda:0
FFMPEG_PATH=/opt/genaiadm/call_quality/ffmpeg/ffmpeg
LLM_BASE_URL=http://localhost:8003/v1
LLM_MODEL=gemma-4
```

Run it — this starts **two** processes, the API and the Streamlit UI:

```bash
./run.sh                # API on :8000, UI on :8501
./run.sh --api-only     # just the API (use this for a systemd unit)
./run.sh --ui-only      # just the UI, against an API elsewhere
```

Open the UI at `http://<server-ip>:8501`, and the API docs at
`http://<server-ip>:8000/docs`. `run.sh` prints both URLs on startup.

Equivalent by hand:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
python -m streamlit run streamlit_app/app.py --server.port 8501
```

Two things that bite:

> **Use a single uvicorn worker.** Job state and the queue live in the
> process. Running `--workers 4` would give you four independent queues that
> cannot see each other's jobs. Scale with `PIPELINE_WORKERS`, not with
> uvicorn workers — see [Scaling out](#scaling-out).

> **Open port 8501, not just 8000.** The UI is what people use; the API port
> only needs to be reachable from wherever Streamlit runs. If they share a
> box, `THEME_ANALYTICS_API_URL` defaults to `http://127.0.0.1:8000` and the
> API can stay on loopback.

### The UI

Streamlit, targeting **1.30** (the version on the deployment box) — so no
`st.fragment`, `st.dialog`, or dataframe row-selection, all of which landed
later. It reaches the backend over HTTP only and holds no pipeline logic.

| Env var | Default | Purpose |
|---|---|---|
| `THEME_ANALYTICS_API_URL` | `http://127.0.0.1:8000` | Where the UI finds the API |
| `THEME_ANALYTICS_POLL_SECONDS` | `3` | Refresh interval while a batch runs |
| `THEME_ANALYTICS_PREVIEW_CHARS` | `600` | Transcript characters in the table |
| `UI_PORT` | `8501` | Streamlit port (read by `run.sh`) |

Streamlit has no push channel, so live progress is a timed rerun rather than
the SSE stream. The `/jobs/{id}/events` endpoint is still there and still
works — it's just for API consumers now. The refresh is a checkbox, on by
default, and only appears while a batch is actually running.

Raise `maxUploadSize` in `.streamlit/config.toml` (currently 2048 MB) if your
batches are bigger than that — Streamlit buffers the whole upload in memory
before forwarding it, and its own default cap is 200 MB.

### Developing without a GPU

```bash
STT_BACKEND=mock python -m backend.main
```

The mock backend returns a fixed English transcript, so the API, queue,
report, and UI can all be exercised on a laptop. Whisper, torch, librosa and
silero are imported lazily and are never touched in this mode.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | Upload files (multipart `files`, optional `name`). Returns `202` immediately. |
| `GET` | `/api/v1/jobs` | List batches, newest first. |
| `GET` | `/api/v1/jobs/{id}` | Status and per-call progress. |
| `GET` | `/api/v1/jobs/{id}/events` | SSE progress stream; closes when the batch finishes. |
| `POST` | `/api/v1/jobs/{id}/cancel` | Drop queued calls; running ones finish. |
| `DELETE` | `/api/v1/jobs/{id}` | Delete the batch and its artefacts. |
| `GET` | `/api/v1/jobs/{id}/results` | The report rows. `?transcript_chars=N` clips transcripts (`0` = full). |
| `GET` | `/api/v1/jobs/{id}/results/{call_id}` | Full detail for one call, both transcripts included. |
| `GET` | `/api/v1/jobs/{id}/export` | Download the `.xlsx`. Works mid-batch too. |
| `GET` | `/api/v1/themes` | The active taxonomy. |
| `POST` | `/api/v1/themes/reload` | Re-read `themes.yaml` with no restart. |
| `GET` | `/api/v1/themes/rendered` | The taxonomy exactly as the model sees it. |
| `POST` | `/api/v1/themes/prompt-preview` | Render the classification prompt for a sample transcript. |
| `GET` | `/api/v1/config` | Upload limits, published so clients validate identically. |
| `GET` | `/api/v1/health`, `/health/ready` | Liveness; readiness probes ffmpeg, the STT model and the LLM. |

```bash
# upload a batch
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "name=Retail queue" -F "files=@call1.wav" -F "files=@call2.wav"

# watch it
curl -N http://localhost:8000/api/v1/jobs/job_abc123/events

# download the report
curl -OJ http://localhost:8000/api/v1/jobs/job_abc123/export
```

---

## The report

Sheet **Theme Analysis** holds exactly the six requested columns:

| Sr. No | FileName | Theme | Specific Issue for the call | Transcription | AI Reasoning |
|---|---|---|---|---|---|

`Transcription` is the final English transcript — the text the classifier
actually read. `AI Reasoning` packs the reason for the issue, the model's
justification, its verbatim evidence quotes, and its confidence into one cell.

Two more sheets come along:

- **Details** — reason, confidence, language, duration, evidence, the original
  Arabic transcript (right-to-left), and the error for anything that failed.
- **Summary** — batch metadata and a theme × issue rollup with percentages.

A call that fails is still a row: theme `PROCESSING FAILED`, the stage it died
at, and the error. One bad recording never costs you the batch.

---

## Tuning the classifier

`backend/data/themes.yaml` **is** the prompt. It is rendered into the
classification request at call time, so editing it changes behaviour with no
code change:

```yaml
- name: "Cards"                     # exact string in the Theme column
  description: >-                   # when to use this theme, and when not to
    Anything about a physical or virtual debit, credit or prepaid card...
  keywords: [card, pin, declined]   # surface cues
  issues:
    - name: "Card not received"     # exact string in the Specific Issue column
      description: Card was issued but never reached the customer.
      reasons:                      # candidates the model picks from / adapts
        - "Delivery address on file is outdated"
        - "Card dispatch delayed beyond SLA"
      examples:                     # ← the highest-value field
        - "Customer applied three weeks ago; agent finds the courier returned it."
```

Then:

```bash
curl -X POST http://localhost:8000/api/v1/themes/reload
```

Practical notes from building this:

- **Examples do the work.** Two or three concrete paraphrased calls per issue
  outperform any amount of abstract description. They are what teach the model
  where the boundary between two themes sits.
- **Give every theme an "Other …" issue.** Otherwise the model is forced into a
  wrong specific issue, and a wrong specific issue is worse than an honest
  vague one.
- **State the precedence rules.** `classification.yaml` tells the model that a
  denied transaction is Fraud and Disputes even though a card is involved, and
  that a broken service promise is Customer Service Experience even though a
  product is involved. Without those, overlapping themes get classified
  inconsistently.
- **Watch the guardrails.** The theme is enum-constrained through vLLM's
  `guided_json`, then re-validated in Python: exact match, then case-insensitive,
  then fuzzy. An invented theme is remapped to `Others` and flagged; an issue
  outside the theme's list is kept verbatim but flagged, because that flag is
  how you find a gap in your taxonomy. Both flags surface as a note in the
  `AI Reasoning` cell and in the UI drawer.
- **Preview before you commit.** `POST /api/v1/themes/prompt-preview` returns
  the exact system and user messages for a transcript you supply, without
  spending a GPU cycle.

The prompt files themselves are `backend/prompts/*.yaml`, using
`{{placeholder}}` tokens (double braces, so the literal JSON braces in the
prompts are left alone). They are re-read whenever their mtime changes.

---

## Scaling out

The current taxonomy renders to roughly **33k characters (~8k tokens)** of
prompt per call, and the prefix is byte-identical across every call in a batch.
Serve gemma-4 with prefix caching on and that prefix is computed once:

```bash
vllm serve <model> --port 8003 --enable-prefix-caching
```

Then tune, in this order:

1. `LLM_MAX_CONCURRENCY` up to what your inference server sustains.
2. `PIPELINE_WORKERS` to roughly `LLM_MAX_CONCURRENCY`.
3. `STT_CONCURRENCY` only if you have more than one GPU.

Beyond one box, two things have to change, and only two: `JobStore` moves to
Redis or Postgres, and the `asyncio.Queue` in `JobManager` becomes a real
broker. Nothing else in the codebase touches job state or the queue directly.

Other operational notes:

- **Progress writes are deferred.** A job record embeds every transcript, so
  per-stage updates mark the job dirty and a 5-second flusher persists them.
  Terminal transitions are written immediately.
- **Restarts are honest.** Jobs still marked running at startup are rehydrated
  and marked interrupted, because the in-process queue did not survive. They
  are never left claiming to be in progress forever.
- **Completion is claimed atomically.** The decision "am I the worker that
  finished this batch?" happens inside the store mutation, under the store's
  lock, so the report is written exactly once no matter how many workers land
  at the same instant.
- **Logs** go to stdout and `storage/logs/theme_analytics.log` (20 MB × 10).
  Every line carries `job=… call=… stage=…`, so one call is greppable across
  all four stages. `LOG_JSON=true` for structured output.

---

## Relationship to `app.py`

The original script is untouched and still runnable. The transcription logic
came across essentially verbatim — VAD-boundary chunking, the per-segment
`no_speech_prob` / `avg_logprob` / `compression_ratio` gates, the silence
flags — because that part was already doing the right thing. What changed
around it:

| `app.py` | here |
|---|---|
| model loaded at import | loaded once behind a lock, warmed at startup |
| blocking `for` loop over a folder | queue + async worker pool |
| a whole batch dies on one bad file | per-call failure isolation |
| `language="ar"` hardcoded | `WHISPER_LANGUAGE`, `auto` supported |
| translation only | translation + theme/issue/reason classification |
| prompt in a string literal | YAML prompts + hot-reloadable taxonomy |
| `filename / transcription / translation` | the six-column report + 2 sheets |
| `print(json.dumps(...))` | REST API + a Streamlit UI |

One latent bug is worth mentioning since it survives in `app.py`:
`JSONToSRTTranslator.load_prompt_from_yaml` calls `yaml.safe_load` but the
module never imports `yaml`, so that path raises `NameError` if reached. The
rewrite loads YAML through `core/yaml_compat.py`, which also removes the hard
dependency on PyYAML specifically.
