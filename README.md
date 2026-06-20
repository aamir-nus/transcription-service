![1781950321671](image/README/1781950321671.png)

# scribe-lite : a speech-to-text transcription-service

Single-file speech-to-text service built on [WhisperX](https://github.com/m-bain/whisperX) for long-form audio. No Docker — a sequential pipeline that transcribes, aligns, diarizes and produces a clean speaker script, exposed through a CLI and an async HTTP API.

Everything lives in [`transcribe.py`](transcribe.py), plus [`output_format_adapters.py`](output_format_adapters.py) for rendering the script to `.md` / `.vtt`. All settings are read from a `.env` file via `load_dotenv()`.

## Project layout

```
transcribe.py                  # engine + CLI + HTTP API
output_format_adapters.py      # md / vtt renderers + save()
.env                            # all configuration
Data/                           # default input audio location
Output/                         # default transcription output location (created on demand)
.transcribe_cache/             # resume checkpoints (gitignored, created on demand)
```

## Pipeline

The engine (`run_pipeline`) runs strictly one stage at a time, freeing each model before loading the next to keep memory flat. Logs are tagged `[n/7]` so you can see which step is running:

1. **`[1/7]` Convert** — normalize the input to 16 kHz mono mp3 via ffmpeg (any container: mp3, m4a, wav, aac, ogg, flac, …).
2. **`[2/7]` Transcribe** — load `large-v3-turbo`, load audio, run ASR with a small batch size.
3. **`[3/7]` Free ASR** — drop the model, reclaim memory.
4. **`[4/7]` Align** — load the alignment model, refine word timestamps, free it. (Skipped gracefully on unsupported languages.)
5. **`[5/7]` Diarize** — load the WhisperX diarization pipeline using `HF_TOKEN`, detect speakers, free it.
6. **`[6/7]` Assign speakers** — map speakers onto segments.
7. **`[7/7]` Build script** — emit a list of `{ "time": { "start", "end" }, "speaker", "text" }`.

Every stage logs how long it took, and the final line prints the total plus a
per-stage breakdown:

```
[done] pipeline complete: 412 segments in 98.4s (convert 1.2, asr 42.3, align 8.1, diarize 40.5, assign 0.4, build 0.1)
```

## Resume & checkpointing

A run can fail at any of the seven steps (OOM, a gated-model 401, …). To avoid
re-doing the expensive ASR pass after a late failure, each stage's output is
written to a per-job folder under `.transcribe_cache/`, keyed by a digest of the
**input file's content** + the params that affect the output. Re-running the same
file resumes from the last stage that completed.

Per-job folder (`.transcribe_cache/<digest>/`):

| file              | stage | what it stores                                       |
| ----------------- | ----- | ---------------------------------------------------- |
| `converted.mp3` | 1     | the normalized 16 kHz mono audio                     |
| `asr.json`      | 2     | raw ASR result (the expensive one)                   |
| `aligned.json`  | 4     | word-aligned result                                  |
| `diarize.json`  | 5     | speaker turns as `[{start, end, speaker}, …]`     |
| `final.json`    | 6     | result after speaker assignment                      |
| `script.json`   | 7     | the final clean script (a hit here = instant return) |
| `meta.json`     | —    | content hash, params, status, timings                |

So if diarization fails on a 3-hour file, you fix `HF_TOKEN`, re-run, and only
steps 5–7 re-run — the multi-gigabyte transcription is reused.

Cache-hit stages are logged (`[5/7] cache hit: diarization`). The cache key
excludes `batch_size` (it doesn't change the output), so tweaking it for memory
won't bust the cache. It **does** include model name, device, compute type,
language, speaker bounds, diarization model, and the WhisperX version.

- CLI: `--no-resume` ignores the cache and runs the full pipeline.
- HTTP: `?resume=false` on `POST /transcribe/async/` does the same.
- Disable globally with `RESUME=false` in `.env`.
- Wipe the cache any time: `rm -rf .transcribe_cache`.

## Prerequisites

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) (used to manage the venv)
- **ffmpeg** on your PATH — `brew install ffmpeg`
- **Hugging Face access to the diarization model.** The pipeline uses
  [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
  which is **gated**. Do this **first**, before running anything:

  1. Open the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
     and accept the **user conditions** (button near the top).
  2. Create a read access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
  3. Put it in `.env` as `HF_TOKEN=hf_xxx`.

  > If diarization later 401/403s on a sub-model, open that model's page and
  > accept its conditions too — but the main repo above is the required one.
  >

  Skip step 1/2 and diarization fails with a `401`/`403`. Thanks to the
  checkpointing below, you can fix the token and **resume from the last
  completed stage** — no re-transcription needed.

## Setup

```bash
# install dependencies (fastapi, uvicorn, whisperx, python-multipart, python-dotenv)
uv sync
```

Create a `.env` (already present — edit to taste):

```dotenv
HF_TOKEN=hf_xxx               # required for diarization
TRANSCRIPTION_MODEL=large-v3-turbo
DEVICE=mps                    # cpu | cuda | mps
COMPUTE_TYPE=int8             # NOTE: int8 (CUDA) requires CUDA (see below)
BATCH_SIZE=8                  # smaller = less memory
HOST=0.0.0.0
PORT=8000
DATA_DIR=Data                 # default input audio directory
OUTPUT_DIR=Output             # default output directory
DEFAULT_OUTPUT_FORMAT=md      # md | vtt  (used when --format and the -o suffix are absent)
RESUME=true                   # checkpoint each stage; re-runs resume from last success
CHECKPOINT_DIR=.transcribe_cache   # where checkpoints live (gitignored)
DIARIZE_MODEL=                # override diarization model (default: pyannote/speaker-diarization-community-1)
```

> **Device / compute type note:** `int8` quantization is a CUDA-only path in
> faster-whisper. On Apple Silicon (`DEVICE=mps`) use `COMPUTE_TYPE=float16`
> (or `default`); on CPU use `int8` (CPU int8) or `default`.

## CLI (local files)

Input audio is resolved as: an explicit path, or a bare filename looked up under
`./Data`. Output is written under `./Output` by default.

```bash
# transcribe ./Data/meeting.m4a -> ./Output/meeting.md (default format)
uv run python transcribe.py transcribe meeting.m4a

# force WebVTT
uv run python transcribe.py transcribe meeting.m4a --format vtt      # -> ./Output/meeting.vtt

# full options
uv run python transcribe.py transcribe /abs/path/audio.mp3 \
    --language en --min-speakers 2 --max-speakers 4 \
    --batch-size 8 --format md --output notes.md

# dump the raw JSON script to stdout instead of a formatted file
uv run python transcribe.py transcribe meeting.m4a --json

# ignore saved checkpoints and run the full pipeline from scratch
uv run python transcribe.py transcribe meeting.m4a --no-resume
```

### Output formats (`output_format_adapters.py`)

`save(script, path, fmt=None)` / `render(script, fmt)`. Format resolution:
explicit `fmt` → path suffix → `DEFAULT_OUTPUT_FORMAT` (default `md`). Supported:
`md`, `vtt`. (`docx` and others are placeholders for later.)

**Markdown** (`--format md`):

```markdown
# Transcript

**[00:00:00.000 → 00:01.500] SPEAKER_00**
Hello, welcome.

**[00:01.500 → 00:02.700] SPEAKER_01**
Thanks for having me.
```

**WebVTT** (`--format vtt`):

```webvtt
WEBVTT

1
00:00:00.000 --> 00:00:01.500
<v SPEAKER_00>Hello, welcome.

2
00:00:01.500 --> 00:00:02.700
<v SPEAKER_01>Thanks for having me.
```

Raw JSON shape (one entry per segment):

```json
[
  { "time": { "start": 0.0, "end": 1.234 }, "speaker": "SPEAKER_00", "text": "Hello, welcome." },
  { "time": { "start": 1.5, "end": 2.7 }, "speaker": "SPEAKER_01", "text": "Thanks for having me." }
]
```

## HTTP API (async)

Start the server:

```bash
uv run python transcribe.py serve                 # uses HOST/PORT from .env
uv run python transcribe.py serve --port 8000      # override
```

Interactive docs are at `http://localhost:8000/docs`.

### `POST /transcribe/async/`

Upload an audio file (multipart). Returns a `task_id` immediately and runs the
pipeline in the background.

```bash
curl -X POST http://localhost:8000/transcribe/async/ \
    -F "file=@path/to/audio.mp3" \
    -F "min_speakers=2" -F "max_speakers=4"
# -> { "task_id": "b1c2...d4", "status": "pending" }
```

Query params (all optional): `language`, `min_speakers`, `max_speakers`.

### `GET /tasks/{task_id}`

Poll until `status` is `completed` (or `failed`).

```bash
curl http://localhost:8000/tasks/b1c2...d4
```

```json
{
  "task_id": "b1c2...d4",
  "status": "completed",
  "created_at": 1718445646.997,
  "result": [
    { "time": { "start": 0.0, "end": 1.234 }, "speaker": "SPEAKER_00", "text": "..." }
  ],
  "error": null
}
```

Task states: `pending` → `processing` → `completed` (or `failed`). The task
store is in-memory and lives only for the lifetime of the server process.
