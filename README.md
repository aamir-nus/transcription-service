# transcription-service

Single-file speech-to-text service built on [WhisperX](https://github.com/m-bain/whisperX). No Docker — a sequential pipeline that transcribes, aligns, diarizes and produces a clean speaker script, exposed through a CLI and an async HTTP API.

Everything lives in [`transcribe.py`](transcribe.py), plus [`output_format_adapters.py`](output_format_adapters.py) for rendering the script to `.md` / `.vtt`. All settings are read from a `.env` file via `load_dotenv()`.

## Project layout

```
transcribe.py                  # engine + CLI + HTTP API
output_format_adapters.py      # md / vtt renderers + save()
.env                            # all configuration
Data/                           # default input audio location
Output/                         # default transcription output location (created on demand)
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

## Prerequisites

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) (used to manage the venv)
- **ffmpeg** on your PATH — `brew install ffmpeg`
- An **Hugging Face token** with access to the gated diarization model
  ([`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)).
  Accept the model conditions on the page, then create an access token.

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
