![1781950321671](image/README/1781950321671.png)

# scribe-lite : a speech-to-text transcription-service

Single-file speech-to-text service built on [WhisperX](https://github.com/m-bain/whisperX) for long-form audio: transcribes, aligns precise word timestamps, tells speakers apart, and writes a clean speaker-attributed script. Exposed through a CLI and an async HTTP API, and audio never leaves your machine.

Everything lives in [`transcribe.py`](transcribe.py), plus [`output_format_adapters.py`](output_format_adapters.py) for rendering to `.md` / `.vtt`. Configuration is read from a `.env` file — copy the committed [`.env.example`](.env.example) and set `HF_TOKEN`.

## Installation

Built and tuned on macOS / Apple Silicon. You need Python ≥ 3.12, [`uv`](https://docs.astral.sh/uv/), and **ffmpeg** on your PATH:

```bash
brew install uv ffmpeg          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

Speaker diarization uses a gated Hugging Face model, so it needs one-time
access setup (skip this if you only ever run `--no-diarize`):

1. Open the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1) and accept the **user conditions**.
2. Create a read access token at <https://huggingface.co/settings/tokens>.

Then, from a fresh clone:

```bash
uv sync                         # install dependencies (fetches Python 3.12 if needed)
cp .env.example .env            # then paste the token into HF_TOKEN
mkdir -p Data                   # default input location
```

The first run downloads the models (Whisper `large-v3-turbo`, the alignment
model, and the diarization pipeline — a few GB) and caches them locally;
subsequent runs are offline.

## Supported audio

Anything ffmpeg can decode: `mp3`, `m4a`/`aac`, `wav`, `ogg`/`opus`, `flac`,
`wma`, `aiff`, and video files (`mp4`, `mov`, `mkv`, … — the audio track is
extracted). Inputs are normalized to 16 kHz mono automatically, so no
pre-processing is needed.

## Quality

Measured on internal two-speaker meeting audio (English; expect different
numbers on other material):

- **Transcription** — Whisper `large-v3-turbo`, ~8.8% word error rate.
- **Diarization** — the default `fast` mode attributes 98.1–98.4% of words to
  the same speaker as the full pyannote pipeline; the residual is almost
  entirely genuinely overlapped speech (both voices active at once). Anything
  the fast mode can't verify escalates to full pyannote automatically.

## Running (CLI)

A bare filename is looked up under `./Data`; output lands in `./Output`:

```bash
cp /path/to/meeting.m4a Data/
uv run python3 transcribe.py transcribe meeting.m4a
# -> Output/meeting.md
```

### Examples

```bash
# transcription only -- same output shape, every segment labeled SPEAKER_00
uv run python3 transcribe.py transcribe meeting.m4a --no-diarize

# force the full pyannote diarizer (most thorough)
uv run python3 transcribe.py transcribe meeting.m4a --diarize-mode pyannote

# WebVTT output with a speaker-count hint
uv run python3 transcribe.py transcribe meeting.m4a \
    --format vtt --min-speakers 2 --max-speakers 3

# explicit input path, raw JSON to stdout
uv run python3 transcribe.py transcribe /abs/path/audio.mp3 --json
```

### Options

| flag | description |
| ---- | ----------- |
| `--no-diarize` | Skip speaker diarization entirely: transcript with word timestamps only, same md/vtt output with every segment labeled `SPEAKER_00` (and no `HF_TOKEN` needed). Ignores the speaker options below. |
| `--diarize-mode fast\|auto\|pyannote` | Diarization strategy. `fast` (default): quick speaker detection that falls back to the full model automatically whenever it can't be trusted. `auto`: same, but falls back more eagerly. `pyannote`: always the full pipeline. |
| `--language en` | Language code; auto-detected if omitted. |
| `--min-speakers N`, `--max-speakers N` | Speaker-count hints for diarization. |
| `--format md\|vtt` | Output file format (default `md`). |
| `--output PATH`, `-o` | Output path; a bare name writes to `./Output/<name>`. |
| `--json` | Print the raw JSON script to stdout instead of writing a file. |
| `--no-resume` | Ignore saved checkpoints and rerun the full pipeline. |
| `--batch-size N` | ASR batch size (smaller = less memory). |

## Running (HTTP API)

```bash
uv run python3 transcribe.py serve          # HOST/PORT from .env; --host/--port to override
```

Interactive docs at `http://localhost:8000/docs`.

`POST /transcribe/async/` accepts the audio file (multipart) and returns a
`task_id` immediately; the pipeline runs in the background. Everything else
goes in the query string:

```bash
curl -X POST "http://localhost:8000/transcribe/async/?min_speakers=2&max_speakers=4" \
    -F "file=@meeting.m4a"
# -> { "task_id": "b1c2...d4", "status": "pending" }

curl -X POST "http://localhost:8000/transcribe/async/?diarize=false" \
    -F "file=@meeting.m4a"          # same as --no-diarize
```

Query params (all optional): `language`, `min_speakers`, `max_speakers`,
`diarize_mode` (`fast`|`auto`|`pyannote`), `diarize` (`false` = skip), `resume`.

Poll `GET /tasks/{task_id}` until `status` is `completed` (or `failed`); the
transcript is in `result`.

## Output

Markdown (`--format md`, default):

```markdown
**[00:00:00.000 → 00:00:01.500] SPEAKER_00**
Hello, welcome.

**[00:00:01.500 → 00:00:02.700] SPEAKER_01**
Thanks for having me.
```

WebVTT (`--format vtt`):

```webvtt
1
00:00:00.000 --> 00:00:01.500
<v SPEAKER_00>Hello, welcome.
```

With `--no-diarize` every segment is labeled `SPEAKER_00`. The raw JSON shape
(one entry per segment):

```json
[
  { "time": { "start": 0.0, "end": 1.234 }, "speaker": "SPEAKER_00", "text": "Hello, welcome." }
]
```

The same voice keeps the same `SPEAKER_XX` tag across files (delete
`.speakers.json` to reset the numbering).

## Configuration

All knobs live in [`.env.example`](.env.example) — copy it to `.env` and edit:
transcription model, devices, diarization options, `DATA_DIR`/`OUTPUT_DIR`,
output format, and more. Two defaults worth knowing:

- Every stage is checkpointed under `.transcribe_cache/`; re-running the same
  file resumes from the last completed stage (`--no-resume` forces a rerun,
  `rm -rf .transcribe_cache` wipes it).
- Real environment variables beat `.env` values.
