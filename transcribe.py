"""Single-file speech transcription service.

Wraps WhisperX into a sequential pipeline (transcribe -> align -> diarize -> script)
that loads and frees each model in turn to keep memory usage flat, and exposes it
through three front-ends:

  * ``run_pipeline``        -- the synchronous engine core
  * ``transcribe_async``    -- an async worker that runs the core off the event loop
  * FastAPI app             -- POST /transcribe/async/ + GET /tasks/{task_id}
  * CLI                     -- ``python transcribe.py transcribe <file>``

Run ``python transcribe.py serve`` to start the HTTP API.

All configuration is read from environment variables (see .env).
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Always refresh from .env so config changes are picked up without a reinstall.
load_dotenv(override=True)

import torch  # noqa: E402  (import after load_dotenv so CUDA/MPS env tweaks apply)
import uvicorn  # noqa: E402
import whisperx  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from whisperx.diarize import DiarizationPipeline  # noqa: E402

from output_format_adapters import (  # noqa: E402
    DEFAULT_FORMAT,
    SUPPORTED_FORMATS,
)
from output_format_adapters import (  # noqa: E402
    save as save_transcript,
)

# --------------------------------------------------------------------------- #
# Configuration (all from the environment)
# --------------------------------------------------------------------------- #
HF_TOKEN: Optional[str] = os.getenv(
    "HF_TOKEN"
)  # required for the gated diarization model
MODEL_NAME: str = os.getenv("TRANSCRIPTION_MODEL", "large-v3-turbo")
DEVICE: str = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "default")
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Default input/output locations (created on demand).
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "Data"))
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "Output"))
DEFAULT_OUTPUT_FORMAT: str = os.getenv("DEFAULT_OUTPUT_FORMAT", "md")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("transcribe")


# --------------------------------------------------------------------------- #
# Memory management
# --------------------------------------------------------------------------- #
def release_model(obj) -> None:
    """Drop a model reference and reclaim accelerator memory.

    WhisperX stages are heavy; calling this between pipeline steps is what keeps
    us from holding two large models resident at once.
    """
    del obj
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as exc:  # cache clearing is best-effort, never fatal
        log.debug("accelerator cache clear skipped: %s", exc)


# --------------------------------------------------------------------------- #
# Input normalization
# --------------------------------------------------------------------------- #
def convert_to_mp3(input_path: str, out_dir: Optional[str] = None) -> str:
    """Convert any audio container to a normalized 16 kHz mono MP3 via ffmpeg.

    Runs once at the start of the pipeline so every downstream stage works from
    the same format regardless of the input (mp3, m4a, wav, aac, ogg, flac, ...).
    Returns the path to the converted file (created in ``out_dir``).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg`)."
        )

    src = Path(input_path)
    work_dir = (
        Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="tscript-convert-"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{src.stem}.mp3"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",  # drop any embedded video stream
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz, speech-friendly; whisper resamples here anyway
        "-b:a",
        "64k",
        str(out_path),
    ]
    log.info("converting %s -> mp3", src.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"ffmpeg conversion failed: {proc.stderr.strip()[-500:] or 'unknown error'}"
        )
    return str(out_path)


# --------------------------------------------------------------------------- #
# Pipeline engine core (sequential, blocking)
# --------------------------------------------------------------------------- #
def run_pipeline(
    audio_path: str,
    *,
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Transcribe ``audio_path`` end-to-end and return a clean speaker script.

    Stages run strictly one at a time, each model freed before the next loads:

      1. convert the input to a normalized mp3 (ffmpeg)
      2. load ASR model (``MODEL_NAME``) -> load audio -> transcribe (small batch)
      3. free the ASR model
      4. load alignment model -> align word timestamps -> free it
      5. load diarization pipeline (uses ``HF_TOKEN``) -> diarize -> free it
      6. assign speakers to segments
      7. shape into ``[{"time": {"start", "end"}, "speaker", "text"}, ...]``

    The list of script dicts is returned.
    """
    convert_dir = Path(tempfile.mkdtemp(prefix="tscript-convert-"))
    try:
        # 1. normalize input to mp3 --------------------------------------- #
        log.info("[1/7] converting input to mp3")
        audio_path = convert_to_mp3(audio_path, out_dir=str(convert_dir))

        # 2. ASR ----------------------------------------------------------- #
        log.info(
            "[2/7] loading ASR model %s on %s (compute_type=%s)",
            MODEL_NAME,
            DEVICE,
            COMPUTE_TYPE,
        )
        model = whisperx.load_model(
            MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, use_auth_token=HF_TOKEN
        )

        log.info("[2/7] loading audio: %s", audio_path)
        audio = whisperx.load_audio(audio_path)

        log.info("[2/7] transcribing (batch_size=%d)", batch_size)
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        detected_language = result.get("language", language)

        # 3. free ASR model before touching alignment --------------------- #
        log.info("[3/7] releasing ASR model")
        release_model(model)

        # 4. alignment ----------------------------------------------------- #
        try:
            log.info("[4/7] aligning timestamps (language=%s)", detected_language)
            align_model, align_meta = whisperx.load_align_model(
                detected_language, DEVICE
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                align_meta,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            log.info("[4/7] releasing alignment model")
            release_model(align_model)
        except Exception as exc:
            # Alignment only refines word boundaries; if the language is
            # unsupported we fall back to the unaligned segments rather than
            # failing the job.
            log.warning(
                "[4/7] timestamp alignment skipped (%s); using unaligned segments", exc
            )

        # 5. diarization --------------------------------------------------- #
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set. The diarization model is gated; set HF_TOKEN in .env."
            )
        log.info("[5/7] running diarization")
        diarize_pipeline = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_pipeline(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        log.info("[5/7] releasing diarization pipeline")
        release_model(diarize_pipeline)

        # 6. assign speakers ----------------------------------------------- #
        log.info("[6/7] assigning speakers to segments")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 7. build the clean script payload -------------------------------- #
        log.info("[7/7] building script payload")
        script = build_script(result)
        log.info("[done] pipeline complete: %d segments", len(script))
        return script
    finally:
        shutil.rmtree(convert_dir, ignore_errors=True)


def build_script(result: dict) -> list[dict]:
    """Turn a WhisperX result into ``[{"time", "speaker", "text"}, ...]``."""
    segments = result["segments"] if isinstance(result, dict) else result.segments
    script: list[dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        text = (seg.get("text") or "").strip()
        script.append(
            {
                "time": {"start": round(start, 3), "end": round(end, 3)},
                "speaker": seg.get("speaker") or "UNKNOWN",
                "text": text,
            }
        )
    return script


# --------------------------------------------------------------------------- #
# Async worker
# --------------------------------------------------------------------------- #
async def transcribe_async(
    audio_path: str,
    *,
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """Run the blocking pipeline in a worker thread so the loop stays responsive."""
    return await asyncio.to_thread(
        run_pipeline,
        audio_path,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


# --------------------------------------------------------------------------- #
# FastAPI service
# --------------------------------------------------------------------------- #
app = FastAPI(title="Transcription Service", version="0.1.0")

# task_id -> {"status", "created_at", "result", "error"}  (in-memory)
_tasks: dict[str, dict] = {}


@app.post("/transcribe/async/")
async def create_transcription_task(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """Accept an audio upload and kick off an asynchronous transcription job."""
    task_id = uuid.uuid4().hex
    workdir = Path(tempfile.mkdtemp(prefix=f"tscript-{task_id}-"))

    # Persist the upload to a temp path the worker can read.
    in_path = workdir / (file.filename or "audio")
    with in_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    _tasks[task_id] = {
        "status": "pending",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    log.info("queued task %s (%s)", task_id, in_path.name)

    asyncio.create_task(
        _run_transcription_task(
            task_id, str(in_path), language, min_speakers, max_speakers, workdir
        )
    )
    return {"task_id": task_id, "status": "pending"}


async def _run_transcription_task(
    task_id: str,
    audio_path: str,
    language: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    workdir: Path,
) -> None:
    """Worker that drives a task through processing -> completed/failed."""
    task = _tasks[task_id]
    task["status"] = "processing"
    try:
        task["result"] = await transcribe_async(
            audio_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        task["status"] = "completed"
        log.info("task %s completed", task_id)
    except Exception as exc:
        task["error"] = str(exc)
        task["status"] = "failed"
        log.exception("task %s failed", task_id)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    """Poll a transcription task by id."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, **task}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cli_transcribe(args: argparse.Namespace) -> int:
    # Resolve input: a bare filename is looked up under DATA_DIR (./Data) first.
    in_path = Path(args.file)
    if not in_path.is_absolute() and not in_path.exists():
        under_data = DATA_DIR / args.file
        if under_data.is_file():
            in_path = under_data
    if not in_path.is_file():
        log.error("audio file not found: %s", args.file)
        return 1

    script = run_pipeline(
        str(in_path),
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
    )

    # --json dumps the raw payload to stdout and skips the formatted file.
    if args.json:
        print(json.dumps(script, ensure_ascii=False, indent=2))
        return 0

    # Otherwise write a formatted file. Default output dir is ./Output.
    fmt = args.format or DEFAULT_OUTPUT_FORMAT
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute() and out_path.parent == Path(""):
            out_path = OUTPUT_DIR / out_path  # bare filename -> ./Output/<name>
    else:
        out_path = OUTPUT_DIR / f"{in_path.stem}.{fmt}"

    resolved = save_transcript(script, out_path, fmt=fmt)
    log.info("wrote %d segments to %s (%s)", len(script), resolved, fmt)
    return 0


def cli_serve(args: argparse.Namespace) -> int:
    log.info("serving on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Single-file speech transcription service powered by WhisperX.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # transcribe <file>
    p_transcribe = sub.add_parser("transcribe", help="Transcribe a local audio file.")
    p_transcribe.add_argument("file", help="Audio path, or a filename under ./Data.")
    p_transcribe.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. 'en'). Auto-detects if omitted.",
    )
    p_transcribe.add_argument("--min-speakers", type=int, default=None)
    p_transcribe.add_argument("--max-speakers", type=int, default=None)
    p_transcribe.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="ASR batch size (smaller = less memory).",
    )
    p_transcribe.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default=None,
        help=f"Output file format (default: {DEFAULT_FORMAT}).",
    )
    p_transcribe.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path. A bare name writes to ./Output/<name>. "
        "Inferred from suffix unless --format is given.",
    )
    p_transcribe.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON script to stdout instead of writing a formatted file.",
    )
    p_transcribe.set_defaults(func=cli_transcribe)

    # serve
    p_serve = sub.add_parser("serve", help="Run the FastAPI HTTP API.")
    p_serve.add_argument("--host", default=HOST)
    p_serve.add_argument("--port", type=int, default=PORT)
    p_serve.set_defaults(func=cli_serve)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
