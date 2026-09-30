"""Single-file speech transcription service.

Wraps WhisperX into a sequential pipeline (transcribe -> align -> diarize -> script)
that loads and frees each model in turn to keep memory usage flat, and exposes it
through three front-ends:

  * ``run_pipeline``        -- the synchronous engine core
  * ``transcribe_async``    -- an async worker that runs the core off the event loop
  * FastAPI app             -- POST /transcribe/async/ + GET /tasks/{task_id}
  * CLI                     -- ``python transcribe.py transcribe <file>``

Run ``python transcribe.py serve`` to start the HTTP API.

Each stage of the pipeline is checkpointed (keyed by the input file's content
hash + params), so a failed run resumes from the last successful stage instead
of starting over. Per-stage and total wall-clock timings are logged.

All configuration is read from environment variables (see .env).
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env as defaults, but let an explicit shell/exported var win. This keeps
# config-overridable-from-the-environment (e.g. the benchmark can force
# DEVICE=cpu per run) while still picking up .env changes without a reinstall.
load_dotenv(override=False)

# pyannote hard-imports torchcodec and warns noisily when its native lib fails to
# load (the wheel can't resolve Homebrew's FFmpeg dylibs on macOS, and brew's
# ffmpeg 8 is newer than torchcodec 0.7's supported FFmpeg 4-7). Harmless here:
# WhisperX always hands diarization an in-memory waveform dict, so pyannote's
# torchcodec-based *file* decoding is never exercised. Silence just this warning.
warnings.filterwarnings(
    "ignore",
    message=r"\s*torchcodec is not installed correctly.*",
    category=UserWarning,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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
COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "int8")
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))


def _resolve_device(preferred: Optional[str]) -> str:
    """Map 'auto'/None to the best available accelerator; honor explicit names.

    Used for the pure-PyTorch stages (alignment + diarization), which DO benefit
    from MPS on Apple Silicon. ASR is separate (see ASR_DEVICE): faster-whisper
    (CTranslate2) has no Metal/MPS backend, so it is always CPU-bound.
    """
    if preferred and preferred.lower() != "auto":
        return preferred.lower()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# PyTorch stages (align + diarize): MPS when available. Default 'auto' picks the
# best accelerator; set DEVICE=cpu to force the old slow path for debugging.
PIPELINE_DEVICE: str = _resolve_device(os.getenv("DEVICE", "auto"))
# ASR stage: faster-whisper/CTranslate2 -- CPU only (no MPS backend exists).
# Override with ASR_DEVICE only if you have a faster-whisper build that supports
# another device; otherwise leave as 'cpu'.
ASR_DEVICE: str = (os.getenv("ASR_DEVICE") or "cpu").lower()
# faster-whisper CPU thread count. whisperx defaults to 4, which leaves half the
# cores idle on Apple Silicon (8 cores) -- a real ASR bottleneck. Default to all
# logical cores.
ASR_THREADS: int = int(os.getenv("ASR_THREADS", str(max(1, os.cpu_count() or 4))))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Default input/output locations (created on demand).
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "Data"))
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "Output"))
DEFAULT_OUTPUT_FORMAT: str = os.getenv("DEFAULT_OUTPUT_FORMAT", "md")

# Resume / checkpointing. Each expensive stage is cached under CHECKPOINT_DIR in
# a folder keyed by the input file's content hash + params; re-running the same
# file resumes from the last completed stage instead of starting over.
RESUME: bool = os.getenv("RESUME", "true").lower() in {"1", "true", "yes", "on"}
CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", ".transcribe_cache"))
DIARIZE_MODEL: Optional[str] = os.getenv("DIARIZE_MODEL") or None

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
def convert_to_mp3(
    input_path: str, out_dir: Optional[str] = None, out_name: Optional[str] = None
) -> str:
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
    out_name = out_name or f"{src.stem}.mp3"
    out_path = work_dir / out_name

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
# Resume / checkpointing
# --------------------------------------------------------------------------- #
# The pipeline has seven stages and any of them can fail (OOM, a gated-model
# hiccup, ...). To get resume safety we persist each stage's output to disk in
# a per-job folder keyed by a digest of the input file's *content* + the params
# that affect the output. Re-running the same file picks up at the last stage
# that completed, so e.g. a diarization failure never forces a re-transcription.
#
# Job folder layout (<CHECKPOINT_DIR>/<digest>/):
#   meta.json         params, content hash, status, timings
#   converted.mp3     [1] normalized audio
#   asr.json          [2] raw ASR result (the expensive one)
#   aligned.json      [4] word-aligned result
#   diarize.json      [5] speaker turns as [{"start","end","speaker"}, ...]
#   final.json        [6] result after speaker assignment
#   script.json       [7] the final clean script
_CKPT_AUDIO = "converted.mp3"
_CKPT_ASR = "asr.json"
_CKPT_ALIGNED = "aligned.json"
_CKPT_DIARIZE = "diarize.json"
_CKPT_FINAL = "final.json"
_CKPT_SCRIPT = "script.json"


def _sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_params(
    language: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> dict:
    """Params that change a stage's output -> part of the cache key.

    ``batch_size`` is intentionally excluded: it only affects speed, not the
    transcription output, so changing it should NOT invalidate the cache.
    """
    return {
        "model": MODEL_NAME,
        "asr_device": ASR_DEVICE,
        "pipeline_device": PIPELINE_DEVICE,
        "compute_type": COMPUTE_TYPE,
        "threads": ASR_THREADS,
        "diarize_model": DIARIZE_MODEL or "default",
        "language": language,  # None == auto-detect
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }


def _jsonable(obj):
    """Coerce numpy/pandas scalars and arrays into JSON-serializable Python.

    WhisperX/faster-whisper sprinkle ``numpy.float32`` etc. through their result
    dicts; ``json.dumps`` can't serialize those directly.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@contextmanager
def _timed(store: dict, key: str):
    """Record wall-clock seconds spent inside the block into ``store[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - start, 3)


class Checkpoint:
    """Per-job artifact store, keyed by input content hash + params digest."""

    def __init__(self, root: Path, digest: str, meta: dict):
        self.root = root
        self.digest = digest
        self.dir = root / digest
        self.meta = meta

    @classmethod
    def open(cls, audio_path: str | Path, params: dict) -> "Checkpoint":
        root = CHECKPOINT_DIR
        root.mkdir(parents=True, exist_ok=True)
        content_hash = _sha256_file(audio_path)
        digest = hashlib.sha256(
            json.dumps(
                {"content": content_hash, "params": params}, sort_keys=True
            ).encode()
        ).hexdigest()[:24]
        job_dir = root / digest
        job_dir.mkdir(parents=True, exist_ok=True)

        meta_path = job_dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # corrupt meta -> start fresh
                meta = {}
        meta.update(
            {
                "digest": digest,
                "content_sha256": content_hash,
                "input_name": Path(audio_path).name,
                "params": params,
            }
        )
        ckpt = cls(root, digest, meta)
        ckpt._write_meta()
        return ckpt

    def _write_meta(self) -> None:
        now = time.time()
        self.meta.setdefault("created_at", now)
        self.meta["updated_at"] = now
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def path(self, name: str) -> Path:
        return self.dir / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def save_json(self, name: str, obj) -> None:
        self.path(name).write_text(
            json.dumps(_jsonable(obj), ensure_ascii=False), encoding="utf-8"
        )

    def load_json(self, name: str):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def record(self, **fields) -> None:
        self.meta.update(fields)
        self._write_meta()


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
    resume: bool = RESUME,
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

    When ``resume`` is true (default; toggle with ``RESUME``) every stage's
    output is checkpointed under ``CHECKPOINT_DIR`` keyed by the input's content
    hash + params, so a re-run on the same file resumes from the last completed
    stage instead of starting over. Per-stage and total timings are logged.
    """
    timings: dict[str, float] = {}
    total_t0 = time.perf_counter()

    ckpt: Optional[Checkpoint] = None
    if resume:
        ckpt = Checkpoint.open(
            audio_path, _cache_params(language, min_speakers, max_speakers)
        )

    # Fast path: a finished script for this exact input+params already exists.
    if ckpt and ckpt.has(_CKPT_SCRIPT):
        log.info(
            "[resume] cache hit for %s -- returning completed script",
            Path(audio_path).name,
        )
        return ckpt.load_json(_CKPT_SCRIPT)

    # The normalized mp3 lives in the checkpoint dir (reusable) or a temp dir.
    convert_dir: Optional[Path] = None
    convert_out = (
        ckpt.dir if ckpt else Path(tempfile.mkdtemp(prefix="tscript-convert-"))
    )
    if not ckpt:
        convert_dir = convert_out

    audio = None  # numpy array; loaded by ASR, or on a cache hit before align
    result: Optional[dict] = None

    try:
        # 1. normalize input to mp3 --------------------------------------- #
        log.info("[1/7] converting input to mp3")
        with _timed(timings, "convert"):
            if ckpt and ckpt.has(_CKPT_AUDIO):
                log.info("[1/7] cache hit: normalized audio")
                converted_mp3 = ckpt.path(_CKPT_AUDIO)
            else:
                converted_mp3 = Path(
                    convert_to_mp3(
                        audio_path,
                        out_dir=str(convert_out),
                        out_name=_CKPT_AUDIO if ckpt else None,
                    )
                )
        audio_path = str(converted_mp3)

        # 2. ASR ----------------------------------------------------------- #
        if ckpt and ckpt.has(_CKPT_ASR):
            log.info("[2/7] cache hit: ASR result")
            with _timed(timings, "asr"):
                result = ckpt.load_json(_CKPT_ASR)
        else:
            with _timed(timings, "asr"):
                log.info(
                    "[2/7] loading ASR model %s on %s (compute_type=%s, threads=%d)",
                    MODEL_NAME,
                    ASR_DEVICE,
                    COMPUTE_TYPE,
                    ASR_THREADS,
                )
                model = whisperx.load_model(
                    MODEL_NAME,
                    ASR_DEVICE,
                    compute_type=COMPUTE_TYPE,
                    threads=ASR_THREADS,
                    use_auth_token=HF_TOKEN,
                )
                log.info("[2/7] loading audio: %s", audio_path)
                audio = whisperx.load_audio(audio_path)
                log.info("[2/7] transcribing (batch_size=%d)", batch_size)
                result = model.transcribe(
                    audio, batch_size=batch_size, language=language
                )
                # 3. free ASR model before touching alignment ----------------- #
                log.info("[3/7] releasing ASR model")
                release_model(model)
            if ckpt:
                ckpt.save_json(_CKPT_ASR, result)

        detected_language = result.get("language", language)

        # align + diarize both need the audio array; load it if ASR was cached.
        if audio is None:
            audio = whisperx.load_audio(audio_path)

        # 4. alignment ----------------------------------------------------- #
        if ckpt and ckpt.has(_CKPT_ALIGNED):
            log.info("[4/7] cache hit: aligned result")
            with _timed(timings, "align"):
                result = ckpt.load_json(_CKPT_ALIGNED)
        else:
            with _timed(timings, "align"):
                try:
                    log.info(
                        "[4/7] aligning timestamps (language=%s, device=%s)",
                        detected_language,
                        PIPELINE_DEVICE,
                    )
                    align_model, align_meta = whisperx.load_align_model(
                        detected_language, PIPELINE_DEVICE
                    )
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_meta,
                        audio,
                        PIPELINE_DEVICE,
                        return_char_alignments=False,
                    )
                    log.info("[4/7] releasing alignment model")
                    release_model(align_model)
                except Exception as exc:
                    # Alignment only refines word boundaries; if the language is
                    # unsupported we fall back to the unaligned segments.
                    log.warning(
                        "[4/7] timestamp alignment skipped (%s); using unaligned segments",
                        exc,
                    )
            if ckpt:
                ckpt.save_json(_CKPT_ALIGNED, result)

        # 5. diarization --------------------------------------------------- #
        if ckpt and ckpt.has(_CKPT_DIARIZE):
            log.info("[5/7] cache hit: diarization")
            with _timed(timings, "diarize"):
                rows = ckpt.load_json(_CKPT_DIARIZE)
                diarize_segments = pd.DataFrame(
                    rows, columns=["start", "end", "speaker"]
                )
        else:
            with _timed(timings, "diarize"):
                if not HF_TOKEN:
                    raise RuntimeError(
                        "HF_TOKEN is not set. The diarization model is gated; "
                        "set HF_TOKEN in .env."
                    )
                log.info("[5/7] running diarization (device=%s)", PIPELINE_DEVICE)
                diarize_kwargs: dict = {"token": HF_TOKEN, "device": PIPELINE_DEVICE}
                if DIARIZE_MODEL:
                    diarize_kwargs["model_name"] = DIARIZE_MODEL
                diarize_pipeline = DiarizationPipeline(**diarize_kwargs)
                diarize_segments = diarize_pipeline(
                    audio, min_speakers=min_speakers, max_speakers=max_speakers
                )
                log.info("[5/7] releasing diarization pipeline")
                release_model(diarize_pipeline)
            if ckpt:
                rows = [
                    {
                        "start": float(r["start"]),
                        "end": float(r["end"]),
                        "speaker": str(r["speaker"]),
                    }
                    for _, r in diarize_segments[["start", "end", "speaker"]].iterrows()
                ]
                ckpt.save_json(_CKPT_DIARIZE, rows)

        # 6. assign speakers ----------------------------------------------- #
        log.info("[6/7] assigning speakers to segments")
        with _timed(timings, "assign"):
            result = whisperx.assign_word_speakers(diarize_segments, result)
        if ckpt:
            ckpt.save_json(_CKPT_FINAL, result)

        # 7. build the clean script payload -------------------------------- #
        log.info("[7/7] building script payload")
        with _timed(timings, "build"):
            script = build_script(result)
        if ckpt:
            ckpt.save_json(_CKPT_SCRIPT, script)
            ckpt.record(status="completed", timings=timings)

        log.info(
            "[done] pipeline complete: %d segments in %.1fs "
            "(convert %.1f, asr %.1f, align %.1f, diarize %.1f, assign %.1f, build %.1f)",
            len(script),
            time.perf_counter() - total_t0,
            timings.get("convert", 0.0),
            timings.get("asr", 0.0),
            timings.get("align", 0.0),
            timings.get("diarize", 0.0),
            timings.get("assign", 0.0),
            timings.get("build", 0.0),
        )
        return script
    finally:
        if convert_dir:
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
    resume: bool = RESUME,
) -> list[dict]:
    """Run the blocking pipeline in a worker thread so the loop stays responsive."""
    return await asyncio.to_thread(
        run_pipeline,
        audio_path,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        resume=resume,
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
    resume: Optional[bool] = None,
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
    use_resume = RESUME if resume is None else resume
    log.info("queued task %s (%s, resume=%s)", task_id, in_path.name, use_resume)

    asyncio.create_task(
        _run_transcription_task(
            task_id,
            str(in_path),
            language,
            min_speakers,
            max_speakers,
            workdir,
            use_resume,
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
    resume: bool,
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
            resume=resume,
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
        resume=RESUME and not args.no_resume,
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
        "--no-resume",
        action="store_true",
        help="Ignore saved checkpoints and run the full pipeline from scratch.",
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
