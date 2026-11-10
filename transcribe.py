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
import fcntl
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
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
# ASR stage: two interchangeable backends.
#   'mlx' -> MLX whisper on the Apple GPU (Metal, fp16) -- several times faster
#            than CPU int8 on Apple Silicon, same model weights.
#   'ct2' -> faster-whisper/CTranslate2 (ASR_DEVICE, COMPUTE_TYPE, ASR_THREADS).
# 'auto' prefers MLX when the package is importable and the configured model has
# a known MLX conversion; otherwise it falls back to ct2.
ASR_BACKEND: str = (os.getenv("ASR_BACKEND") or "auto").lower()
ASR_DEVICE: str = (os.getenv("ASR_DEVICE") or "cpu").lower()
# faster-whisper CPU thread count (ct2 backend only). whisperx defaults to 4,
# which leaves cores idle; default to all logical cores.
ASR_THREADS: int = int(os.getenv("ASR_THREADS", str(max(1, os.cpu_count() or 4))))
# Optional explicit MLX repo id (default: mapped from TRANSCRIPTION_MODEL below).
MLX_MODEL: Optional[str] = os.getenv("MLX_MODEL") or None

# Known MLX (mlx-community) conversions of the whisperx model names. An
# unmapped TRANSCRIPTION_MODEL with no MLX_MODEL override makes 'auto' fall
# back to ct2 rather than guess at a conversion.
_MLX_MODEL_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny",
}


def _mlx_model_repo() -> Optional[str]:
    """MLX repo id for the configured model, or None if there is no known one."""
    if MLX_MODEL:
        return MLX_MODEL
    return _MLX_MODEL_REPOS.get(MODEL_NAME.lower())


def _effective_asr_backend() -> str:
    """Resolve ASR_BACKEND ('auto' -> 'mlx' when usable, else 'ct2')."""
    if ASR_BACKEND in {"mlx", "ct2"}:
        return ASR_BACKEND
    try:
        import mlx_whisper  # noqa: F401

        return "mlx" if _mlx_model_repo() else "ct2"
    except ImportError:
        return "ct2"


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
# Segmentation window hop in seconds (default 2.0). The hop controls how many
# overlapping 10s windows (and therefore per-speaker embedding crops) the
# diarizer processes: the default upstream 1.0s hop costs 2x the wall-clock
# for ~1.6% frame-level speaker changes on our eval audio. 1.0 restores
# upstream behavior exactly; see OPTIMIZATION_LOG.md.
DIARIZE_SEG_STEP: float = float(os.getenv("DIARIZE_SEG_STEP", "2.0"))
# Diarization strategy ladder (see _diarize_mode). 'pyannote' = always the full
# pipeline. 'fast' (default) = embed VAD windows with the diarizer's own
# embedding model, cluster them, and verify the file's voice centroids against
# the speaker roster; escalate to full pyannote only when that is impossible
# (unknown voice, too many voices, empty roster, clustering failure). 'auto'
# additionally escalates on weak quality signals (low per-window margin).
DIARIZE_MODE: str = (os.getenv("DIARIZE_MODE") or "fast").lower()
# Fast path: escalate to full pyannote when clustering finds more voices than
# this (the K probe validated 2-3 voice files; clustering gets shakier as
# voices grow).
DIARIZE_FAST_MAX_VOICES: int = int(os.getenv("DIARIZE_FAST_MAX_VOICES", "4"))
# Fast path: greedy clustering cosine threshold (probe measured intra-cluster
# ~0.9 vs inter-cluster <=0.45 -- 0.5 sits in the canyon).
DIARIZE_FAST_CLUSTER_THRESHOLD: float = float(
    os.getenv("DIARIZE_FAST_CLUSTER_THRESHOLD", "0.50")
)
# Fast path: centroid-level merge threshold. One voice can split into several
# clusters when its acoustic mode shifts mid-file (mic change, channel drift);
# such fragments measured >=0.71 centroid cosine to their main cluster while
# different voices stayed <=0.51 on the same file. 0.65 sits in that canyon,
# biased toward NOT merging distinct voices (a missed merge only escalates to
# full pyannote; a wrong merge silently loses a speaker).
DIARIZE_FAST_MERGE_THRESHOLD: float = float(
    os.getenv("DIARIZE_FAST_MERGE_THRESHOLD", "0.65")
)
# Fast path: per-window margin (top1-top2 cosine) below which a window counts
# as low-margin ('auto'-mode escalation stat; measured 4-8% of windows on
# all-known files vs 40% on an unknown-voice file). 0 disables the frame-level
# refinement substrate entirely (coarse window labels only -- debug/AB knob).
DIARIZE_FAST_REFINE: float = float(os.getenv("DIARIZE_FAST_REFINE", "0.10"))
# 'auto' mode: escalate when more than this fraction of windows is low-margin.
DIARIZE_AUTO_MAX_LOWMARGIN: float = float(
    os.getenv("DIARIZE_AUTO_MAX_LOWMARGIN", "0.15")
)
# Fast path safety valve: escalate when the refined frame-level trace
# disagrees with the coarse window labels on more than this speech fraction
# (the fast path's own two views conflicting = untrustworthy).
DIARIZE_FAST_MAX_FLIP: float = float(os.getenv("DIARIZE_FAST_MAX_FLIP", "0.20"))
if DIARIZE_MODE not in {"fast", "auto", "pyannote"}:
    raise ValueError(f"DIARIZE_MODE must be fast|auto|pyannote, got {DIARIZE_MODE!r}")
if DIARIZE_FAST_MAX_VOICES < 1:
    raise ValueError("DIARIZE_FAST_MAX_VOICES must be >= 1")
for _name in (
    "DIARIZE_FAST_CLUSTER_THRESHOLD",
    "DIARIZE_FAST_MERGE_THRESHOLD",
    "DIARIZE_AUTO_MAX_LOWMARGIN",
    "DIARIZE_FAST_MAX_FLIP",
):
    if not 0.0 < globals()[_name] <= 1.0:
        raise ValueError(f"{_name} must be in (0, 1]")
if not 0.0 <= DIARIZE_FAST_REFINE <= 1.0:
    raise ValueError("DIARIZE_FAST_REFINE must be in [0, 1] (0 disables refinement)")
# Run diarization on a worker thread alongside ASR+alignment when both halves
# need computing: diarization depends only on the audio, so the branches are
# independent. 'auto' enables it with the mlx backend (MLX and MPS share the
# one GPU but interleave well); 'true'/'false' force it on/off.
PARALLEL_STAGES: str = (os.getenv("PARALLEL_STAGES") or "auto").lower()

# Speaker memory: link diarized voices to stable SPEAKER_XX tags across files.
# Diarization already computes per-speaker centroid embeddings internally (for
# its own clustering); with linking on we also ask for them (no extra
# inference) and match them against a small on-disk store, so the same voice
# keeps the same tag in every file (new voices get the next unused tag).
# 'false' restores per-file labels and skips the store entirely.
SPEAKER_LINKING: bool = (os.getenv("SPEAKER_LINKING") or "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# The store (tags + centroids). Gitignored; deleting it resets the namespace.
SPEAKER_STORE: Path = Path(os.getenv("SPEAKER_STORE", ".speakers.json"))
# Cosine threshold above which a file speaker counts as a stored voice.
# Calibrated on two recordings with the same speaker pair (see
# OPTIMIZATION_LOG.md) and biased high: a missed match only costs one extra
# tag, a false merge corrupts two identities. Same-voice cross-file cosines
# measured 0.775-0.833; different-voice pairs 0.382-0.460.
SPEAKER_MATCH_THRESHOLD: float = float(os.getenv("SPEAKER_LINK_THRESHOLD", "0.70"))

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
#   diarize.json      [5] {"turns": [{"start","end","speaker"}, ...],
#                     "embeddings": {tag: centroid},
#                     "diarization": {strategy, mode, reasons, ...}} -- the
#                     embeddings/diarization halves are present only when
#                     speaker linking is on / the strategy ladder ran
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
    diarize_mode: Optional[str] = None,
) -> dict:
    """Params that change a stage's output -> part of the cache key.

    ``batch_size`` is intentionally excluded: it only affects speed, not the
    transcription output, so changing it should NOT invalidate the cache.
    """
    return {
        "model": MODEL_NAME,
        # Explicit MLX repo overrides change the ASR output (e.g. a quantized
        # conversion) but not MODEL_NAME -- scope the cache key to the actual
        # repo so swapped models never serve each other's cached transcripts.
        "mlx_model": MLX_MODEL,
        "asr_backend": _effective_asr_backend(),
        "asr_device": ASR_DEVICE,
        "pipeline_device": PIPELINE_DEVICE,
        "compute_type": COMPUTE_TYPE,
        "threads": ASR_THREADS,
        "diarize_model": DIARIZE_MODEL or "default",
        # Changes diarize output (turn boundaries) -> must scope the cache.
        "diarize_seg_step": DIARIZE_SEG_STEP,
        # Same story for the strategy ladder: fast and pyannote produce
        # different turn boundaries, and the fast knobs (voice cap, cluster
        # threshold, refinement) reshape its output. One-time invalidation,
        # like asr_backend. Escalation-only knobs (auto_max_lowmargin,
        # max_flip) are excluded: they never change a *successful* fast
        # output, only whether fast is attempted/kept.
        "diarize_mode": _diarize_mode(diarize_mode),
        "diarize_fast_max_voices": DIARIZE_FAST_MAX_VOICES,
        "diarize_fast_cluster_threshold": DIARIZE_FAST_CLUSTER_THRESHOLD,
        "diarize_fast_merge_threshold": DIARIZE_FAST_MERGE_THRESHOLD,
        "diarize_fast_refine": DIARIZE_FAST_REFINE,
        "language": language,  # None == auto-detect
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "speaker_linking": SPEAKER_LINKING,
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
# ASR backends
# --------------------------------------------------------------------------- #
def _transcribe_asr_mlx(audio: "np.ndarray", language: Optional[str]) -> dict:
    """Transcribe with MLX whisper on the Apple GPU (Metal, fp16), VAD-chunked.

    Same model weights as the ct2 path (mlx-community conversions of the
    OpenAI checkpoints) but the encoder/decoder matmuls run on the GPU, which
    is several times faster than CTranslate2 int8 on this class of CPU.

    The audio is first split into speech chunks with whisperx's own Pyannote
    VAD -- the exact chunker the ct2 backend uses -- and each chunk is
    transcribed independently. This is not optional: whisper's fixed 30s
    windowing over unchunked audio skips over speech (seek jumps) and
    repetition-loops on long context (a known turbo artifact), which the VAD
    chunking avoids. A bonus: each chunk starts with fresh context, so the
    greedy decode cannot inherit a loop from previous windows.

    Word timestamps come from the alignment stage as before, so only
    start/end/text are kept per segment.
    """
    import mlx_whisper
    from whisperx.audio import SAMPLE_RATE
    from whisperx.vads import Pyannote

    repo = _mlx_model_repo()
    if not repo:
        raise RuntimeError(
            f"no MLX conversion known for TRANSCRIPTION_MODEL={MODEL_NAME!r}; "
            "set MLX_MODEL or ASR_BACKEND=ct2."
        )
    log.info(
        "[2/7] transcribing with MLX whisper (%s, fp16, VAD-chunked, language=%s)",
        repo,
        language or "auto-detect",
    )

    vad_t0 = time.perf_counter()
    vad = Pyannote(
        torch.device(PIPELINE_DEVICE),
        token=None,
        chunk_size=30,
        vad_onset=0.5,
        vad_offset=0.363,
    )
    vad_out = vad(
        {"waveform": Pyannote.preprocess_audio(audio), "sample_rate": SAMPLE_RATE}
    )
    chunks = Pyannote.merge_chunks(vad_out, chunk_size=30)
    log.info(
        "[2/7] VAD split audio into %d speech chunks (%.1fs)",
        len(chunks),
        time.perf_counter() - vad_t0,
    )
    if not chunks:
        return {
            "segments": [],
            "word_timestamps": [],
            "language": language or "en",
        }

    segments: list[dict] = []
    detected: Optional[str] = language
    for ch in chunks:
        s0, s1 = ch["start"], ch["end"]
        piece = audio[int(s0 * SAMPLE_RATE) : int(s1 * SAMPLE_RATE)]
        r = mlx_whisper.transcribe(piece, path_or_hf_repo=repo, language=detected)
        if detected is None and r.get("language"):
            detected = r["language"]  # lock in detection from the first chunk
        for seg in r.get("segments", []):
            segments.append(
                {
                    "start": s0 + float(seg["start"]),
                    "end": s0 + float(seg["end"]),
                    "text": seg["text"],
                }
            )
    return {
        "segments": segments,
        "word_timestamps": [],
        "language": detected or "en",
    }


# --------------------------------------------------------------------------- #
# Diarization (shared by the sequential and parallel paths)
# --------------------------------------------------------------------------- #
def _parallel_diarize_enabled() -> bool:
    """Resolve PARALLEL_STAGES ('auto' -> on with the mlx backend)."""
    if PARALLEL_STAGES in {"1", "true", "yes", "on"}:
        return True
    if PARALLEL_STAGES in {"0", "false", "no", "off"}:
        return False
    return _effective_asr_backend() == "mlx"


def _diarize_mode(override: Optional[str] = None) -> str:
    """Resolve the diarization strategy: 'pyannote' | 'fast' | 'auto'.

    ``override`` comes from the CLI/API (``--diarize-mode`` / ``diarize_mode``)
    and beats DIARIZE_MODE from the environment, which defaults to 'fast'.
    """
    mode = (override or DIARIZE_MODE).lower()
    if mode not in {"pyannote", "fast", "auto"}:
        raise ValueError(f"diarize mode must be fast|auto|pyannote, got {mode!r}")
    return mode


# Single worker so concurrent API jobs never stack diarizations (one GPU).
_DIARIZE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")


def _diarize_rows(diarize_segments: "pd.DataFrame") -> list[dict]:
    """Checkpointable rows from a diarization result frame."""
    return [
        {
            "start": float(r["start"]),
            "end": float(r["end"]),
            "speaker": str(r["speaker"]),
        }
        for _, r in diarize_segments[["start", "end", "speaker"]].iterrows()
    ]


def _run_diarization(
    audio: "np.ndarray",
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    diarize_mode: Optional[str] = None,
) -> tuple["pd.DataFrame", dict, float, dict]:
    """Run the diarization strategy ladder, then free the pipeline.

    'pyannote' goes straight to the full pipeline. 'fast' (default) and
    'auto' first try the fast path (VAD-window embeddings verified against
    the speaker roster -- see the fast-path section above) and escalate to
    the full pipeline on the same loaded object when the fast path refuses
    its own result; 'auto' additionally escalates on weak quality signals
    (a large fraction of low-margin windows). Both rungs' triggers land in
    ``info["reasons"]``.

    Returns ``(segments frame, centroids, elapsed seconds, info)``. The full
    pipeline returns its per-speaker centroids alongside the turns (computed
    for its clustering anyway); the fast path returns its verified per-tag
    centroids. ``info`` records the ladder decision (strategy, mode,
    reasons, stats) and is checkpointed next to the turns. Runs on the main
    thread or on the diarize worker; it must not touch the Checkpoint
    (meta.json races with the main thread) -- the caller persists the result
    after joining.
    """
    t0 = time.perf_counter()
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. The diarization model is gated; set HF_TOKEN in .env."
        )
    mode = _diarize_mode(diarize_mode)
    log.info("[5/7] running diarization (mode=%s, device=%s)", mode, PIPELINE_DEVICE)
    diarize_kwargs: dict = {"token": HF_TOKEN, "device": PIPELINE_DEVICE}
    if DIARIZE_MODEL:
        diarize_kwargs["model_name"] = DIARIZE_MODEL
    diarize_pipeline = DiarizationPipeline(**diarize_kwargs)
    if DIARIZE_SEG_STEP > 0 and DIARIZE_SEG_STEP != 1.0:
        # The embedding stage (94% of diarize wall-clock) runs one 10s crop per
        # local speaker per segmentation window; the default 1.0s hop (90%
        # window overlap) makes it process ~20x the audio duration. A 2.0s hop
        # halves that for a 2x stage speedup with 98.4% frame-level agreement
        # on our eval audio (see OPTIMIZATION_LOG.md). The Inference bakes its
        # step at pipeline init, so patch the built object directly. The fast
        # path's refinement substrate inherits the same patched hop.
        diarize_pipeline.model._segmentation.step = DIARIZE_SEG_STEP

    def _full(reasons: list[str]) -> tuple["pd.DataFrame", dict, dict]:
        info = {"strategy": "pyannote", "mode": mode, "reasons": reasons}
        if SPEAKER_LINKING:
            segments, centroids = diarize_pipeline(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        else:
            segments = diarize_pipeline(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            centroids = {}
        return segments, (centroids or {}), info

    try:
        if mode == "pyannote":
            segments, centroids, info = _full(["mode=pyannote"])
        elif not SPEAKER_LINKING:
            # The fast path is roster verification; with speaker memory off
            # there is nothing to verify against -- always full pipeline.
            segments, centroids, info = _full(["speaker linking off"])
        else:
            try:
                rows, centroids, info = _fast_diarize(
                    diarize_pipeline, audio, min_speakers, max_speakers, mode
                )
            except Exception as exc:  # safety net: never lose a run to the fast path
                log.warning(
                    "[5/7] fast diarization failed (%s: %s); escalating to full pyannote",
                    type(exc).__name__,
                    exc,
                )
                segments, centroids, info = _full([f"fast path exception: {exc}"])
            else:
                escalate: Optional[list[str]] = None
                if rows is None:
                    escalate = info["reasons"]
                elif (
                    mode == "auto"
                    and info.get("low_margin_frac", 0.0) > DIARIZE_AUTO_MAX_LOWMARGIN
                ):
                    escalate = info["reasons"] + [
                        f"{info['low_margin_frac']:.0%} of windows below margin "
                        f"{DIARIZE_FAST_REFINE:.2f} > "
                        f"DIARIZE_AUTO_MAX_LOWMARGIN={DIARIZE_AUTO_MAX_LOWMARGIN} (auto)"
                    ]
                if escalate is not None:
                    log.warning(
                        "[5/7] fast diarization escalating to full pyannote (%s)",
                        "; ".join(escalate),
                    )
                    segments, centroids, info = _full(escalate)
                else:
                    return (
                        pd.DataFrame(rows, columns=["start", "end", "speaker"]),
                        centroids,
                        time.perf_counter() - t0,
                        info,
                    )
        return segments, centroids, time.perf_counter() - t0, info
    finally:
        log.info("[5/7] releasing diarization pipeline")
        release_model(diarize_pipeline)


# --------------------------------------------------------------------------- #
# Speaker memory (stable tags across files)
# --------------------------------------------------------------------------- #
# Diarization labels each file in isolation (SPEAKER_00, SPEAKER_01, ...), so
# the same person gets a different tag in every recording. With SPEAKER_LINKING
# on, one centroid per voice is kept in a small on-disk store and every new
# file is relabeled against it: a known voice keeps its tag across all files,
# a new voice takes the next unused tag (in first-appearance order). Tags are
# deliberately anonymous -- stable identity, never names.
_SPEAKER_STORE_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unit_centroid(centroid: list[float]) -> Optional["np.ndarray"]:
    """L2-normalized centroid as float64, or None if it is degenerate.

    pyannote zero-pads centroids when its clustering finds fewer clusters than
    labels; a zero vector carries no identity and must not be matched or
    stored.
    """
    vec = np.asarray(centroid, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return None
    return vec / norm


def _read_store() -> dict:
    """Load the speaker store; a missing or corrupt store starts fresh.

    A corrupt store is parked as ``.speakers.json.corrupt-<epoch>`` rather than
    silently renumbering the tag namespace.
    """
    empty = {
        "version": _SPEAKER_STORE_VERSION,
        "next_tag": 0,
        "speakers": [],
    }
    if not SPEAKER_STORE.exists():
        return empty
    try:
        data = json.loads(SPEAKER_STORE.read_text(encoding="utf-8"))
        if data.get("version") != _SPEAKER_STORE_VERSION:
            raise ValueError(f"unknown store version {data.get('version')!r}")
        return data
    except Exception as exc:
        backup = SPEAKER_STORE.with_name(
            f"{SPEAKER_STORE.name}.corrupt-{int(time.time())}"
        )
        try:
            os.replace(SPEAKER_STORE, backup)
            parked = f"parked at {backup.name}"
        except OSError:
            parked = "could not park it"
        log.warning("speaker store unreadable (%s); %s, starting fresh", exc, parked)
        return empty


def _write_store(store: dict) -> None:
    """Atomically write the speaker store (tmp file + rename)."""
    tmp = SPEAKER_STORE.with_name(f"{SPEAKER_STORE.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, SPEAKER_STORE)


def _update_entry(entry: dict, unit: "np.ndarray", source_name: str) -> None:
    """Fold one unit centroid into an existing store entry (running mean).

    The mean is weighted by how many enrollments each side already carries,
    then re-normalized so the stored centroid stays unit-length. Shared by the
    linking path (new evidence for a matched or freshly allocated tag) and the
    fast-path refresh (evidence for a tag verified against the roster).
    """
    n = int(entry.get("n_embeddings", 1))
    prev = _unit_centroid(entry["centroid"])
    mean = unit + n * prev if prev is not None else unit * (n + 1)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-9:
        mean = mean / norm
    entry["centroid"] = mean.tolist()
    entry["n_embeddings"] = n + 1
    if source_name not in entry.get("files", []):
        entry.setdefault("files", []).append(source_name)
    entry["last_seen"] = _now_iso()


def _link_speakers(
    rows: list[dict], centroids: dict[str, list[float]], source_name: str
) -> tuple[list[dict], dict[str, list[float]]]:
    """Relabel per-file speaker tags to stable cross-file tags.

    Matches each file speaker's centroid against the stored voices by cosine
    similarity (best match per file speaker above ``SPEAKER_MATCH_THRESHOLD``;
    two local labels may legitimately map to the same stored voice when
    clustering over-split one person). New voices allocate the next unused tag.
    The store is read and written under an exclusive lock on a sidecar lock
    file, so concurrent jobs (CLI + server, or two API uploads) serialize
    cleanly; the data file itself is always replaced atomically.

    Returns ``(relabeled rows, centroids keyed by the global tag)`` and logs
    every candidate similarity -- which doubles as the calibration instrument
    (see OPTIMIZATION_LOG.md).
    """
    if not rows:
        return rows, {}

    # Local labels in order of first appearance (rows are time-ordered).
    order: list[str] = []
    for r in rows:
        if r["speaker"] not in order:
            order.append(r["speaker"])

    local_unit = {tag: _unit_centroid(centroids.get(tag, [])) for tag in order}
    for tag, unit in local_unit.items():
        if unit is None:
            log.info(
                "[5/7] speaker linking: %s has a degenerate centroid; "
                "it gets a tag but is not enrolled",
                tag,
            )

    lock_path = SPEAKER_STORE.with_name(f"{SPEAKER_STORE.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        store = _read_store()
        stored: dict[str, dict] = {s["tag"]: s for s in store["speakers"]}
        stored_unit = {tag: _unit_centroid(s["centroid"]) for tag, s in stored.items()}

        # Best stored match per local label. All candidate scores are logged so
        # the threshold can be calibrated from real runs.
        candidates: list[tuple[float, str, str]] = []
        for ltag in order:
            lu = local_unit[ltag]
            if lu is None:
                continue
            for gtag, gu in stored_unit.items():
                if gu is None:
                    continue
                candidates.append((float(np.dot(lu, gu)), ltag, gtag))
        if candidates:
            log.info(
                "[5/7] speaker linking candidates: %s",
                ", ".join(
                    f"{local}->{glob} {score:.3f}"
                    for score, local, glob in sorted(candidates, reverse=True)
                ),
            )
        matched: dict[str, tuple[float, str]] = {}
        for score, ltag, gtag in sorted(candidates, reverse=True):
            if score < SPEAKER_MATCH_THRESHOLD:
                break
            if ltag not in matched:  # keep each local label's best match only
                matched[ltag] = (score, gtag)

        label_map: dict[str, str] = {}
        embeddings_out: dict[str, list[float]] = {}
        for ltag in order:
            if ltag in matched:
                score, gtag = matched[ltag]
                label_map[ltag] = gtag
                log.info(
                    "[5/7] speaker linking: %s -> %s (cos %.3f, %d file(s))",
                    ltag,
                    gtag,
                    score,
                    len(stored[gtag].get("files", [])),
                )
            else:
                gtag = f"SPEAKER_{store['next_tag']:02d}"
                while gtag in stored:  # never reuse a live tag
                    store["next_tag"] += 1
                    gtag = f"SPEAKER_{store['next_tag']:02d}"
                store["next_tag"] += 1
                label_map[ltag] = gtag
                log.info("[5/7] speaker linking: %s -> %s (new voice)", ltag, gtag)

            unit = local_unit[ltag]
            if unit is None:
                continue  # tagged but not enrolled
            entry = stored.get(gtag)
            if entry is None:
                stored[gtag] = {
                    "tag": gtag,
                    "centroid": unit.tolist(),
                    "n_embeddings": 1,
                    "files": [source_name],
                    "last_seen": _now_iso(),
                }
            else:
                _update_entry(entry, unit, source_name)
            embeddings_out[gtag] = stored[gtag]["centroid"]

        store["speakers"] = list(stored.values())
        _write_store(store)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    relabeled = [{**r, "speaker": label_map[r["speaker"]]} for r in rows]
    return relabeled, embeddings_out


def _refresh_enrolled(
    embeddings: dict[str, list[float]], source_name: str
) -> dict[str, list[float]]:
    """Fold a fast-path run's centroids into already-enrolled store entries.

    The fast path only keeps its result when every voice verified against the
    roster, so its rows already carry global tags -- nothing to relabel and
    nothing to allocate. This just refreshes those tags' running means (the
    run is fresh evidence about each voice) under the same exclusive lock
    ``_link_speakers`` uses. A tag that vanished from the store mid-run is
    skipped with a warning rather than resurrected: never allocate here.
    """
    if not embeddings:
        return {}
    lock_path = SPEAKER_STORE.with_name(f"{SPEAKER_STORE.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        store = _read_store()
        stored: dict[str, dict] = {s["tag"]: s for s in store["speakers"]}
        refreshed: dict[str, list[float]] = {}
        for tag, centroid in embeddings.items():
            entry = stored.get(tag)
            unit = _unit_centroid(centroid)
            if entry is None:
                log.warning(
                    "[5/7] speaker refresh: %s vanished from the store mid-run; "
                    "its rows keep the tag but it is not re-enrolled",
                    tag,
                )
                continue
            if unit is None:
                continue
            _update_entry(entry, unit, source_name)
            refreshed[tag] = entry["centroid"]
        store["speakers"] = list(stored.values())
        _write_store(store)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return refreshed


# --------------------------------------------------------------------------- #
# Fast diarization (the 'fast'/'auto' rungs of the diarization ladder)
# --------------------------------------------------------------------------- #
# The full pyannote pipeline slides a 10s segmentation window over the file
# and embeds one crop per local speaker per window -- that embedding stage is
# 94% of the diarize wall-clock. The K probe (OPTIMIZATION_LOG.md) measured
# that embedding ONE window per VAD region instead (with the diarizer's own
# embedding model, the same object the full pipeline uses) costs ~13s/hour vs
# ~127s for the full stage, and that a file's voice centroids separate known
# from unknown roster voices by a wide cosine canyon (0.76+ vs <=0.45).
# The fast path exploits exactly that: tile the ASR path's VAD speech regions
# into <=10s windows, embed them, cluster in numpy, and verify each cluster's
# centroid against the speaker roster. It only keeps the result when every
# voice is known and its own frame-level refinement agrees; otherwise the
# ladder escalates to the full pipeline on the same loaded object, which is
# exactly today's behavior. All quality gates come from probe measurements.
_FAST_WIN = 10.0  # embedding window (s) -- matches pyannote's crop length
_FAST_MIN_SEG = 1.0  # VAD regions shorter than this are skipped (the
# embedding model NaNs out on too few masked frames)
_FAST_MERGE_TAIL = 2.0  # trailing window piece shorter than this folds back
_FAST_BATCH = 32  # embedding/segmentation batch size (pyannote's own)
_FAST_CHANNEL_MIN = 0.4  # refinement: channel keeps a tag above this activity
_FAST_EMBED_MASS = 1.0  # unmapped channel activation (s) worth an identity
# probe -- its masked crop is embedded and matched to the clusters (see
# _refine_turns: window-level supervision is blind to sub-window turns)
_FAST_MIN_RUN = 0.15  # refinement: activation flicker shorter than this folds
# into a neighbor (pyannote's own min_duration_on default is 0 -- its turns
# can be sub-second, so keep this to flicker removal only)


def _read_roster_snapshot() -> dict[str, "np.ndarray"]:
    """``{tag: unit centroid}`` snapshot of the speaker store (read-only).

    Runs on the diarize worker under a SHARED lock on the sidecar lock file.
    The worker never writes the store; a shared lock is enough to keep the
    snapshot consistent against a concurrent main-thread enrollment.
    """
    lock_path = SPEAKER_STORE.with_name(f"{SPEAKER_STORE.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        store = _read_store()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    snapshot: dict[str, "np.ndarray"] = {}
    for s in store["speakers"]:
        unit = _unit_centroid(s.get("centroid", []))
        if unit is not None:
            snapshot[s["tag"]] = unit
    return snapshot


def _fast_vad(audio: "np.ndarray") -> tuple[list[dict], float]:
    """Speech regions via whisperx's Pyannote VAD -- the ASR path's chunker.

    Same VAD model and parameters as ``_transcribe_asr_mlx`` (onset 0.5 /
    offset 0.363) but keeping the *raw* binarized regions instead of merging
    them into 30s ASR chunks: diarization wants voice-homogeneous spans, not
    transcription-sized bites. This is a second VAD pass over audio the ASR
    path has already VAD-chunked (the ASR chunking is deliberately not
    persisted); on the worker it hides under ASR in parallel mode.
    """
    from whisperx.audio import SAMPLE_RATE
    from whisperx.vads import Pyannote
    from whisperx.vads.pyannote import Binarize

    t0 = time.perf_counter()
    vad = Pyannote(
        torch.device(PIPELINE_DEVICE),
        token=None,
        chunk_size=30,
        vad_onset=0.5,
        vad_offset=0.363,
    )
    scores = vad(
        {"waveform": Pyannote.preprocess_audio(audio), "sample_rate": SAMPLE_RATE}
    )
    # The binarizer merge_chunks applies internally, minus the 30s merge --
    # these active regions are the diarization units.
    regions = Binarize(max_duration=30, onset=0.5, offset=0.363)(scores)
    segs = [
        {"start": float(r.start), "end": float(r.end)} for r in regions.get_timeline()
    ]
    return segs, time.perf_counter() - t0


def _tile_windows(
    segs: list[dict],
) -> tuple[list[tuple[float, float, float, int]], float]:
    """Tile VAD regions into <=``_FAST_WIN`` windows.

    Returns ``(windows, skipped_seconds)`` where each window is
    ``(win_start, piece_start, piece_end, seg_idx)``: the piece is the part
    of the window that actually contains speech (the embedding mask zeroes
    the rest) and a trailing piece shorter than ``_FAST_MERGE_TAIL`` folds
    into a window ending at the region end.
    """
    windows: list[tuple[float, float, float, int]] = []
    skipped = 0.0
    for i, seg in enumerate(segs):
        s, e = seg["start"], seg["end"]
        if e - s < _FAST_MIN_SEG:
            skipped += e - s
            continue
        p = s
        while p < e - 1e-6:
            q = min(p + _FAST_WIN, e)
            if q < e and e - q < _FAST_MERGE_TAIL:  # fold the tail in
                q = e
            windows.append((p, p, q, i))
            p = q
    return windows, skipped


def _embed_windows(
    pipeline: "DiarizationPipeline",
    audio: "np.ndarray",
    windows: list[tuple[float, float, float, int]],
) -> tuple["np.ndarray", "np.ndarray", float]:
    """Embed every window with the diarizer's own embedding model.

    Waveforms are ``_FAST_WIN`` crops starting at ``win_start`` (zero-padded
    at the file tail) with a sample-resolution mask marking the speech piece
    -- the exact input contract ``pipeline.model._embedding`` uses internally.
    Returns ``(unit-normalized embeddings, usable mask, seconds)``; windows
    whose embedding is NaN or zero-norm (too little masked speech) come back
    unusable and are dropped by the caller.
    """
    t0 = time.perf_counter()
    if not windows:
        return np.empty((0, 1)), np.empty(0, dtype=bool), time.perf_counter() - t0
    emb = pipeline.model._embedding
    sr = emb.sample_rate
    n = int(_FAST_WIN * sr)
    waves: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for w0, ps, pe, _si in windows:
        i0 = int(w0 * sr)
        w = audio[i0 : i0 + n]
        if len(w) < n:
            w = np.concatenate([w, np.zeros(n - len(w), dtype=np.float32)])
        m = np.zeros(n, dtype=np.float32)
        m[max(int((ps - w0) * sr), 0) : min(int((pe - w0) * sr), n)] = 1.0
        waves.append(w)
        masks.append(m)
    wave_t = torch.from_numpy(np.stack(waves)[:, None, :])
    mask_t = torch.from_numpy(np.stack(masks))
    out = []
    for i in range(0, len(wave_t), _FAST_BATCH):
        out.append(emb(wave_t[i : i + _FAST_BATCH], masks=mask_t[i : i + _FAST_BATCH]))
    E = np.concatenate(out).astype(np.float64)
    norms = np.linalg.norm(E, axis=1)
    ok = np.isfinite(norms) & (norms > 1e-6)
    return E[ok] / norms[ok][:, None], ok, time.perf_counter() - t0


def _cluster_windows(
    E: "np.ndarray",
    durations: "np.ndarray",
    join_threshold: float,
    merge_threshold: float,
) -> tuple[list[int], int]:
    """Cluster window embeddings into voices (numpy-only, deterministic).

    Four passes, each justified by eval-file measurements (OPTIMIZATION_LOG):

    1. Greedy leaders, duration-desc: the longest unassigned window founds a
       cluster and claims every still-unassigned window within
       ``join_threshold`` -- first-claim, cheap seeds on the cleanest speech.
    2. Lloyd cleanup: recompute duration-weighted unit centroids and move
       each window to its best-cosine cluster (>= ``join_threshold``) until
       no window moves. First-claim seeding strands windows on the wrong
       side of two voices in the same file (a 74/26 mixed cluster on the
       minsuk-1-1 eval); best-centroid reassignment pulls them back.
    3. Consolidation: noise clusters (<3 windows AND <1% of speech) fold
       into their nearest survivor; near-duplicate centroids (>=
       ``merge_threshold``) merge -- one voice splits into fragments when
       its acoustic mode drifts mid-file (mic/channel change; fragments
       measured >=0.71 centroid cosine while distinct voices stayed <=0.51).
    4. One more Lloyd pass over the consolidated clusters.

    Returns ``(labels 0..k-1, number of raw seeds before consolidation)``.
    """
    n = len(E)

    def centroids_of(labs: list[int]) -> tuple[list[int], "np.ndarray"]:
        cids = sorted(set(labs))
        cent = []
        for c in cids:
            m = np.asarray(labs) == c
            v = np.average(E[m], axis=0, weights=durations[m])
            norm = float(np.linalg.norm(v))
            cent.append(v / norm if norm > 1e-9 else v)
        return cids, np.stack(cent)

    def lloyd(labs: list[int]) -> list[int]:
        for _ in range(10):
            cids, cent = centroids_of(labs)
            sims = E @ cent.T
            best = np.argmax(sims, axis=1)
            moved = False
            for j in range(n):
                b = cids[best[j]]
                if b != labs[j] and sims[j, best[j]] >= join_threshold:
                    labs[j] = b
                    moved = True
            if not moved or len(set(labs)) < 2:
                break
        return labs

    # 1. greedy duration-desc leaders
    labels = [-1] * n
    order = sorted(range(n), key=lambda i: (-float(durations[i]), i))
    for i in order:
        if labels[i] != -1:
            continue
        cid = max(labels) + 1
        labels[i] = cid
        rest = [j for j in order if labels[j] == -1]
        if rest:
            sims = E[rest] @ E[i]
            for j, sim in zip(rest, sims):
                if sim >= join_threshold:
                    labels[j] = cid
    n_raw = max(labels) + 1

    # 2. pull stranded windows to their best centroid
    labels = lloyd(labels)

    # 3a. fold noise clusters into the nearest survivor
    counts: dict[int, int] = {}
    durs: dict[int, float] = {}
    for j, cid in enumerate(labels):
        counts[cid] = counts.get(cid, 0) + 1
        durs[cid] = durs.get(cid, 0.0) + float(durations[j])
    survivors = [
        c
        for c in sorted(durs)
        if counts[c] >= 3 or durs[c] >= 0.01 * float(durations.sum())
    ]
    if 0 < len(survivors) < len(durs):
        cids, cent = centroids_of(labels)
        keep = {c: cent[i] for i, c in enumerate(cids) if c in set(survivors)}
        for j, cid in enumerate(labels):
            if cid not in keep:
                labels[j] = max(survivors, key=lambda c: (float(E[j] @ keep[c]), -c))

    # 3b. merge same-voice fragments (closest centroid pair first)
    while True:
        cids, cent = centroids_of(labels)
        if len(cids) < 2:
            break
        best: Optional[tuple[float, int, int]] = None
        for i in range(len(cids)):
            for j2 in range(i + 1, len(cids)):
                score = float(cent[i] @ cent[j2])
                if score >= merge_threshold and (best is None or score > best[0]):
                    best = (score, cids[i], cids[j2])
        if best is None:
            break
        _score, a, b = best
        labels = [a if lab == b else lab for lab in labels]

    # 4. final cleanup over the consolidated clusters
    labels = lloyd(labels)
    compact: dict[int, int] = {}
    return [compact.setdefault(lab, len(compact)) for lab in labels], n_raw


def _verify_roster(
    centroids: list["np.ndarray"], roster: dict[str, "np.ndarray"]
) -> list[tuple[str, float]]:
    """Best roster match per cluster centroid (non-exclusive).

    Mirrors ``_link_speakers``: every cluster independently takes its best
    stored voice -- two clusters may legitimately share a tag when clustering
    over-split one person. Returns one ``(tag, cosine)`` per cluster.
    """
    tags = sorted(roster)
    C = np.stack([roster[t] for t in tags])
    matches: list[tuple[str, float]] = []
    for c in centroids:
        sims = C @ c
        best = int(np.argmax(sims))
        matches.append((tags[best], float(sims[best])))
    return matches


def _coarse_turns(
    wins: list[tuple[float, float, float, int]], window_tags: list[str]
) -> list[dict]:
    """Turns from per-window tags: merge same-tag windows within a VAD region.

    Deliberately no neighborhood smoothing: with 10s windows a turn change
    lands inside two of any three voting windows, and ablating on the eval
    files showed median smoothing flipping whole windows near boundaries
    (93.4% -> 88.9% word agreement) while the frame-level refinement --
    which has real boundary evidence -- recovers them (96.5% from raw
    labels). Runs merge only within a region, so turns always break at
    silence.
    """
    rows: list[dict] = []
    last_si = -1
    for k, (_w0, ps, pe, si) in enumerate(wins):
        tag = window_tags[k]
        if rows and rows[-1]["speaker"] == tag and si == last_si:
            rows[-1]["end"] = pe
        else:
            rows.append({"start": ps, "end": pe, "speaker": tag})
        last_si = si
    return rows


def _refine_turns(
    pipeline: "DiarizationPipeline",
    audio: "np.ndarray",
    segs: list[dict],
    coarse: list[dict],
    centroids: dict[str, "np.ndarray"],
) -> tuple[list[dict], dict]:
    """Refine coarse window labels with the diarizer's segmentation substrate.

    The coarse path labels whole 10s windows, so a turn change inside a window
    lands on the wrong side of the boundary -- the K probe found nearly all
    fast-path error within 2-3s of a real turn boundary and *not* recoverable
    from label transitions alone. The segmentation head is cheap (~6% of the
    full stage) and yields frame-level (~16ms) per-channel activations for the
    whole file in one batched pass. Its output channels are anonymous and
    permuted per chunk, so each chunk's channels are mapped to tags under
    supervision of the coarse labels (a channel keeps its top tag iff it is
    that tag's best channel and carries >= ``_FAST_CHANNEL_MIN`` of its own
    activation). Overlapping chunks then contribute their mapped channels'
    activation mass per frame (soft voting -- ablation showed hard per-chunk
    argmax saturates ~4 points lower, because it discards how *close* the
    second voice is at every frame); frames with no mass inherit the coarse
    label. Turns are re-cut from that frame consensus, breaking at VAD gaps
    and folding sub-``_FAST_MIN_RUN`` runs into their longer neighbor.

    Supervision alone cannot see turns shorter than a window: a 10s window
    dominated 8:2 by voice A labels B's channel as A, B never earns vote
    mass, and whole interjections get swallowed (minsuk-1-1: 38 refined turns
    vs pyannote's 190, a 4.2s SPEAKER_01 turn fully inside a SPEAKER_00 run).
    So once, after the first supervised mapping, every *unexplained* channel
    (>= ``_FAST_EMBED_MASS`` of activation that supervision did not keep) gets
    an identity probe: its masked crop is embedded -- pyannote's own channel
    identification mechanism -- and matched against ``centroids`` (the
    roster-verified cluster centroids). A match at the join threshold
    overrides supervision: identity evidence outranks blocky timing evidence.
    Only mixed chunks carry unexplained mass, so the probe costs a fraction
    of the window-embedding stage.

    Returns ``(rows, stats)``. ``stats["flip_frac"]`` is the speech fraction
    where the refined trace disagrees with the coarse labels -- the fast
    path's two independent views of the same audio, and the ladder's T9
    escalation trigger when they disagree too much.
    """
    t0 = time.perf_counter()
    from whisperx.audio import SAMPLE_RATE

    seg_inf = pipeline.model._segmentation
    wave = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    feat = seg_inf({"waveform": wave[None], "sample_rate": SAMPLE_RATE})
    data = np.asarray(feat.data, dtype=np.float64)
    if data.ndim != 3:
        # Without per-chunk channels the channel->tag supervision below is
        # meaningless (e.g. an aggregated 2D output) -- keep the coarse turns.
        log.warning(
            "[5/7] fast diarization: segmentation substrate returned shape %s "
            "(expected chunks x frames x channels); keeping coarse turns",
            data.shape,
        )
        return coarse, {"flip_frac": 0.0, "fallback": f"shape {data.shape}"}
    n_chunks, n_frames, _n_channels = data.shape
    chunk_dur = float(getattr(seg_inf, "duration", 0.0) or 10.0)
    step = float(getattr(seg_inf, "step", 0.0) or DIARIZE_SEG_STEP)
    res = chunk_dur / n_frames  # frame resolution (~16ms)
    tags_list = sorted({r["speaker"] for r in coarse})
    tag_ids = {t: i for i, t in enumerate(tags_list)}
    n_global = int(round((n_chunks - 1) * step / res)) + n_frames

    coarse_trace = np.full(n_global, -1, dtype=int)
    for r in coarse:
        lo = int(np.floor(r["start"] / res))
        hi = int(np.ceil(r["end"] / res))
        coarse_trace[lo:hi] = tag_ids[r["speaker"]]
    speech = np.zeros(n_global, dtype=bool)
    for s in segs:
        speech[int(np.floor(s["start"] / res)) : int(np.ceil(s["end"] / res))] = True

    def _forward_fill(final: np.ndarray) -> np.ndarray:
        # Label trace over *every* frame: silent/unvoted frames carry the
        # previous label forward (backfilling the head/tail) so short pauses do
        # not shred turns -- pyannote labels silent frames too, and its default
        # min_duration_on/off are 0, so its turns bridge micro-gaps and can be
        # sub-second long. Breaking runs at any non-speech frame instead cost
        # several points of word agreement on the eval files (it swallowed the
        # short GT turns whole).
        trace = final.copy()
        valid = np.nonzero(trace >= 0)[0]
        if len(valid):
            trace[: valid[0]] = trace[valid[0]]
            for i in range(valid[0] + 1, valid[-1] + 1):
                if trace[i] < 0:
                    trace[i] = trace[i - 1]
            trace[valid[-1] + 1 :] = trace[valid[-1]]
        return trace

    def _probe_unexplained(
        kept: list[dict[int, int]],
    ) -> dict[int, list[tuple[int, int, float]]]:
        """Identify channels by masked-crop embedding, not by time labels.

        Supervision goes backwards exactly where it matters: a chunk of pure
        voice B whose covering window straddled a boundary A-majority gets
        B's channel mapped to A (minsuk-1-1 chunk [686,696]: 9.9s of pure
        SPEAKER_01 at masked-crop cos 0.86, supervised as SPEAKER_00), and it
        then votes for the wrong voice with full mass. So every channel
        carrying >= ``_FAST_EMBED_MASS`` of activation in a chunk with two or
        more such channels -- the mixed chunks, and every boundary-adjacent
        chunk -- is identified by embedding its own masked crop; a lone
        massy channel is only probed when supervision left it unmapped (pure
        stretches deep inside a turn are already right).
        """
        jobs: list[tuple[int, int]] = []
        for c in range(n_chunks):
            act = data[c]
            massy = [
                k
                for k in range(act.shape[1])
                if act[:, k].sum() * res >= _FAST_EMBED_MASS
            ]
            if len(massy) >= 2:
                jobs.extend((c, k) for k in massy)
            elif len(massy) == 1 and massy[0] not in kept[c]:
                jobs.append((c, massy[0]))
        if not jobs:
            return {}
        emb = pipeline.model._embedding
        sr = emb.sample_rate
        n = int(round(chunk_dur * sr))
        spf = n // n_frames  # samples per frame (256 at 16ms/16kHz)
        waves: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for c, k in jobs:
            i0 = int(round(c * step * sr))
            w = audio[i0 : i0 + n]
            if len(w) < n:
                w = np.concatenate([w, np.zeros(n - len(w), dtype=np.float32)])
            m = np.repeat(data[c][:, k], spf).astype(np.float32)
            if len(m) < n:
                m = np.concatenate([m, np.zeros(n - len(m), dtype=np.float32)])
            waves.append(w)
            masks.append(m[:n])
        wave_t = torch.from_numpy(np.stack(waves)[:, None, :])
        mask_t = torch.from_numpy(np.stack(masks))
        out = []
        for i in range(0, len(wave_t), _FAST_BATCH):
            out.append(
                emb(wave_t[i : i + _FAST_BATCH], masks=mask_t[i : i + _FAST_BATCH])
            )
        E = np.concatenate(out).astype(np.float64)
        tags_c = np.stack([centroids[t] for t in tags_list])
        fixes: dict[int, list[tuple[int, int, float]]] = {}
        for (c, k), vec in zip(jobs, E):
            nrm = float(np.linalg.norm(vec))
            if not np.isfinite(nrm) or nrm <= 1e-6:
                continue  # too little masked audio for an identity
            sims = tags_c @ (vec / nrm)
            tid = int(np.argmax(sims))
            if sims[tid] >= DIARIZE_FAST_CLUSTER_THRESHOLD:
                fixes.setdefault(c, []).append((k, tid, float(sims[tid])))
        for lst in fixes.values():
            lst.sort(key=lambda f: -f[2])
        log.info(
            "[5/7] fast diarization: identity probe embedded %d unexplained "
            "channel(s) from mixed chunks, %d matched a voice cluster",
            len(jobs),
            sum(len(v) for v in fixes.values()),
        )
        return fixes

    # Iterated refinement. Round 0 supervises the per-chunk channel->tag maps
    # with the *coarse* labels, whose boundaries are blocky at window
    # granularity: a chunk straddling a real turn change often holds no
    # supervised frames for the second voice, so that voice's channel stays
    # unmapped and gets no vote mass near the boundary -- exactly the 1-2s
    # displacement band in the error anatomy. The identity probe patches the
    # gaps supervision cannot fill, and the round-0 consensus then places
    # boundaries at the activation-mass crossover, so re-supervising the maps
    # with it and re-voting sharpens them further; the loop converges in 2-3
    # rounds and everything past the substrate pass is cheap numpy.
    trace = _forward_fill(coarse_trace)
    embed_fixes: dict[int, list[tuple[int, int, float]]] = {}
    for _round in range(3):
        kept = []
        for c in range(n_chunks):
            g0 = int(round(c * step / res))
            frame_tag = trace[g0 : g0 + n_frames]
            act = data[c]  # (frames x channels) activations
            score = np.zeros((act.shape[1], len(tags_list)))
            for tid in range(len(tags_list)):
                m = frame_tag == tid
                if m.any():
                    score[:, tid] = act[m].sum(axis=0)
            channel_total = score.sum(axis=1)
            mapping: dict[int, int] = {}
            for k in range(act.shape[1]):
                tid = int(np.argmax(score[k]))
                if score[k, tid] <= 0:
                    continue
                if int(np.argmax(score[:, tid])) == k and score[k, tid] >= (
                    _FAST_CHANNEL_MIN * max(channel_total[k], 1e-9)
                ):
                    mapping[k] = tid
            kept.append(mapping)
        if _round == 0:
            embed_fixes = _probe_unexplained(kept)
        for c, fixes in embed_fixes.items():
            m = kept[c]
            for k, tid, _cos in fixes:  # best cosine first
                for k2 in [kx for kx, tx in m.items() if tx == tid and kx != k]:
                    del m[k2]
                m[k] = tid
        # Frame-level consensus: each covering chunk contributes its mapped
        # channels' activation mass, so a boundary lands at the mass crossover
        # between the two voices rather than at some chunk's hard argmax edge.
        votes = np.zeros((n_global, len(tags_list)))
        for c in range(n_chunks):
            if not kept[c]:
                continue
            act = data[c]
            g0 = int(round(c * step / res))
            for k, tid in kept[c].items():
                votes[g0 : g0 + n_frames, tid] += act[:, k]
        won = votes.argmax(axis=1)
        final = np.where(votes.sum(axis=1) > 0, won, trace)
        new_trace = _forward_fill(final)
        if np.array_equal(new_trace, trace):
            break
        trace = new_trace

    # T9 measures refinement movement against the *original* coarse view.
    judged = speech & (coarse_trace >= 0)
    flip_frac = (
        float((trace[judged] != coarse_trace[judged]).mean()) if judged.any() else 0.0
    )

    # Runs = maximal same-label stretches; trim each to the VAD speech inside
    # it and drop runs with none. Only activation flicker (sub-min-run) folds
    # into its longer neighbor.
    runs: list[list[int]] = []  # [start_frame, end_frame, tag_id]
    start = 0
    for i in range(1, n_global + 1):
        if i == n_global or trace[i] != trace[start]:
            runs.append([start, i, int(trace[start])])
            start = i
    trimmed: list[dict] = []
    for a, b, t in runs:
        idx = np.nonzero(speech[a:b])[0]
        if not len(idx):
            continue
        trimmed.append(
            {
                "start": (a + int(idx[0])) * res,
                "end": (a + int(idx[-1]) + 1) * res,
                "speaker": tags_list[t],
            }
        )
    if len(trimmed) > 1:
        changed = True
        while changed and len(trimmed) > 1:
            changed = False
            for i, r in enumerate(trimmed):
                if r["end"] - r["start"] >= _FAST_MIN_RUN:
                    continue
                prev_r = trimmed[i - 1] if i > 0 else None
                next_r = trimmed[i + 1] if i + 1 < len(trimmed) else None
                if prev_r is None:
                    next_r["start"] = r["start"]
                elif next_r is None or (prev_r["end"] - prev_r["start"]) >= (
                    next_r["end"] - next_r["start"]
                ):
                    prev_r["end"] = r["end"]
                else:
                    next_r["start"] = r["start"]
                trimmed.pop(i)
                changed = True
                break
    # Single-frame VAD seams can split a turn into two same-tag rows that meet
    # exactly -- stitch those back.
    rows: list[dict] = []
    for r in trimmed:
        if (
            rows
            and rows[-1]["speaker"] == r["speaker"]
            and r["start"] - rows[-1]["end"] < res
        ):
            rows[-1]["end"] = r["end"]
        else:
            rows.append(r)
    stats = {
        "flip_frac": round(flip_frac, 4),
        "coarse_turns": len(coarse),
        "refined_turns": len(rows),
        "probe_fixed": sum(len(v) for v in embed_fixes.values()),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    log.info(
        "[5/7] fast diarization: refinement %.1fs, %d chunks, flip_frac %.3f "
        "(%d coarse -> %d refined turns)",
        stats["seconds"],
        n_chunks,
        flip_frac,
        len(coarse),
        len(rows),
    )
    return rows, stats


def _fast_diarize(
    pipeline: "DiarizationPipeline",
    audio: "np.ndarray",
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    mode: str,
) -> tuple[Optional[list[dict]], dict[str, list[float]], dict]:
    """One fast-path attempt. Returns ``(rows, centroids_by_tag, info)``.

    ``rows is None`` means the attempt refused its own result -- ``info[
    "reasons"]`` names the ladder trigger(s) that fired and the caller
    escalates to the full pipeline. Empty ``rows`` (VAD found no speech) is a
    *success*: there is nothing for pyannote to diarize either. On success the
    rows already carry global roster tags (that is what was verified).
    """
    info: dict = {"strategy": "fast", "mode": mode, "reasons": []}

    roster = _read_roster_snapshot()
    if not roster:
        info["reasons"].append("roster empty -- no enrolled voices to verify against")
        return None, {}, info

    segs, vad_s = _fast_vad(audio)
    if not segs:
        info.update({"voices": 0, "windows": 0})
        log.info("[5/7] fast diarization: VAD found no speech (%.1fs)", vad_s)
        return [], {}, info
    windows, skipped = _tile_windows(segs)
    log.info(
        "[5/7] fast diarization: %d VAD regions -> %d windows, %.1fs speech "
        "skipped <%ss (vad %.1fs)",
        len(segs),
        len(windows),
        skipped,
        _FAST_MIN_SEG,
        vad_s,
    )
    E, ok, emb_s = _embed_windows(pipeline, audio, windows)
    wins = [w for w, o in zip(windows, ok) if o]
    log.info(
        "[5/7] fast diarization: %d/%d windows embedded in %.1fs",
        len(wins),
        len(windows),
        emb_s,
    )
    if len(wins) < 2:
        info["reasons"].append(f"only {len(wins)} usable window embeddings (<2)")
        return None, {}, info
    info["windows"] = len(wins)
    durs = np.array([w[2] - w[1] for w in wins])
    labels, n_raw = _cluster_windows(
        E, durs, DIARIZE_FAST_CLUSTER_THRESHOLD, DIARIZE_FAST_MERGE_THRESHOLD
    )
    labels_arr = np.asarray(labels)
    voices = int(labels_arr.max()) + 1
    info["voices"] = voices
    log.info(
        "[5/7] fast diarization: %d voice cluster(s) (join>=%.2f; %d raw seed(s) "
        "consolidated at %.2f)",
        voices,
        DIARIZE_FAST_CLUSTER_THRESHOLD,
        n_raw,
        DIARIZE_FAST_MERGE_THRESHOLD,
    )
    if voices > DIARIZE_FAST_MAX_VOICES:
        info["reasons"].append(
            f"{voices} voices > DIARIZE_FAST_MAX_VOICES={DIARIZE_FAST_MAX_VOICES}"
        )
        return None, {}, info
    if (min_speakers is not None and voices < min_speakers) or (
        max_speakers is not None and voices > max_speakers
    ):
        info["reasons"].append(
            f"{voices} voices outside requested [{min_speakers}, {max_speakers}]"
        )
        return None, {}, info

    cent: list["np.ndarray"] = []
    for cid in range(voices):
        member = labels_arr == cid
        vec = np.average(E[member], axis=0, weights=durs[member])
        cent.append(vec / float(np.linalg.norm(vec)))
    matches = _verify_roster(cent, roster)
    info["matches"] = {
        f"cluster {i}": {"tag": tag, "cos": round(score, 3)}
        for i, (tag, score) in enumerate(matches)
    }
    for i, (tag, score) in enumerate(matches):
        log.info(
            "[5/7] fast diarization: cluster %d -> %s (cos %.3f)",
            i,
            tag,
            score,
        )
        if score < SPEAKER_MATCH_THRESHOLD:
            info["reasons"].append(
                f"cluster {i} is an unknown voice (best roster cos "
                f"{score:.3f} < {SPEAKER_MATCH_THRESHOLD})"
            )
    if info["reasons"]:
        return None, {}, info

    # Per-window roster margins -- the 'auto' escalation stat. The same
    # threshold doubles as the refinement on/off knob (0 disables both).
    tags_sorted = sorted(roster)
    S = E @ np.stack([roster[t] for t in tags_sorted]).T
    if S.shape[1] > 1:
        srt = np.sort(S, axis=1)
        margins = srt[:, -1] - srt[:, -2]
        low_margin_frac = float((margins < DIARIZE_FAST_REFINE).mean())
    else:
        low_margin_frac = 0.0
    info["low_margin_frac"] = round(low_margin_frac, 4)
    log.info(
        "[5/7] fast diarization: %.1f%% of windows below roster margin %.2f",
        low_margin_frac * 100,
        DIARIZE_FAST_REFINE,
    )

    window_tags = [matches[lab][0] for lab in labels]
    coarse = _coarse_turns(wins, window_tags)
    if DIARIZE_FAST_REFINE > 0:
        centroids_by_tag = {matches[cid][0]: cent[cid] for cid in range(voices)}
        rows, rstats = _refine_turns(pipeline, audio, segs, coarse, centroids_by_tag)
        info["refined"] = rstats
        if rstats.get("flip_frac", 0.0) > DIARIZE_FAST_MAX_FLIP:
            info["reasons"].append(
                f"refinement flipped {rstats['flip_frac']:.0%} of speech frames "
                f"> DIARIZE_FAST_MAX_FLIP={DIARIZE_FAST_MAX_FLIP}"
            )
            return None, {}, info
    else:
        info["refined"] = {"disabled": True}
        rows = coarse
    log.info(
        "[5/7] diarization strategy: FAST -- %d known voices, %d windows, %d turns",
        voices,
        len(wins),
        len(rows),
    )

    # Per-tag centroids: clusters that share a tag (over-split voice) merge,
    # duration-weighted -- mirroring the non-exclusive matching above.
    acc: dict[str, tuple["np.ndarray", float]] = {}
    for cid, (tag, _score) in enumerate(matches):
        d = float(durs[labels_arr == cid].sum())
        if tag in acc:
            v0, d0 = acc[tag]
            acc[tag] = ((v0 * d0 + cent[cid] * d) / (d0 + d), d0 + d)
        else:
            acc[tag] = (cent[cid], d)
    centroids_out: dict[str, list[float]] = {}
    for tag, (v, _d) in acc.items():
        norm = float(np.linalg.norm(v))
        centroids_out[tag] = (v / norm if norm > 1e-9 else v).tolist()
    return rows, centroids_out, info


# --------------------------------------------------------------------------- #
# Pipeline engine core (sequential, blocking)
# --------------------------------------------------------------------------- #
def run_pipeline(
    audio_path: str,
    *,
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarize_mode: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    resume: bool = RESUME,
) -> list[dict]:
    """Transcribe ``audio_path`` end-to-end and return a clean speaker script.

    Stages run one model at a time on the critical path (each freed before the
    next loads), with one structural exception: diarization depends only on the
    audio, so when ASR/alignment also need computing it runs concurrently on a
    worker thread and is joined before speaker assignment (PARALLEL_STAGES):

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
    input_name = Path(audio_path).name  # before it becomes converted.mp3 below

    ckpt: Optional[Checkpoint] = None
    if resume:
        ckpt = Checkpoint.open(
            audio_path,
            _cache_params(language, min_speakers, max_speakers, diarize_mode),
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

        # What still needs computing? (a cache hit skips that stage.)
        need_asr = not (ckpt and ckpt.has(_CKPT_ASR))
        need_aligned = not (ckpt and ckpt.has(_CKPT_ALIGNED))
        need_diarize = not (ckpt and ckpt.has(_CKPT_DIARIZE))

        # ASR, alignment and diarization all consume the same 16 kHz waveform;
        # load it once up front.
        if need_asr or need_aligned or need_diarize:
            audio = whisperx.load_audio(audio_path)

        # Diarization depends only on the audio, not on the ASR result -- when
        # the other heavy half also needs computing, run it on a worker thread
        # in parallel and join before speaker assignment.
        diarize_future = None
        if need_diarize and (need_asr or need_aligned) and _parallel_diarize_enabled():
            log.info("[5/7] diarization running in parallel with ASR/alignment")
            diarize_future = _DIARIZE_POOL.submit(
                _run_diarization, audio, min_speakers, max_speakers, diarize_mode
            )

        # 2. ASR ----------------------------------------------------------- #
        if not need_asr:
            log.info("[2/7] cache hit: ASR result")
            with _timed(timings, "asr"):
                result = ckpt.load_json(_CKPT_ASR)
        else:
            with _timed(timings, "asr"):
                if _effective_asr_backend() == "mlx":
                    result = _transcribe_asr_mlx(audio, language)
                    # MLX caches the model in-process (ModelHolder); nothing to
                    # free between runs, and the next stage owns the GPU anyway.
                    log.info("[3/7] MLX model stays cached in-process")
                else:
                    log.info(
                        "[2/7] loading ASR model %s on %s "
                        "(compute_type=%s, threads=%d)",
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
                    log.info("[2/7] transcribing (batch_size=%d)", batch_size)
                    result = model.transcribe(
                        audio, batch_size=batch_size, language=language
                    )
                    # 3. free ASR model before touching alignment ------------- #
                    log.info("[3/7] releasing ASR model")
                    release_model(model)
            if ckpt:
                ckpt.save_json(_CKPT_ASR, result)

        detected_language = result.get("language", language)

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
        if not need_diarize:
            log.info("[5/7] cache hit: diarization")
            with _timed(timings, "diarize"):
                payload = ckpt.load_json(_CKPT_DIARIZE)
                if isinstance(payload, dict):
                    rows = payload["turns"]
                    diarize_info = payload.get("diarization") or {}
                else:
                    # Legacy bare-list checkpoint from before speaker linking:
                    # already labeled, and no centroids to relink against.
                    log.info(
                        "[5/7] legacy diarize.json (pre-linking); keeping stored labels"
                    )
                    rows = payload
                    diarize_info = {}
                if diarize_info.get("strategy"):
                    log.info(
                        "[5/7] cache hit: diarization (strategy %s, mode %s)",
                        diarize_info.get("strategy"),
                        diarize_info.get("mode"),
                    )
                diarize_segments = pd.DataFrame(
                    rows, columns=["start", "end", "speaker"]
                )
        else:
            if diarize_future is not None:
                # Join the parallel diarization; re-raises any worker error
                # here, after the ASR/alignment checkpoints were saved above.
                diarize_segments, centroids, elapsed, diarize_info = (
                    diarize_future.result()
                )
                timings["diarize"] = round(elapsed, 3)
            else:
                with _timed(timings, "diarize"):
                    diarize_segments, centroids, _, diarize_info = _run_diarization(
                        audio, min_speakers, max_speakers, diarize_mode
                    )
            rows = _diarize_rows(diarize_segments)
            embeddings_out: dict[str, list[float]] = {}
            if SPEAKER_LINKING and rows:
                if diarize_info.get("strategy") == "fast" and centroids:
                    # Fast-path rows already carry stable roster tags (that is
                    # what was verified) -- nothing to relabel or allocate;
                    # fold this run's centroids into the existing entries.
                    link_t0 = time.perf_counter()
                    embeddings_out = _refresh_enrolled(centroids, input_name)
                    log.info(
                        "[5/7] speaker refresh done in %.2fs",
                        time.perf_counter() - link_t0,
                    )
                elif centroids:
                    link_t0 = time.perf_counter()
                    rows, embeddings_out = _link_speakers(rows, centroids, input_name)
                    log.info(
                        "[5/7] speaker linking done in %.2fs",
                        time.perf_counter() - link_t0,
                    )
                else:
                    log.info(
                        "[5/7] speaker linking skipped: no speaker centroids returned"
                    )
            payload: dict = {"turns": rows}
            if SPEAKER_LINKING:
                payload["embeddings"] = embeddings_out
            payload["diarization"] = _jsonable(diarize_info)
            if ckpt:
                ckpt.save_json(_CKPT_DIARIZE, payload)
            diarize_segments = pd.DataFrame(rows, columns=["start", "end", "speaker"])

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
    diarize_mode: Optional[str] = None,
    resume: bool = RESUME,
) -> list[dict]:
    """Run the blocking pipeline in a worker thread so the loop stays responsive."""
    return await asyncio.to_thread(
        run_pipeline,
        audio_path,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarize_mode=diarize_mode,
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
    diarize_mode: Optional[str] = None,
    resume: Optional[bool] = None,
) -> dict:
    """Accept an audio upload and kick off an asynchronous transcription job."""
    try:
        _diarize_mode(diarize_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    log.info(
        "queued task %s (%s, resume=%s, diarize_mode=%s)",
        task_id,
        in_path.name,
        use_resume,
        _diarize_mode(diarize_mode),
    )

    asyncio.create_task(
        _run_transcription_task(
            task_id,
            str(in_path),
            language,
            min_speakers,
            max_speakers,
            diarize_mode,
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
    diarize_mode: Optional[str],
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
            diarize_mode=diarize_mode,
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
        diarize_mode=args.diarize_mode,
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
        "--diarize-mode",
        choices=["fast", "auto", "pyannote"],
        default=None,
        help="Diarization strategy: 'fast' (VAD-window embeddings verified "
        "against the speaker roster, escalates to full pyannote when it "
        "cannot be trusted), 'auto' (fast, additionally escalating on weak "
        "quality signals), or 'pyannote' (always the full pipeline). "
        "Defaults to DIARIZE_MODE from the environment ('fast').",
    )
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
