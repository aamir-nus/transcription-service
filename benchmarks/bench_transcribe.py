"""Wall-clock + quality benchmark for the transcription pipeline.

Runs the pipeline (`transcribe.py transcribe`) on short audio snippets with
`--no-resume`, parses the per-stage timings from the `[done]` log line, and
compares the resulting transcript against the matching window of a known-good
full-file baseline. Designed to answer two questions quickly:

  * **Speed**: how many seconds per stage, and the real-time factor
    (stage seconds / audio seconds)? Especially the diarization stage, which was
    the bottleneck on CPU.
  * **Quality**: does the optimized run (e.g. MPS for align/diarize) produce the
    same transcript text as the existing baseline for the first 90s/120s?

Usage::

    python benchmarks/bench_transcribe.py                  # default matrix
    python benchmarks/bench_transcribe.py --snippets 90 120 --devices mps cpu

The default matrix runs both snippets under the current device config (mps) and
also runs the longest snippet under forced CPU, so the diarization speedup is
visible head-to-head. Snippets are created on demand from
``Data/s3-and-research.opus``.

All output is a plain-text report to stdout; nothing here mutates the repo.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "Data"
BASELINE_VTT = REPO / "Output" / "s3-and-research.vtt"
SNIPPET_SOURCE = DATA_DIR / "s3-and-research.opus"

# Stage order in the [done] summary line, as emitted by run_pipeline.
STAGE_KEYS = ["convert", "asr", "align", "diarize", "assign", "build"]

_DONE_RE = re.compile(
    r"\[done\] pipeline complete: (\d+) segments in ([\d.]+)s \((.*)\)"
)
_STAGE_RE = re.compile(r"(\w+) ([\d.]+)")


@dataclass
class RunResult:
    """One pipeline run's parsed outcome."""

    label: str
    audio_seconds: float
    segments: int = 0
    total: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)
    vtt_path: Path | None = None
    error: str | None = None

    @property
    def rtf(self) -> float:
        """Real-time factor: total wall-clock / audio length (>1 = slower than realtime)."""
        return self.total / self.audio_seconds if self.audio_seconds else 0.0

    def stage_rtf(self, stage: str) -> float:
        t = self.stages.get(stage, 0.0)
        return t / self.audio_seconds if self.audio_seconds else 0.0


# --------------------------------------------------------------------------- #
# Snippet preparation
# --------------------------------------------------------------------------- #
def ensure_snippet(seconds: int) -> Path:
    """Create Data/s3-and-research-<N>s.opus from the source if missing."""
    out = DATA_DIR / f"s3-and-research-{seconds}s.opus"
    if out.exists():
        return out
    if not SNIPPET_SOURCE.exists():
        raise FileNotFoundError(f"source audio missing: {SNIPPET_SOURCE}")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "0",
            "-t",
            str(seconds),
            "-i",
            str(SNIPPET_SOURCE),
            "-c",
            "copy",
            str(out),
        ],
        check=True,
    )
    return out


def audio_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


# --------------------------------------------------------------------------- #
# Running the pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(snippet: Path, device: str, out_dir: Path) -> RunResult:
    """Run transcribe.py on `snippet` (no-resume) and parse its [done] line.

    `device` is the PIPELINE_DEVICE for align/diarize ('mps' or 'cpu'); ASR is
    always CPU (faster-whisper has no MPS backend).
    """
    label = f"{snippet.stem} @ {device} +asr:{os.getenv('ASR_BACKEND', 'auto')}"
    out_vtt = out_dir / f"{snippet.stem}.{device}.vtt"
    res = RunResult(
        label=label, audio_seconds=audio_duration(snippet), vtt_path=out_vtt
    )

    env = dict(os.environ)
    env["DEVICE"] = device  # PIPELINE_DEVICE for the PyTorch stages
    env["RESUME"] = "false"  # force a full fresh run each time
    # Keep the speaker store out of benchmark runs: snippets are cuts of the
    # baseline file, so linking them would re-tag their voices with global
    # tags and make the report's speaker column incomparable.
    env["SPEAKER_LINKING"] = "false"
    # Force the full pipeline: with linking off the fast ladder would never
    # run anyway, but pin the mode so the diarize bucket stays comparable to
    # the historical numbers regardless of the DIARIZE_MODE default.
    env["DIARIZE_MODE"] = "pyannote"

    py = sys.executable
    cmd = [
        py,
        str(REPO / "transcribe.py"),
        "transcribe",
        str(snippet),
        "--format",
        "vtt",
        "--no-resume",
        "-o",
        str(out_vtt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stderr = proc.stderr

    if proc.returncode != 0:
        res.error = (stderr or proc.stdout)[-1500:]
        return res

    for line in stderr.splitlines():
        m = _DONE_RE.search(line)
        if not m:
            continue
        res.segments = int(m.group(1))
        res.total = float(m.group(2))
        res.stages = {k: float(v) for k, v in _STAGE_RE.findall(m.group(3))}
        break
    else:
        res.error = "[done] line not found in log:\n" + stderr[-1500:]
    return res


# --------------------------------------------------------------------------- #
# Quality comparison vs the full-file baseline
# --------------------------------------------------------------------------- #
_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[\.,](\d{3})")
_SPEAKER = re.compile(r"<v\s+([^>]+)>\s*(.*)")


def _vtt_to_rows(path: Path) -> list[tuple[float, float, str, str]]:
    """Parse a VTT into [(start, end, speaker, text), ...]."""
    rows: list[tuple[float, float, str, str]] = []
    blocks = path.read_text(encoding="utf-8").replace("\r", "").split("\n\n")
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        # find the cue-time line
        cue_idx = next((i for i, ln in enumerate(lines) if " --> " in ln), None)
        if cue_idx is None:
            continue
        m = _TS.search(lines[cue_idx])
        if not m:
            continue
        h, mi, s, ms = (int(x) for x in m.groups())
        start = h * 3600 + mi * 60 + s + ms / 1000.0
        m2 = _TS.search(lines[cue_idx][m.end() :])
        end = start
        if m2:
            h2, mi2, s2, ms2 = (int(x) for x in m2.groups())
            end = h2 * 3600 + mi2 * 60 + s2 + ms2 / 1000.0
        text = " ".join(lines[cue_idx + 1 :]).strip()
        sp = _SPEAKER.match(text)
        speaker, clean = (sp.group(1), sp.group(2)) if sp else ("?", text)
        rows.append((start, end, speaker, clean.strip()))
    return rows


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _wer(ref: str, hyp: str) -> float:
    """Word error rate of `hyp` against `ref` (both normalized text).

    Unlike the char-level similarity, WER is comparable across ASR backends:
    char similarity collapses when decoder styles differ (fillers, punctuation)
    even when the words are the same.
    """
    ref_w, hyp_w = ref.split(), hyp.split()
    dp = list(range(len(hyp_w) + 1))
    for i, rw in enumerate(ref_w, 1):
        prev, dp[0] = dp[0], i
        for j, hw in enumerate(hyp_w, 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (rw != hw))
            prev, dp[j] = dp[j], cur
    return dp[-1] / max(1, len(ref_w))


def compare_to_baseline(run: RunResult, window: float) -> dict:
    """Compare run's transcript to the baseline's first `window` seconds."""
    if not run.vtt_path or not run.vtt_path.exists() or not BASELINE_VTT.exists():
        return {"similarity": None}
    run_rows = _vtt_to_rows(run.vtt_path)
    base_rows = [r for r in _vtt_to_rows(BASELINE_VTT) if r[0] < window]

    run_text = _normalize(" ".join(r[3] for r in run_rows))
    base_text = _normalize(" ".join(r[3] for r in base_rows))
    sim = difflib.SequenceMatcher(None, base_text, run_text).ratio()

    # speaker-label stability: do the same labels appear in the same rough order?
    run_spk = [r[2] for r in run_rows]
    base_spk = [r[2] for r in base_rows]
    return {
        "similarity": sim,
        "wer": _wer(base_text, run_text),
        "run_segments": len(run_rows),
        "base_segments": len(base_rows),
        "run_speakers": sorted(set(run_spk)),
        "base_speakers": sorted(set(base_spk)),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(results: list[RunResult], window_by_snippet: dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("SPEED (wall-clock seconds per stage)")
    print("=" * 78)
    header = f"{'run':<26}{'audio':>7}{'total':>8}{'RTF':>7}"
    for s in STAGE_KEYS:
        header += f"{s:>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"{r.label:<26}{'ERROR':>7}  {r.error.splitlines()[-1][:80]}")
            continue
        line = f"{r.label:<26}{r.audio_seconds:>7.1f}{r.total:>8.1f}{r.rtf:>7.2f}"
        for s in STAGE_KEYS:
            line += f"{r.stages.get(s, 0.0):>9.1f}"
        print(line)

    print("\n" + "=" * 78)
    print("QUALITY (transcript text vs full-file baseline, first N seconds)")
    print("=" * 78)
    print(f"{'run':<26}{'sim':>7}{'wer':>7}{'run_seg':>9}{'base_seg':>9}  speakers")
    print("-" * 78)
    for r in results:
        if r.error or not r.vtt_path:
            continue
        # infer the window from the snippet name (e.g. "...-90s..." -> 90)
        m = re.search(r"-(\d+)s", r.label)
        window = window_by_snippet.get(r.label, float(m.group(1)) if m else 0.0)
        q = compare_to_baseline(r, window)
        sim = q["similarity"]
        simstr = f"{sim:.3f}" if sim is not None else "  n/a"
        wer = q.get("wer")
        werstr = f"{wer:.3f}" if wer is not None else "  n/a"
        spk = ",".join(q.get("run_speakers") or []) or "-"
        print(
            f"{r.label:<26}{simstr:>7}{werstr:>7}{q.get('run_segments', '-'):>9}"
            f"{q.get('base_segments', '-'):>9}  {spk}"
        )
    print()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--snippets",
        type=int,
        nargs="+",
        default=[90, 120],
        help="snippet lengths in seconds (default: 90 120)",
    )
    ap.add_argument(
        "--devices",
        nargs="+",
        default=["mps", "cpu"],
        help="pipeline devices to test (default: mps cpu)",
    )
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="only run the current env device, skip the head-to-head",
    )
    args = ap.parse_args()

    snippets = [ensure_snippet(s) for s in args.snippets]
    out_dir = Path(tempfile.mkdtemp(prefix="bench-"))

    # Build the run matrix: every snippet under the env device, plus each extra
    # device on the longest snippet for a clean head-to-head speedup.
    env_device = os.getenv("DEVICE", "auto")
    longest = max(snippets, key=lambda p: audio_duration(p))
    plan: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(snip: Path, dev: str) -> None:
        key = (snip.name, dev)
        if key not in seen:
            seen.add(key)
            plan.append((snip, dev))

    for snip in snippets:
        add(snip, env_device)
    if not args.baseline_only:
        for dev in args.devices:
            if dev != env_device:
                add(longest, dev)

    print(f"snippets : {[s.name for s in snippets]}")
    print(f"baseline : {BASELINE_VTT.name} (exists={BASELINE_VTT.exists()})")
    print(f"out dir  : {out_dir}")
    print(f"plan     : {[(s.name, d) for s, d in plan]}")

    results: list[RunResult] = []
    for snip, dev in plan:
        print(f"\n--- running {snip.name} @ {dev} ---", flush=True)
        results.append(run_pipeline(snip, dev, out_dir))

    window_by_snippet = {
        f"{s.name} @ {d}": float(re.search(r"-(\d+)s", s.name).group(1))
        for s, d in plan
    }
    print_report(results, window_by_snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
