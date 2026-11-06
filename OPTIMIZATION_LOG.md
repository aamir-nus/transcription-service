# Optimization log (append-only)

Running notes on performance experiments, so parallel agents can pick up
context without re-deriving it. One entry per experiment: what → outcome →
next. Newest at the bottom; never rewrite old entries.

**House rule for benchmarks**: numbers are only comparable **back-to-back in
the same machine state**. CT2 CPU speed on this Mac swings ~4× between
cold/throttled and warm states (same 90s clip: 162–170s cold vs 45.2s warm).
The early "~5×" claims were inflated by cold-cache morning runs.

**How to bench**: `uv run python3 benchmarks/bench_transcribe.py --snippets 600
--baseline-only` (add `ASR_BACKEND=ct2` to the env for the fallback side).
WER vs the cached CT2 baseline is the cross-backend quality metric — char
similarity collapses when decoder styles differ (fillers/punctuation).

## Machine (verified, not guessed)

- Base Apple M2 (**not** Pro/Max): 4 P-cores + 4 E-cores, 16 GB unified memory,
  one GPU shared by every accelerated stage
  (`sysctl -n machdep.cpu.brand_string`, `hw.perflevel*.logicalcpu`).
- Workload: WebEx calls, mostly/all English, files 90s–30min.

## Shipped — feat/optimize-performance (2026-11-06)

- **MPS for align + diarize** (earlier commit, before this log): diarize 7.7×
  faster on the full file, byte-identical output.
- **ASR_BACKEND=mlx** (commit 8ce4d46): mlx-whisper on the Apple GPU (fp16),
  audio pre-chunked with whisperx's own bundled Pyannote VAD
  (`merge_chunks(chunk_size=30)`), one `mlx_whisper.transcribe` call per chunk,
  timestamps shifted by chunk start.
  - Naive whole-file MLX **failed quality** (WER 0.464): whisper's fixed 30s
    window seek-skips speech and turbo repetition-loops ("I do, I do…").
    VAD chunking fixed it: WER 0.326 @90s, 0.178 @600s vs CT2 baseline.
  - Word-diff forensics: remaining gap is mostly CT2's own 76-word
    hallucination in a 5s tail + lateral filler/punctuation diffs. Second real
    file (ask-redesign-p1) word-count parity: MLX 306 vs CT2 298.
  - **Visible change**: MLX emits finer segments than CT2 (225 vs 127 cues on
    the 600s clip) — same words, shorter cues in md/vtt. Noted for the user.
- **Parallel diarization** (commit c224ef7): diarize depends only on the audio
  → runs on a 1-worker ThreadPoolExecutor overlapping ASR+align, joined before
  assign. 600s wall 143.4s → 113.5s (~20%). Each contended stage runs ~1.7×
  slower solo but net win. Worker never touches meta.json (checkpoint races).
- **Honest head-to-head (warm, back-to-back, same machine state)**:

  | clip | main (ct2, seq) | feat (mlx, ∥) | speedup |
  | ---- | --------------- | ------------- | ------- |
  | 90s  | 45.2s           | 25.4s         | 1.8×    |
  | 600s | 217.1s          | 113.5s        | 1.9×    |

  asr alone @600s: ct2 153.9s vs mlx 53.1s solo / 92.3s under diarize
  contention. Feat's align is slower than main's (19.1 vs 11.0) because MLX
  yields more segments/words to align. Feat RTF @600s: 0.19 (~5× real-time).

## Tried and rejected (do not retry)

- **fp16 autocast on diarize (MPS)**: 0.77× *slower* + output changed (45≠44
  turns). Rejected.
- **fp16 autocast on align (MPS)**: 0.85× slower + 0.18s word-timestamp drift.
  Rejected.
- **mlx_whisper beam_size**: `NotImplementedError` (greedy only). Temperature/
  compression-ratio fallback sweep on our audio: fallbacks never fire → greedy
  is fine.
- **HF_HUB_OFFLINE=1**: saves ~1–2s, risks silent failure on any new model
  download. Skip.
- **parakeet-mlx ASR**: different model family → quality is non-negotiable.
  Skip.
- **convert → wav instead of mp3**: ~1%, adds divergence from upstream whisperx
  behavior. Skip.
- **ASR_THREADS / batch_size sweeps on ct2**: no measurable effect (4 P-cores,
  memory-bandwidth-bound at int8). Dead end on this chip.

## Next ideas (untried, by expected value)

1. MPS allocator knob A/B: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for
   diarize+align (one quick bench).
2. Cut per-chunk overhead in the MLX path (mel + tokenizer per call) — asr solo
   RTF is already 0.09, so ceiling is small; only worth it if profiling shows
   Python overhead, not GPU time.
3. Segmentation parity option: merge adjacent same-speaker MLX segments if the
   225-vs-127 cue granularity difference bothers anyone.
4. Kernel-level work: premature per project guidance — exhaust API/library
   level first.
