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

**Working checklist** lives at the bottom of this file (the one living
section — everything else stays append-only). Claim with `[~] → <who> (date)`
before starting an item; see it for what's done, open, and taken.

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

## Direction opened 2026-11-06 — diarization as streaming source separation

User prompt: borrow from live TTS / streaming voice models; rethink diarization
as source signal separation on a live stream. Lit scan grounded on: Sortformer
(NVIDIA 2025, simultaneous sorting+separation), NSD-MS2S (memory-aware
multi-speaker embeddings, CHiME-7 best), Nemotron-3-Diarization (100M open
weights, streaming, 14.7% DER), TS-VAD → OTS-VAD/EDA-TS-VAD (online target-
speaker VAD), Diart, SSND. Conclusions:

- **The inversion**: separate the mix into K per-speaker signals first → label
  = stream identity; clustering + word-assignment stages dissolve; overlapped
  speech (whisper transcribes only the dominant talker today) becomes
  recoverable — that's where the remaining WebEx WER lives.
- **Borrowed principles**: (1) causality w/ small lookahead beats global
  context (who-spoke-when is decidable frame-by-frame); (2) memory/enrollment
  instead of clustering — NSD-MS2S's online voice profiles are the same
  embedding trick zero-shot TTS uses for cloning; gives stable cross-file
  labels ("Aamir", not a fresh SPEAKER_00); (3) structure tasks need small
  models — 100M streaming diarization beats big systems; right budget for
  16 GB M2; (4) separation for WHO, mixture for WHAT — ASR on separated audio
  loses WER to artifacts, so whisper keeps the original mix as input.
- **Honest wall-clock take**: diarize is already hidden under ASR via
  PARALLEL_STAGES — this direction buys overlap-region WER, identity
  stability, and time-to-first-transcript (w/ streaming refactor), NOT total
  runtime. ASR stays the runtime lever.
- **Next experiments (ranked, small first)**:
  1. WebEx "record individual audio tracks" at capture = free exact
     separation. Check recording settings BEFORE building anything.
  2. Enrollment on the existing stack, no new deps: per-speaker embedding
     centroids from diarize.json turns → persist to disk → match on later
     files → stable speaker names across meetings.
  3. Offline eval of Nemotron-3-Diarization (or Sortformer) on an
     overlap-heavy 2-speaker clip. Gate: beats pyannote DER in overlap
     regions ONLY, then consider integration behind PARALLEL_STAGES.
  4. Full streaming producer-consumer refactor: DEFER — changes the batch
     CLI/API contract; latency-to-first-word is not a current requirement.
- **Do NOT** transcribe separated streams (artifacts hurt WER; the mixture
  stays the ASR input).

## 2026-11-06 — overlap probe (sizing the separation prize)

User constraint confirmed: input is ALWAYS a single mixed audio file — capture-
level separation (WebEx per-track recording) is moot. So separation, if it
happens, must happen in-pipeline.

Probed overlap share by running community-1's segmentation head directly
(`pipe._segmentation` → non-aggregated Inference output, shape
(W, 589, 3) = 10s windows, 1.0s step, 3 local-speaker channels; manual
overlap-add aggregation, threshold 0.5, "overlap" = mean expected active
speakers ≥ 1.5). Throwaway script, not shipped:

- s3-and-research.opus (1718s): speech 89.9%, overlap **7.2% of speech time**
  (6.5% of file), peak 2 simultaneous.
- ask-redesign-p1.opus (835s): speech 75.6%, overlap **0.36% of speech time**,
  peak 2.

Read: the overlap prize is real but modest and HIGHLY file-dependent (20×
between our two files; never >2 simultaneous talkers locally). Separation is a
marginal quality play — at most ~7% of speech time on chatty calls, ~0% on
disciplined ones — not a wall-clock play. Verdict: keep the eval gate (only
worth integrating if a separator beats pyannote DER in overlap regions and
costs nothing on the critical path); **enrollment (stable cross-file speaker
names) moves up as the highest-certainty win** — it helps every file
regardless of overlap.

Bonus internals: community-1 segmentation = 10s windows / 1.0s hop / 3 local
speakers; direct `pyannote/segmentation-3.0` load 401s on our token (conditions
not accepted for that repo — the community-1 sub-model works, no extra click
needed).

## 2026-11-06 — design decision: stable tags (not names); P(inf) future note

- Enrollment will use **stable tags across files** (SPEAKER_00, SPEAKER_01, …
  in order of first-ever appearance), not named entities. Names are diverse
  across the board (south asian, nordic, spanish, irish, …) — auto-naming is
  inappropriate at this stage. Stable tags are repeatable and give future
  improvements a deterministic check-layer to build on top of.
- **P(inf) — lowest priority "coolness factor", do NOT chase now**: a
  native-audio multimodal LLM (Gemma-3n-12B-style: understands audio signals
  directly, no encoder→text input mapping) that identifies repeated audio
  signals and does anaphora/entity resolution to map speaker tags to entities.
  Strictly future scope; recorded so it isn't lost.

## 2026-11-07 — SHIPPED: speaker enrollment (stable tags across files)

Implementation: `return_embeddings=True` on the whisperx diarize call returns
pyannote's own per-speaker wespeaker centroids (256-dim, zero extra inference).
Each file's local centroids are cosine-matched against `.speakers.json`
(flock-guarded, atomic replace, n-weighted running-mean update); matched voices
keep their global tag, new voices take `SPEAKER_{next_tag}` in first-appearance
order. `speaker_linking` joined the cache key (one-time invalidation, ASR-
backend precedent); cache hits return frozen tags by design. Bench runs set
`SPEAKER_LINKING=false` to keep the speaker column comparable.

**Calibration** (aamir-minsuk-1-1_0411.opus → aamir-minsuk-1on1-aug-20.opus,
same two speakers, one enrolled then the other linked):

- same-voice cross-file cosines: **0.833, 0.775**
- different-voice cosines: **0.460, 0.382**
- gap 0.315 → threshold set at **0.70** (`SPEAKER_LINK_THRESHOLD`): 0.075
  below the lowest positive, 0.24 above the highest negative, biased high
  because a missed match only costs one extra tag while a false merge corrupts
  two identities.
- file 2's diarizer had *swapped* the local labels; linking mapped each voice
  to its true global tag through the permutation — exactly the failure mode
  per-file labels have.

Linking cost: 0.00–0.01s per file (pure vector math, off the model path).

Verified: full-script cache hit returns frozen tags with the store
byte-identical; `SPEAKER_LINKING=false` writes turns-only diarize.json and
never touches the store; sequential (`PARALLEL_STAGES=false`) and parallel
diarize paths both link; 8-process enrollment hammer on a shared store →
unique tags, second wave re-links every voice at cos 1.000, `next_tag` stable,
store clean after 16 lock-guarded writes.

## 2026-11-08 — working checklist (the one LIVING section of this log)

Marker convention for parallel agents:

- `[x]` done and verified (outcome entry exists above/below with numbers)
- `[~]` **claimed / in progress** — append `→ <who> (date)` when you pick it up;
  do not start an item someone else has claimed
- `[ ]` open

Everything else in this file stays append-only; this section is edited in
place. When you complete an item, tick it AND append a dated outcome entry.

### Shipped (verified — do not redo)

- [x] MPS align + diarize (7.7× diarize, byte-identical output)
- [x] MLX ASR backend, VAD-chunked (WER-gated; naive whole-file failed)
- [x] Parallel diarization under PARALLEL_STAGES (~20% @600s)
- [x] Speaker enrollment — stable cross-file tags (calibrated thr 0.70)
- [x] Honest benchmarking protocol (back-to-back, same machine state)

### Speed — low-hanging fruit first

- [~] **A. MLX turbo 8-bit quant** — `MLX_MODEL=mlx-community/whisper-large-v3-turbo-8bit`,
  zero code, back-to-back bench + WER gate vs baseline. Expect ~1.1–1.3× asr.
  → claude (2026-11-08)
- [~] **B. MPS high-watermark A/B** — `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
  for diarize+align; one bench. Attacks the 39s GPU contention tax on ASR.
  → claude (2026-11-08)
- [ ] **C. Parakeet TDT v3 (mlx)** behind `ASR_BACKEND` — WER eval on BOTH
  eval files (s3-and-research + ask-redesign-p1) vs ct2 baseline. Biggest
  single asr lever if quality holds; different model family = gate is the WER.

### Speed — saturation probe (the gate before any runtime rewrite)

- [~] **D. Profile the MLX asr path**: GPU-busy fraction vs Python/runtime
  overhead (per-chunk mel+tokenizer+decode loop). py-spy / Instruments /
  time-to-first-token deltas. Decision rule: GPU-busy > ~80% → Python stack
  is saturated, runtime swap buys little; overhead-dominated → the cpp/rust
  track (E) has real headroom. asr solo RTF is 0.09 @600s — bound the ceiling
  BEFORE building anything. → claude (2026-11-08)

### Speed — compiled-runtime track (only if D says overhead-bound, or A–C saturated)

- [ ] **E1. whisper.cpp (GGML/Metal)** — same whisper family, C++ runtime,
  quantized gguf weights, long-form + VAD. Quality gate ≈ quantization only.
  The lowest-friction "compiled backend" A/B.
- [ ] **E2. ONNX Runtime, CoreML/Metal EP** — C++ runtime, python bindings;
  whisper ONNX exports exist. Medium toolchain cost.
- [ ] **E3. whisperkit / CoreML on the Neural Engine** — ANE is idle today
  (everything shares the GPU); would REDUCE contention rather than add to it.
  Swift toolchain cost.
- [ ] **E4. candle (Rust) / burn (Rust) whisper** — cleanest "rust backend"
  story, least mature quant/Metal support; try after E1.
  Gate for all of E: WER parity + back-to-back wall-clock + no heavy new
  toolchain burden on users of this repo.

### Speed — bounded/conditional (known ceilings, do not prioritize yet)

- [ ] **F. Merge-cues-before-align** (~5–7% e2e) — output-visible granularity
  change; needs explicit user OK.
- [ ] **G. Faster/streaming diarizer** (pyannote streaming, Sortformer-class)
  — graduates to TOP priority only if asr gets ~2× faster (diarize ~70s solo
  @600s becomes the binder; floor with current diarizer ≈ 90s). Gate: DER in
  overlap regions AND solo < 70s @600s.

### Quality

- [ ] **H. Hotwords / initial_prompt** for recurring product names + jargon —
  zero speed cost, whisper-native; micro-test WER on jargon-dense spans.
- [ ] **I. Overlap recovery / separation** — marginal by measurement (0.4–7.2%
  of speech, file-dependent); stays behind the DER gate.
- [ ] P(inf) native-audio entity linking — coolness factor only, do NOT chase.

### Ceiling math every agent should know (600s clip, warm)

Wall 113.5s = convert ~1.2 + asr 92.3 (contended; solo 53.1) + align 19.1 +
ε; diarize ~70s solo, fully hidden. Even a 2× asr only lands ~85–95s wall
because diarize becomes the binder. ASR backend work is capped at ~1.3× e2e
until G is done; G is worth ~10% e2e even today (contention tax).
