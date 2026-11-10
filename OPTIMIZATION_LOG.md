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

- [x] **A. MLX turbo quant** — REJECTED 2026-11-08 (see entry below): q4
  showed NO speed gain (compute-bound, not memory-bound) + WER regression;
  8-bit blocked by repo layout and moot for speed.
- [x] **B. MPS high-watermark A/B** — REJECTED 2026-11-08: no effect at full
  power (inside control spread on every stage); the throttled-state −11% was
  a mirage.
- [x] **C. Parakeet TDT v3 (mlx)** — REJECTED 2026-11-08 (see entry below):
  2.6× faster than turbo-MLX but WER fails the gate on BOTH eval files
  (0.294/0.241 vs turbo's 0.177/baseline-parity); beam-5 recovers neither
  quality (0.290) nor speed (slower than turbo).
- [x] **C2. Canary-1b service-class spike** — REJECTED 2026-11-10 (see entry
  below): canary-1b-flash 0.136, canary-1b-v2 0.156, parakeet-tdt-1.1b 0.144
  vs turbo-MLX's 0.088 (same harness, same VAD chunks, same reference).
  Leaderboard parity does not transfer to our accented WebEx audio at any
  tested size; the transducer family is closed locally.

### Speed — saturation probe (the gate before any runtime rewrite)

- [~] **D. Profile the MLX asr path**: GPU-busy fraction vs Python/runtime
  overhead (per-chunk mel+tokenizer+decode loop). py-spy / Instruments /
  time-to-first-token deltas. Decision rule: GPU-busy > ~80% → Python stack
  is saturated, runtime swap buys little; overhead-dominated → the cpp/rust
  track (E) has real headroom. asr solo RTF is 0.09 @600s — bound the ceiling
  BEFORE building anything. DONE 2026-11-08: ~85% GPU-bound → stack
  saturated, E-track closed (see entry below).

### Speed — compiled-runtime track — CLOSED by D (GPU-bound; harness rewrite
### ceiling ~5% e2e; revisit only bundled with an algorithm change)

- [x] **E1. whisper.cpp (GGML/Metal)** — same whisper family, C++ runtime,
  quantized gguf weights, long-form + VAD. Quality gate ≈ quantization only.
  The lowest-friction "compiled backend" A/B.
- [x] **E2. ONNX Runtime, CoreML/Metal EP** — C++ runtime, python bindings;
  whisper ONNX exports exist. Medium toolchain cost.
- [x] **E3. whisperkit / CoreML on the Neural Engine** — ANE is idle today
  (everything shares the GPU); would REDUCE contention rather than add to it.
  Swift toolchain cost.
- [x] **E4. candle (Rust) / burn (Rust) whisper** — cleanest "rust backend"
  story, least mature quant/Metal support; try after E1.
  Gate for all of E: WER parity + back-to-back wall-clock + no heavy new
  toolchain burden on users of this repo.

### Speed — bounded/conditional (known ceilings, do not prioritize yet)

- [ ] **F. Merge-cues-before-align** (~5–7% e2e) — output-visible granularity
  change; needs explicit user OK.
- [ ] **J. Whisper-native word timestamps (drop the align stage)** — MLX
  whisper's own DTW timestamps vs wav2vec2's; if accuracy holds, −20% wall
  (align is 86s/hour uncontended). QUALITY-GATED on timestamp accuracy, not
  WER. Cheap probe: compare word-time deltas on the 600s clip.
- [x] **G. Faster/streaming diarizer** — SHIPPED 2026-11-08 (see entry below):
  segmentation hop 2.0s (`DIARIZE_SEG_STEP`) → diarize 2.0× solo, e2e −22%
  @600s, 100% final word-speaker agreement, linking intact. Remaining
  follow-up: swap-in streaming/Sortformer diarizer only if the residual
  ~20s contention tax ever matters.

### Quality

- [ ] **H. Hotwords / initial_prompt** for recurring product names + jargon —
  zero speed cost, whisper-native; micro-test WER on jargon-dense spans.
- [ ] **I. Overlap recovery / separation** — marginal by measurement (0.4–7.2%
  of speech, file-dependent); stays behind the DER gate.
- [ ] P(inf) native-audio entity linking — coolness factor only, do NOT chase.

### Ceiling math every agent should know (post-G, 2026-11-10)

**600s clip, warm**: wall 92.7s = convert 1.5 + asr 72.1 (contended; solo
53.1) + align 18.6 + ε; diarize 48.1 contended / 24.1 solo under hop 2.0.
**Hour-scale (3151s file, measured)**: wall 442.5s = 7.1× real-time; GPU work
(asr 279 + diarize 127 + align 86 ≈ 492s solo) ≈ wall — the run is AT the
scheduling floor; one GPU time-slices parallel stages (~11% true concurrency).
Wall only moves via less GPU work (C2/J/F/Sortformer) or more silicon.
Turbo-quality ASR on this chip = 11.3× RT solo; "hour in 1 min" (60× RT)
needs the transducer model class to pass our WER gate (C2) — runtime work is
measured-dead (D/E). Cloud APIs would hit 60× but are REJECTED: **audio must
never leave the machine** (settled 2026-11-10, hosting/privacy constraint).

## 2026-11-08 — A/B/D session 1: low-power trap, q4 data point, cache-key fix

**Machine-state lesson (extends the house rule):** this session's batch ran
with `lowpowermode 1` on battery — ALL GPU stages ~2.4× slower than the
logged warm state (600s clip: asr 241.7s vs 92.3, diarize 236.7 vs ~70,
total 274.8 vs 113.5; WER identical at 0.178 — pure throttle, no quality
change). **Check `pmset -g | grep lowpower` + battery before every bench.**
Timing numbers below from this session are low-power and NOT comparable to
earlier entries; relative A/B within the batch is still tentatively valid.

- **A (8-bit): blocked.** `mlx-community/whisper-large-v3-turbo-8bit` (and
  `-4bit`, `-asr-*`) now ship the new `model.safetensors` layout; our
  mlx-whisper 0.4.3 (PyPI latest!) only loads `weights.safetensors`/`npz` →
  ValueError at load. Path to 8-bit: git-install mlx-extras' mlx_whisper —
  deferred (unpinned dep in a user-facing repo) until quant upside is proven.
- **A (q4 pivot): runs, quality plausible, not certified.**
  `whisper-large-v3-turbo-q4` keeps the classic `weights.npz` layout. 600s:
  WER vs ct2 baseline 0.225 (fp16-mlx is 0.178), **WER vs fp16-mlx output
  0.136** (fp16-vs-ct2 is 0.178), coverage parity (same span, 482s vs 487s
  cued speech, word counts within 2.4%), no skips/loops. Char-sim collapse
  (0.101 vs baseline) is decoder-style drift, per this log's own warning.
  Spot-read: q4 even fixed a name ("Amir" vs ct2's "Ameria"). Verdict:
  acceptable as a SPEED experiment if the timing win is large; not adopted
  without either 8-bit working or a ground-truth check. Timing pending
  full-power re-run.
- **Cache-key fix (committed 806ce25):** `MLX_MODEL` overrides now join the
  cache key (as `mlx_model`), so a quantized repo never serves the fp16
  run's cached transcripts. Default (no override) keys unchanged — no
  invalidation for existing users.
- **B (watermark): ran but invalid.** Under throttle: total 245.2 vs 274.8
  (−11%), asr −13%, diarize −14% — suggestive, NOT trustworthy (throttled
  state changes contention dynamics; run-to-run variance unmeasured). Redo
  back-to-back at full power with a repeated control.
- **D: not started** (deferred — profile at full power to be representative).

Checklist markers: A/B/D stay `[~]` (claimed, in progress).

## 2026-11-08 — A/B/D session 2 (full power, AC, lowpowermode 0): three closures

Machine state verified before benching (`pmset -g`: AC attached, lowpowermode
0). Controls reproduced the logged warm state (fp16: 117.1 / 111.5s vs logged
113.5s; run-to-run spread ~5%).

**B REJECTED — watermark is a no-op.** fp16 controls 117.1/111.5 vs wm0
112.2 total; asr 96.6/90.9 vs 91.7; diarize 91.1/87.3 vs 87.9 — every stage
inside the control spread. The −11% seen under Low Power Mode was a
throttled-state artifact.

**A REJECTED — quantization buys nothing on this chip.** q4 (turbo, npz
layout): total 116.2 vs fp16 mean 114.3, asr 95.2 vs 93.8 — no gain, plus
WER 0.226 vs 0.177 and q4-vs-fp16 disagreement 0.136. Read: turbo decode is
**compute-bound, not memory-bound** on the base M2 → quant dequant overhead
cancels bandwidth savings. Also closes 8-bit (already layout-blocked; now
moot for speed).

**D DONE — the Python stack is saturated.** cProfile on the sequential 600s
run (asr solo 51.2s; cProfile overhead didn't move it → not python-bound).
Decomposition of the 45.2s in mlx_whisper.transcribe:
- `_main_loop` self 26.5s + KV-cache `apply` self 11.6s = **~84% native
  MLX dispatch/eval** (MLX is lazy: forward frames only build the graph,
  1.3s; GPU time lands where mx.eval forces it)
- VAD chunking 3.4s (serial inside the asr stage); model load 0.5s; our
  chunk loop ~0.02s
→ GPU-busy ≈ 85-90% of asr. A cpp/rust harness rewrite (whisper.cpp,
whisperkit, candle/burn) caps out at ~10-15% of asr ≈ **~5% end-to-end**.
E-track CLOSED unless bundled with an algorithm change. Bonus: per-step
KV-cache apply is ~25% of decode (4.3ms × 2690 steps) — an mlx-extras
library inefficiency, upstream territory.

**Where this leaves the checklist:** ASR speed on whisper-turbo is done —
fp16 MLX is the shape of this model on this chip. Remaining levers: **C
(parakeet TDT — algorithmic, fewer decode steps; WER-gated)** and **G
(diarize/streaming — the binder once ASR improves, ~10% e2e even today via
contention relief)**. Quality: H (hotwords) is cheap and untried.

## 2026-11-08 — C REJECTED: parakeet-tdt-0.6b-v3 fails the WER gate on both files

Spike via `parakeet-mlx 0.7.0` (pip-installed for the experiment; removed
after), model `mlx-community/parakeet-tdt-0.6b-v3` bf16, driven through our
exact VAD chunker (Pyannote, chunk_size=30), sequential/uncontended:

- **Speed is real**: s3-600s asr solo 19.5–21.3s vs turbo-MLX 53.1s (**2.6×**);
  ask-redesign-p1 (835s) 26.0s. Warm model load 0.8s.
- **Quality fails, on both files**:
  - s3-and-research-600s: WER vs ct2 baseline **0.294** (turbo-MLX 0.177).
  - ask-redesign-p1 full file: WER **0.241** (turbo ≈ baseline parity here).
  - Forensics: not just style drift (ITN "five percent" vs "5%" inflates the
    number a little) — real dropped sentences ("do you want this to be in
    english…"), garbles ("salary right okay it's really strictly confirmed" →
    "summary. okay. actually, confirm"). It also fixes some turbo errors
    ("glimps"→"glitch"), i.e. a different error profile, not a better one.
- **Beam-5 doesn't rescue it**: WER 0.290 AND 62.6s — slower than turbo's
  greedy. The gap is the model on accented WebEx meeting audio, not the
  decoder.

Quality is non-negotiable → C closed. The rejected-forever entry from
2026-11-06 ("different model family → skip") was right for the wrong reason;
now it's right for a measured one.

**Consequence: the ASR speed story is COMPLETE.** A (quant), B (watermark),
C (algorithm swap), E (compiled runtimes) all dead; fp16 MLX turbo is the
ceiling for this chip. Remaining checklist levers: G (faster/streaming
diarize — ~10% e2e via contention relief, binder once ASR improves) and
H (hotwords quality). F (cue merge) needs user sign-off on output granularity.

## 2026-11-08 — G SHIPPED: diarize hop 2.0s halves the stage, e2e −22% @600s, zero quality cost

**Diagnosis first (why diarize cost ~48s solo / ~70-91s contended @600s).**
Timing pyannote's internals via whisperx's progress hook on the 600s clip:
segmentation 2.6s / **embeddings 45.2s (94%)** / VBx clustering 0.4s. The
embedding stage runs one 10s crop per *local speaker* per *segmentation
window*; at the default 1.0s hop (90% window overlap) that's ~590 windows ×
~2 speakers ≈ 1180 crops ≈ **20× the audio duration**. The hop, not the
model, was the cost.

**The lever.** pyannote's `Inference` bakes `step` at `__init__` (setting
`pipeline.segmentation_step` post-hoc is a silent no-op — verified), so the
shipped code patches the built object: `diarize_pipeline.model._segmentation.step
= DIARIZE_SEG_STEP` (default 2.0; 1.0 restores upstream exactly). Added to the
cache key (`diarize_seg_step`) — one-time invalidation.

**Solo hop A/B (600s clip, uncontended, same model):**

| hop | diarize solo | turns | frame agreement vs 1.0 |
| --- | --- | --- | --- |
| 1.0 (upstream) | 48.2s | 219 | — |
| 2.0 (shipped) | 24.1s | 229 | 98.35% |
| 2.5 (rejected) | 20.9s | 217 | 97.25% |

2.0 is the knee: 2.5 gives 13% more speed for 1.1pp agreement.

**End-to-end bench (back-to-back, AC power, lowpowermode 0, 600s clip):**

| | hop 2.0 | hop 1.0 | Δ |
| --- | --- | --- | --- |
| total | **92.7s** | 119.5s | **−22.4%** |
| diarize (contended) | 48.1 | 91.4 | 1.90× |
| asr (contended) | 72.1 | 96.9 | −26% (contention relief) |
| WER | 0.177 | 0.178 | — |

**Quality: the 1.6% raw-turn drift washes out completely.** Final VTT
word-speaker agreement between the two runs: **100.00%** (225 cues each, every
both-labeled 10ms frame identical) — assignment maps speakers per *word*, and
the boundary shifts don't move a single word across a speaker change.

**Speaker linking intact under hop 2.0** (centroids come from hop-2.0 crops
now): fresh no-resume run of `aamir-minsuk-1on1-aug-20.opus` vs the
hop-1.0-blended store — positives 0.942 / 0.957, negatives 0.475 / 0.532;
correct one-to-one remap at threshold 0.70 with a wide margin. (Local tag
order flipped vs the store under the new hop; linking fixed it — that's its
job.)

**Where this leaves the pie @600s:** wall 92.7s = convert 1.5 + asr 72.1
contended (solo 53.1) + align 18.6 + ε. The residual contention tax is
~19s. Next speed levers, in order: F (cue merge, needs user OK on output
granularity, also −5-7%) and a streaming/Sortformer diarizer swap (new
dependency; only worth it if the remaining tax matters). Quality: H
(hotwords) still cheap and untried.

## 2026-11-10 — hour-scale run, the 9× debate, strategy realignment (cloud OFF, C2 ON)

**Real 52.5-min file (aamir-kelwin-video-ingestion-sync.opus, 3151s)**:
wall 442.5s = **7.1× real-time** (convert 7.6, asr 347 contended / ~279 solo,
diarize 218 contended / ~127 solo, align 86.1). Speaker linking worked in the
wild: one voice matched at 0.919, the other correctly allocated SPEAKER_02
(new person Kelwin). 1028 segments, ~1000 of whisperx's "failed to align
(empty segment)" warnings on backchannels — cosmetic, not a regression.

**The GPU-work identity (why scheduling is done):** asr 279 + diarize 127 +
align 86 ≈ 492s of serialized GPU compute finished in 442.5s wall — the one
GPU gives ~11% true concurrency beyond time-slicing. PARALLEL_STAGES can't be
tuned further; wall from here = less GPU work or more silicon.

**User target: 1 hour in 1 min (≈9×, 60× RT).** How commercial services do
sub-minute synchronous (AAI Universal-2 ~22s/hr; Deepgram nova-3 ~600× RT
batch): they don't run whisper — single-pass/small-decoder
transducer-conformer models (~10× less decode compute, quality held by
training scale) on A100/H100 fleets (~10-30× base-M2 GPU) with chunk fan-out.
Open-weights equivalents: nvidia canary-1b-flash (883M, >1000 RTFx on A100,
leaderboard WER 6.35% vs whisper-large-v3 7.44%); canary-1b-v2 ("outperforms
whisper-large-v3 on English, 10× faster", arXiv 2509.14128).

**Key nuance reopening the algorithm track:** parakeet-tdt-0.6b-v3 BEATS
whisper-large-v3 on the Open ASR leaderboard (6.32% vs 7.44%) yet failed our
WER gate badly (0.294 vs turbo's 0.177) → our accented WebEx audio is
adversarial for the family, OR 0.6b (the family's weakest member) is just too
small for hard meeting audio. C2 = test the class at proper size (canary-1b
flash + v2) through the same WER harness. Honest prior: 50/50 — canary
already underperforms whisper on accented speech elsewhere (Quebec French
benchmarks), and our speakers are exactly that profile.

**Decision (settled with user):** cloud API backend is REJECTED — the audio
must never leave the machine; this is a self-hosted service. All future
speed work is local-only. Remaining levers: C2 (canary spike), J (native
word timestamps), F (cue merge, needs user OK), Sortformer diarizer swap,
and the quality items H/I.

## 2026-11-10 — C2 REJECTED: the transducer family fails our WER gate at every tested size

Spike setup (throwaway envs in /tmp, nothing shipped): NeMo for canary
(`nemo_toolkit[asr]`, torch CPU — **pynini, the usual macOS blocker, turned
out unnecessary** for canary ASR inference), parakeet-mlx 0.7.0 for TDT.
All candidates driven through the pipeline's EXACT chunker (whisperx
Pyannote VAD, 30s chunks; 23 chunks / 585s speech for the 600s eval file).
One WER harness for everyone — whisper-normalizer (leaderboard-style EN
normalization) + word-levenshtein vs the ct2 full-file baseline
(`Output/s3-and-research.vtt`, window-filtered to 600s):

| model | WER @600s | decode time @600s |
| --- | --- | --- |
| **turbo-MLX fp16 (control)** | **0.088** | 53.1s GPU (11.3× RT) |
| canary-1b-flash (883M) | 0.136 | ~3× RT CPU (5s→1.6s probe) |
| parakeet-tdt-1.1b (MLX, bf16) | 0.144 | 23.0s GPU (25× RT) |
| canary-1b-v2 (1B) | 0.156 | CPU, slower than flash |
| parakeet-tdt-0.6b-v3 (prior spike) | ~0.29 (old harness) | 19.5–21.3s GPU |

(These absolutes use the leaderboard normalizer — the in-repo bench's 0.177
turbo figure is its own metric; only same-harness comparisons gate.)

- **Quality verdict**: every candidate ≥ 1.55× turbo's error. Forensics
  (flash vs turbo, 250 diffs/1757 words): a DIFFERENT error profile, not a
  better one — both garble the name ("amir"/"amiria"), real drops and clause
  insertions on both sides, filler style drift. Same shape as the 0.6b
  parakeet finding.
- **Family size curve on our audio**: parakeet 0.6b ~0.29 → 1.1b 0.14;
  canary-1b 0.14–0.16. Bigger halved the error but turbo is still 1.6×+
  ahead. Extrapolated parakeet-tdt-2b-v3 ≈ 0.10–0.12 — probably still short,
  NeMo-CPU-only (no MLX port; 2b on CPU ≈ 20-40 min/600s) → NOT chased. If
  someone ports 2b-v3 to MLX, revisit with the same harness.
- **Speed is moot anyway**: parakeet-1.1b decodes 25× RT on the GPU (would be
  fast enough), but canary on CPU is ~3× RT — a regression vs turbo-MLX; no
  MLX canary exists.
- **Why the services still win**: they run this model class on A100/H100
  fleets (canary-flash: >1000 RTFx there vs 3× RT on our CPU) AND their WER
  edge on leaderboards doesn't transfer to our audio. Local + this chip +
  this quality bar = turbo-MLX stays.

**Consequence: the 60×-RT ("hour in 1 min") target is now measured-dead on
this chip for open-weights ASR.** The answer to that target is hardware
(faster Apple silicon / discrete GPU running the local stack), not
algorithms. Remaining local levers: J (native word timestamps, ~−20% wall,
quality-gated), F (cue merge, needs user OK), Sortformer diarizer (residual
contention), H (hotwords, quality).
