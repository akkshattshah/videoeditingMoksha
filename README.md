# vre — video reference engine

Decompiles a reference video into a structured **Blueprint**, then routes it to
the cheapest recompile path that can preserve its structure.

This repo currently contains the **decompiler and router** — the half of the
system with no model-quality risk. Recompile (Aleph / Seedance calls, QC gate,
Remotion assembly) is not built yet.

## Quick start

```bash
pip install -r requirements.txt

python vre.py doctor                  # what backends and keys are active
python tests/make_fixture.py          # build a video with known ground truth
python tests/test_decompile.py        # score the decompiler against it
python vre.py analyze <video.mp4>     # decompile + route a real video
```

## Commands

| Command | Purpose |
|---|---|
| `vre doctor` | Which analysis backends and API keys are live |
| `vre analyze <video>` | Decompile, write `blueprint.json`, print the route |
| `vre route <blueprint.json>` | Re-route without re-analysing |
| `vre models` | Video models with per-second cost and arena rank |
| `vre cost` | Unit economics and validation spend for a profile |

`vre cost --profile local` (default) or `--profile cloud`. Add
`--duration`, `--shots`, `--price` to re-run against other assumptions. Line
items marked `~` are estimates, not verified rates.

## Why not just edit every frame with an image model?

Because a video is 30 images per second that must agree with each other. Edit
each one independently and every frame is individually excellent while the
sequence boils.

`tools/flicker_test.py` measures this rather than asserting it. It extracts N
consecutive frames, sends each independently to an image model with the same
prompt, reassembles, and scores the result:

```bash
python tools/flicker_test.py clip.mp4 "change the jacket to red" --frames 10
python tools/flicker_test.py clip.mp4 "..." --dry-run     # plumbing only, $0
```

**Flicker ratio** = frame-to-frame change in the edited sequence ÷ the same
measure on the source. ~1.0 means the edit is as stable as the original;
above 4 means it does not survive motion. Writes `sidebyside.mp4` with the
original on the left.

This is the argument for Aleph in one number: the image model is better at the
*edit*, but only a video model keeps that edit coherent across time. The
pipeline uses both — an image model builds one approved anchor frame, and
Aleph propagates it.

## Cost profiles

**LOCAL** is the pilot profile: your own machine, no managed services, a free
vision model, the cheapest image model, 480p output, 10s references. It exists
to prove the machinery works, not to look good.

**CLOUD** is production: Modal workers, best-in-class models, 720p, managed
Postgres and object storage.

| | local | cloud |
|---|---|---|
| Fixed monthly | **$0** | $297 |
| Quality job (Aleph) | $3.23 | $10.03 |
| Budget job (Seedance 1.5 Pro) | **$0.55** | $4.70 |
| Validation spend | **$216** | $1,384 |

Three decisions do most of that work:

- **Free VLM.** 14 OpenRouter models carry a `:free` tier and five accept
  images, including `google/gemma-4-31b-it:free` at 262k context. Even the
  cheap paid tier is $0.0014 per understand pass — the $0.20 figure only
  applies if you route it through Claude Opus, which this job does not need.
- **480p.** Seedance is token-billed on `H * W * 24 / 1024`, so 480p costs 4.5x
  less than 1080p. Structural fidelity is just as measurable at 480p.
- **Short references.** A 10s clip with 4 cuts exercises every pipeline stage
  exactly as a 30s one does, at a third the cost.

Useful flags: `--threshold` (shot sensitivity, lower finds more cuts),
`--resolution`, `--no-frames`, `--no-motion`.

## How it works

```
ref.mp4 -> probe -> shots -> motion -> keyframes -> audio -> blueprint.json -> router
```

The **Blueprint** ([schema.py](src/vre/schema.py)) is the contract every other
module speaks. It records shot boundaries, camera moves, keyframe paths, the
beat grid, and the swappable **slots** a user fills in.

The **router** ([router.py](src/vre/router.py)) picks the path:

- **ALEPH** — one in-context edit call. Requires ≤30s, ≤10 cuts, ≤1080p.
- **PER_SHOT** — regenerate shot by shot. Slower and pricier, but unbounded.

### Backend registry

Every analysis stage declares backends best-first
([backends.py](src/vre/backends.py)). The pipeline picks the best available and
records the choice in the blueprint, so the same code runs on a CPU laptop now
and against GPU workers on Modal later without branching at the call site.

| Stage | Local (active) | Better (not installed) |
|---|---|---|
| shots | `opencv-content` 0.75 | `pyscenedetect`, `transnetv2` |
| motion | `opencv-lk` 0.70 | `cotracker` |
| audio | `numpy-flux` 0.62 | `librosa` |
| ocr | `vlm-openrouter` 0.75 | `paddleocr` |
| transcript | none | `whisperx` |
| masks | none | `sam2` |
| depth | none | `video-depth-anything` |

Every derived value carries a `Provenance` recording which backend produced it
and how confident to be. The QC gate needs that to know what to re-verify.

Shot boundaries are cross-checked against ffmpeg's independent `scene` filter —
agreement between two unrelated detectors is the cheapest confidence signal
available, and it sets the recorded confidence between 0.62 and 0.94.

## Current accuracy

Measured against `tests/fixtures/ref_6shot.mp4` (6 shots, known cut times, a
synthetic push-in and pan, 120 BPM click track):

```
PASS  shot count        6/6
PASS  cut placement     5/5 within 0.2s   (worst error 0.000s)
PASS  camera motion     6/6
PASS  bpm               123.0 vs 120.0
PASS  route             aleph
12/12 passed
```

Synthetic footage is the easy case. Real ad footage with dissolves, motion
blur, and fast cuts will score lower — that is what `transnetv2` is registered
for.

## Pricing

[pricing.py](src/vre/pricing.py) holds per-model rates verified **2026-08-10**.
Seedance tiers are derived from OpenRouter's published token formula
`tokens/sec = (H * W * 24) / 1024`, anchored on each model's quoted 480p floor,
which resolves to a clean $0.00700/1K tokens for Seedance 2.0.

Two facts that drive the cost model:

- **Per-clip minimums dominate.** Seedance bills a 4s floor, Kling 3s. A
  1.2-second shot in a fast-cut ad costs the same as a 4-second one, which is
  why `PER_SHOT` gets expensive on exactly the references editors most want to
  copy.
- **OpenRouter routes Seedance at half fal's rate** (direct to ByteDance's
  "Seed" provider), but exposes Kling audio-on only. For audio-off Kling, fal
  is 33% cheaper — hence the separate `KLING_*` and `FAL_KEY` slots in `.env`.

Rates move monthly. Re-verify before trusting these for billing.

## Known gaps

- `ALEPH_MAX_CUTS = 10` is **unverified**. Runway documents multi-shot
  propagation without committing to a number; the figure comes from
  third-party reporting. The router emits a warning above 4 cuts. Benchmarking
  this on real footage is the highest-value next test.
- No slot detection yet — that needs the VLM pass (Gemini for perception,
  Claude for structured authoring).
- Dissolve classification is a heuristic on the score curve, not a trained
  detector.
- Tempo estimation is CPU-only spectral flux; install `librosa` for materially
  better beat tracking on music with a weak downbeat.
