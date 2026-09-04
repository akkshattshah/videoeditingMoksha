"""Regenerate only the shots that failed, then re-cut the finished video.

The BM1 run swapped the product and background correctly but lost two camera
moves: the opening push-in and the slanted extreme close-up. Regenerating the
whole 10s again would cost $1.31 and risk losing beats that already work.

Instead: regenerate just those two shots from the ORIGINAL reference, and take
the other three segments from the result we already have. Splices land on cut
boundaries, where a small shift in look is invisible.

    python tools/patch_shots.py            # preflight, $0
    python tools/patch_shots.py --go       # ~$0.57
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre import providers, tunnel  # noqa: E402
from vre.probe import probe, run as ffrun  # noqa: E402

JOB = ROOT / "testbench" / "output" / "20260810_194720_h3"
ORIGINAL = JOB / "segment.mp4"          # the untouched reference segment
RESULT = JOB / "result.mp4"             # the good-but-drifted generation
ELEMENT = ROOT / "BM1" / "rm2.jpeg"
OUT = JOB / "patched"

MODEL = "minimax/hailuo-3"
RATE = 0.13
W, H, FPS = 1440, 2560, 24

BRIEF = (
    "Replace the blue Parachute SkinPure body lotion bottle with the Parachute "
    "Advansed Onion Enriched Coconut Hair Oil bottle shown in the reference "
    "image: a tall slim amber translucent bottle with a magenta flip cap, its "
    "label carrying the Parachute logo, the words ONION, Enriched Coconut Hair "
    "Oil, HAIR FALL CONTROL, the Arabic text and the coconut-and-onion artwork, "
    "all sharp and legible. Replace the blue background, blue light rays and "
    "water droplets with the soft pink gradient and glossy reflective surface "
    "from the reference image. Keep the camera movement, framing and pacing of "
    "this shot exactly as they are."
)

# hailuo-3 only accepts these output durations. A 1.96s shot cannot be
# regenerated on its own, so the smallest useful unit is a 5s window.
SUPPORTED_DURATIONS = list(range(5, 16))

# The rebuilt timeline. Each entry lands on an original shot boundary.
#   ("gen",    a, b)  -> regenerate original[a:b] through the model
#   ("result", a, b)  -> lift result[a:b] as-is
#
# Both missing moves (the push-in and the slanted ECU) live in the first 6s,
# so one 6s pass covers them. The pour and hero already work, so they are
# lifted from the existing result rather than paid for twice.
EDIT = [
    ("gen",    0.00, 6.00, "s01 push-in + s02 rotate + s03 slanted ECU"),
    ("result", 5.00, 9.11, "s04 pour + s05 hero"),
]


def trim(src: Path, dest: Path, a: float, b: float, scale: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease," \
         f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1" if scale else f"fps={FPS}"
    r = ffrun([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(src),
        "-t", f"{b - a:.3f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(dest),
    ])
    if r.returncode != 0:
        sys.exit(f"trim failed ({src.name} {a}-{b}): {r.stderr[:300]}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--budget", type=float, default=1.00)
    args = ap.parse_args()

    for p in (ORIGINAL, RESULT, ELEMENT):
        if not p.exists():
            sys.exit(f"missing: {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    gens = [(i, a, b, lbl) for i, (k, a, b, lbl) in enumerate(EDIT) if k == "gen"]
    secs = sum(b - a for _, a, b, _ in gens)
    est = round(secs * RATE, 4)

    print(f"\n  rebuilding {sum(b - a for _, a, b, _ in EDIT):.2f}s timeline")
    for k, a, b, lbl in EDIT:
        mark = "REGENERATE" if k == "gen" else "reuse     "
        print(f"    {mark}  {a:5.2f}-{b:5.2f}  ({b - a:.2f}s)  {lbl}")
    print(f"\n  regenerating {secs:.2f}s at ${RATE}/s  ->  ${est:.3f}")

    try:
        left = providers.remaining()
        print(f"  credits ${left:.2f}")
    except Exception:
        left = None
    if est > args.budget:
        sys.exit(f"  estimate ${est:.3f} over budget ${args.budget:.2f}")

    # Cut every reused piece now - free, and proves the splice before spending.
    pieces: dict[int, Path] = {}
    for i, (kind, a, b, _) in enumerate(EDIT):
        if kind == "result":
            pieces[i] = trim(RESULT, OUT / f"p{i}_reuse.mp4", a, b)

    stage = OUT / "upload"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ELEMENT, stage / ELEMENT.name)
    for i, a, b, _ in gens:
        trim(ORIGINAL, stage / f"shot{i}.mp4", a, b, scale=False)

    if not args.go:
        print(f"\n  [preflight] verifying tunnel ...")
        with tunnel.serve_public(stage) as base:
            print(f"    OK  {tunnel.url_for(base, ELEMENT.name)}")
        print(f"\n  PREFLIGHT PASSED - spent $0.00")
        print(f"  run:  python tools/patch_shots.py --go     (${est:.3f})\n")
        return 0

    before = left
    with tunnel.serve_public(stage) as base:
        img = [tunnel.url_for(base, ELEMENT.name)]

        def launch(g):
            i, a, b, lbl = g
            want = int(round(b - a))
            if want not in SUPPORTED_DURATIONS:
                want = min(SUPPORTED_DURATIONS, key=lambda d: (abs(d - want), d))
                print(f"    note: snapped duration to {want}s (model only "
                      f"accepts {SUPPORTED_DURATIONS[0]}-{SUPPORTED_DURATIONS[-1]}s)")
            job = providers.submit_video(
                MODEL, BRIEF,
                video_url=tunnel.url_for(base, f"shot{i}.mp4"),
                image_urls=img, duration=want,
            )
            print(f"    submitted {lbl} -> {job.id}")
            return i, job, lbl

        # Both shots run concurrently; one wall-clock wait instead of two.
        with ThreadPoolExecutor(max_workers=len(gens)) as pool:
            jobs = list(pool.map(launch, gens))

        def finish(item):
            i, job, lbl = item
            payload = providers.poll_video(job)
            urls = providers.video_urls(payload)
            if not urls:
                raise RuntimeError(f"{lbl}: no output url")
            raw = OUT / f"p{i}_raw.mp4"
            providers.download(urls[0], raw)
            print(f"    done {lbl}")
            return i, raw

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for i, raw in pool.map(finish, jobs):
                a, b = EDIT[i][1], EDIT[i][2]
                pieces[i] = trim(raw, OUT / f"p{i}_gen.mp4", 0.0, b - a)
        print(f"    both back in {time.time() - t0:.0f}s")

    listing = OUT / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{pieces[i].resolve().as_posix()}'"
                  for i in range(len(EDIT))), encoding="utf-8")
    silent = OUT / "_silent.mp4"
    r = ffrun(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-c", "copy", str(silent)])
    if r.returncode != 0:
        sys.exit(f"concat failed: {r.stderr[:300]}")

    # The rebuilt cut matches the original's timing, so the original audio syncs.
    final = OUT / "final.mp4"
    r = ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(silent),
               "-i", str(ORIGINAL), "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)])
    if r.returncode != 0:
        shutil.copy2(silent, final)

    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(ORIGINAL), "-i", str(final),
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(OUT / "sidebyside.mp4")])

    spent = None
    if before is not None:
        try:
            spent = round(before - providers.remaining(), 4)
        except Exception:
            pass

    s = probe(final)
    print(f"\n  final    {final}")
    print(f"           {s.width}x{s.height}  {s.duration:.2f}s  "
          f"audio={s.audio_codec or 'none'}")
    if spent is not None:
        print(f"  spent    ${spent:.3f}  (estimated ${est:.3f})")
    print(f"  compare  {OUT / 'sidebyside.mp4'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
