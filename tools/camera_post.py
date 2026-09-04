"""Apply the reference's camera moves in post instead of asking the model.

Six image generations and three video runs all proved the same thing: these
models render the product and background well and refuse to honour framing.
They reset to a centred packshot every time.

But a push-in is a crop over time and a slant is a rotation. ffmpeg does both
exactly, for free. So the model supplies the product; ffmpeg supplies the
camera.

Source is 1440x2560, output is 720x1280 - a 2x crop still lands at the
reference's native resolution, so tightening the frame costs no detail.

    python tools/camera_post.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre.probe import probe, run as ffrun  # noqa: E402

JOB = ROOT / "testbench" / "output" / "20260810_211831_h3"   # v3
SOURCE = JOB / "result.mp4"                # good product + background
REFERENCE = JOB / "segment.mp4"            # for audio and side-by-side
OUT = JOB / "camera"

W, H, FPS = 720, 1280, 24

# (start, end, move, label) - timings are the reference's own shot boundaries.
SHOTS = [
    (0.00, 1.96, "push_in",  "s01 hero, push in"),
    (1.96, 3.88, "hold",     "s02 rotate to back"),
    (3.88, 6.33, "slant_in", "s03 slanted extreme close-up"),
    (6.33, 8.29, "hold",     "s04 pour"),
    (8.29, 10.03, "hold",    "s05 hero"),
]

# How hard each move pushes. Tuned against the reference.
ZOOM_START, ZOOM_END = 1.00, 1.45   # s01 push-in
SLANT_DEG = -9.0                    # s03 lean (negative = leans left, as the reference does)
SLANT_ZOOM = 1.30                   # must exceed the rotation's corner loss


def filter_for(move: str, frames: int) -> str:
    """Build the ffmpeg filter chain for one camera move."""
    if move == "push_in":
        # zoompan drives the crop window; `on` is the output frame index.
        z = f"{ZOOM_START}+{ZOOM_END - ZOOM_START}*on/{max(frames - 1, 1)}"
        return (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s={W}x{H}:fps={FPS},setsar=1")

    if move == "slant_in":
        # Scale up first so rotating cannot pull empty corners into frame,
        # then rotate, then crop back to the target window.
        return (f"scale=iw*{SLANT_ZOOM}:ih*{SLANT_ZOOM},"
                f"rotate={SLANT_DEG}*PI/180:c=none,"
                f"crop={W}*{SLANT_ZOOM}:{H}*{SLANT_ZOOM},"
                f"scale={W}:{H},fps={FPS},setsar=1")

    return f"scale={W}:{H},fps={FPS},setsar=1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slant", type=float, default=SLANT_DEG)
    ap.add_argument("--zoom", type=float, default=ZOOM_END)
    args = ap.parse_args()

    globals()["SLANT_DEG"] = args.slant
    globals()["ZOOM_END"] = args.zoom

    if not SOURCE.exists():
        sys.exit(f"missing source: {SOURCE}")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n  source {SOURCE.name}  ->  {W}x{H}\n")
    pieces = []
    for i, (a, b, move, label) in enumerate(SHOTS):
        dest = OUT / f"c{i}_{move}.mp4"
        frames = max(int(round((b - a) * FPS)), 1)
        vf = filter_for(move, frames)
        r = ffrun([
            "ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(SOURCE),
            "-t", f"{b - a:.3f}", "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(dest),
        ])
        if r.returncode != 0:
            sys.exit(f"{label} failed: {r.stderr[:400]}")
        pieces.append(dest)
        print(f"    {move:9} {a:5.2f}-{b:5.2f}  {label}")

    listing = OUT / "concat.txt"
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'"
                                 for p in pieces), encoding="utf-8")
    silent = OUT / "_silent.mp4"
    r = ffrun(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-c", "copy", str(silent)])
    if r.returncode != 0:
        sys.exit(f"concat failed: {r.stderr[:300]}")

    final = OUT / "final.mp4"
    r = ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(silent),
               "-i", str(REFERENCE), "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)])
    if r.returncode != 0:
        shutil.copy2(silent, final)

    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(REFERENCE), "-i", str(final),
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(OUT / "sidebyside.mp4")])

    s = probe(final)
    print(f"\n  final    {final}")
    print(f"           {s.width}x{s.height}  {s.duration:.2f}s  "
          f"audio={s.audio_codec or 'none'}   cost $0.00")
    print(f"  compare  {OUT / 'sidebyside.mp4'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
