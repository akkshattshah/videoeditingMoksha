"""Re-cut generated footage onto the reference's timeline.

The model does not honour the reference's shot timings - v3's back panel runs
twice as long as it should and the slanted close-up never happens at all. So
every target beat has to name where it is sourced from, rather than assuming
the generated clip shares the reference's clock.

The slanted close-up is built from the closing hero footage: crop 2x, tilt.
The model never produced that framing and never will, but the footage it did
produce contains everything the crop needs.

    python tools/recut.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre.probe import probe, run as ffrun  # noqa: E402

JOB = ROOT / "testbench" / "output" / "20260810_211831_h3"
SRC = JOB / "result.mp4"          # v3, 1440x2560
REF = JOB / "segment.mp4"         # for audio + side-by-side
OUT = JOB / "recut"

W, H, FPS = 720, 1280, 24

# target_start, target_end, source_start, move, label
#
# Source times are where the beat actually lives in v3, mapped by eye from the
# 0.5s timeline strip. Target times are the reference's own shot boundaries.
BEATS = [
    (0.00, 1.96, 0.00, "push_in", "s01 hero, push in"),
    (1.96, 3.88, 2.00, "hold",    "s02 turn to back panel"),
    (3.88, 6.33, 7.55, "slant",   "s03 slanted ECU (built from hero footage)"),
    (6.33, 8.29, 5.50, "hold",    "s04 pour"),
    (8.29, 10.03, 8.26, "hold",   "s05 hero"),
]

ZOOM_START, ZOOM_END = 1.00, 1.45
SLANT_DEG = -9.0      # negative = leans left, matching the reference
SLANT_CROP = 3.2      # how far into the frame the close-up pushes
Y_BIAS = 0.42         # <0.5 frames above centre, onto the label


def vf_for(move: str, frames: int) -> str:
    if move == "push_in":
        z = f"{ZOOM_START}+{ZOOM_END - ZOOM_START}*on/{max(frames - 1, 1)}"
        return (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s={W}x{H}:fps={FPS},setsar=1")

    if move == "slant":
        # Crop hard into the label, then tilt. Rotating first would drag empty
        # corners in, so crop wide, rotate, crop again.
        #
        # The source is a wide hero shot with the whole bottle centred; the
        # reference's close-up frames the label alone, so the crop has to be
        # aggressive and biased above centre to land on the branding rather
        # than the base.
        cw = int(W * 2 * (2.0 / SLANT_CROP))
        ch = int(H * 2 * (2.0 / SLANT_CROP))
        return (f"crop={cw}:{ch}:(iw-{cw})/2:(ih-{ch})*{Y_BIAS},"
                f"rotate={SLANT_DEG}*PI/180:c=0xE9A0A8,"
                f"crop={int(cw * 0.68)}:{int(ch * 0.68)},"
                f"scale={W}:{H},fps={FPS},setsar=1")

    return f"scale={W}:{H},fps={FPS},setsar=1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slant", type=float, default=SLANT_DEG)
    ap.add_argument("--zoom", type=float, default=ZOOM_END)
    ap.add_argument("--crop", type=float, default=SLANT_CROP)
    ap.add_argument("--ybias", type=float, default=Y_BIAS)
    a = ap.parse_args()
    globals().update(SLANT_DEG=a.slant, ZOOM_END=a.zoom,
                     SLANT_CROP=a.crop, Y_BIAS=a.ybias)

    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n  source {SRC.name} -> {W}x{H}\n")
    pieces = []
    for i, (t0, t1, s0, move, label) in enumerate(BEATS):
        dur = t1 - t0
        dest = OUT / f"b{i}_{move}.mp4"
        r = ffrun([
            "ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.3f}", "-i", str(SRC),
            "-t", f"{dur:.3f}", "-vf", vf_for(move, max(int(dur * FPS), 1)),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(dest),
        ])
        if r.returncode != 0:
            sys.exit(f"{label} failed: {r.stderr[:400]}")
        pieces.append(dest)
        print(f"    {move:8} target {t0:5.2f}-{t1:5.2f}  <- v3 {s0:5.2f}  {label}")

    listing = OUT / "concat.txt"
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'"
                                 for p in pieces), encoding="utf-8")
    silent = OUT / "_silent.mp4"
    if ffrun(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", str(listing), "-c", "copy", str(silent)]).returncode != 0:
        sys.exit("concat failed")

    final = OUT / "final.mp4"
    if ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(REF),
              "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
              "-shortest", str(final)]).returncode != 0:
        shutil.copy2(silent, final)

    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(REF), "-i", str(final),
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(OUT / "sidebyside.mp4")])

    s = probe(final)
    print(f"\n  final   {final}")
    print(f"          {s.width}x{s.height}  {s.duration:.2f}s  cost $0.00")
    print(f"  compare {OUT / 'sidebyside.mp4'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
