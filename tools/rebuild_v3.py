"""Rebuild BM1 from v3's real content, on the reference's timeline.

Two mistakes are corrected here.

First: the previous post pass applied the ORIGINAL's shot timings to generated
footage whose internal timing had drifted, so the slant landed on the pour.
Every source window below is taken from v3's actual content, mapped by eye:

    0.0-2.2  front-on hold      2.2-3.4  turning edge
    3.4-5.4  BACK PANEL         5.4-7.0  pour          7.0-10.0  hero

Second: v3's back panel reads "HYALURON BODY LOTON" - hailuo-3 traces source
text no matter how explicitly it is forbidden. So the turn is cut at three-
quarters and v3's 3.4-5.4 window is never used. The defect is removed rather
than fixed, which is a normal choice in product films.

Where the reference has a slanted front close-up, v3 has the back panel. That
beat is therefore built from v3's hero section instead, cropped and rotated.

    python tools/rebuild_v3.py
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
SOURCE = JOB / "result.mp4"          # v3, 1440x2560
REFERENCE = ROOT / "BM1" / "reference.mp4"
OUT = JOB / "rebuild"

W, H, FPS = 720, 1280, 24

ZOOM_START, ZOOM_END = 1.00, 1.45    # opening push-in
SLANT_DEG = -9.0                     # negative leans left, as the reference does
SLANT_COVER = 1.30                   # scale-up so rotation cannot expose corners
ECU_ZOOM = 2.0                       # 1440->720 centre crop == 2x magnification

# (src_in, src_out, out_duration, move, label)
# out_duration comes from the reference's shot lengths; src windows come from
# v3's actual content. Where they differ the clip is retimed.
EDIT = [
    (0.00, 1.96, 1.96, "push_in", "s01 hero, push in"),
    (2.00, 3.50, 1.92, "retime",  "s02 turn, stopped at 3/4"),
    (7.30, 9.75, 2.45, "slant",   "s03 slanted close-up (from hero)"),
    (5.40, 7.36, 1.96, "plain",   "s04 pour"),
    (7.20, 9.02, 1.82, "plain",   "s05 hero"),
]


def filter_for(move: str, frames: int, src_dur: float, out_dur: float) -> str:
    if move == "push_in":
        z = f"{ZOOM_START}+{ZOOM_END - ZOOM_START}*on/{max(frames - 1, 1)}"
        return (f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s={W}x{H}:fps={FPS},setsar=1")

    if move == "retime":
        # Stretch the shortened turn back to the reference's shot length.
        factor = out_dur / max(src_dur, 0.01)
        return (f"setpts={factor:.5f}*PTS,scale={W}:{H},fps={FPS},setsar=1")

    if move == "slant":
        # Cover the rotation, lean it, then take a centre crop tight enough to
        # read as an extreme close-up.
        #
        # The crop must be a fraction of the SCALED frame, not a multiple of
        # the output size - cropping 1440x2560 out of an 1872x3328 frame is a
        # 1.3x zoom, not the 2x intended.
        return (f"scale=iw*{SLANT_COVER}:ih*{SLANT_COVER},"
                f"rotate={SLANT_DEG}*PI/180:c=none,"
                f"crop=iw/{ECU_ZOOM}:ih/{ECU_ZOOM},"
                f"scale={W}:{H},fps={FPS},setsar=1")

    return f"scale={W}:{H},fps={FPS},setsar=1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slant", type=float, default=SLANT_DEG)
    ap.add_argument("--ecu", type=float, default=ECU_ZOOM)
    args = ap.parse_args()
    globals()["SLANT_DEG"] = args.slant
    globals()["ECU_ZOOM"] = args.ecu

    if not SOURCE.exists():
        sys.exit(f"missing {SOURCE}")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    total = sum(e[2] for e in EDIT)
    print(f"\n  source {SOURCE.name} -> {W}x{H}, rebuilding {total:.2f}s\n")

    pieces = []
    for i, (a, b, dur, move, label) in enumerate(EDIT):
        dest = OUT / f"r{i}_{move}.mp4"
        vf = filter_for(move, max(int(round(dur * FPS)), 1), b - a, dur)
        # -ss and -t must precede -i so they bound the INPUT. After -i, -t caps
        # output duration and would trim the retimed clip straight back down.
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(SOURCE),
               "-vf", vf, "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", str(dest)]
        r = ffrun(cmd)
        if r.returncode != 0:
            sys.exit(f"{label} failed: {r.stderr[:400]}")
        got = probe(dest).duration
        print(f"    {move:8} src {a:5.2f}-{b:5.2f} -> {got:4.2f}s   {label}")
        pieces.append(dest)

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
