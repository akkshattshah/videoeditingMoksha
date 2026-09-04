"""Generate a synthetic reference video with known ground truth.

Six visually distinct shots at exact durations, two deliberate camera moves,
and a 120 BPM click track. Lets the decompiler be measured against truth
rather than inspected by eye.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

W, H, FPS = 360, 640, 30  # vertical, like the ad format this targets

LAVFI = "lavfi"
TEXTURE = "texture"

# (label, duration, (kind, source), camera move)
#
# Camera-motion segments must use a *static, feature-rich* source. Two traps
# worth recording: animated sources (testsrc2, mandelbrot) make optical flow
# track the content instead of the camera, and striped sources (smptebars)
# have no 2-D corners at all, so Shi-Tomasi finds nothing to track. Blurred
# noise avoids both.
SEGMENTS = [
    ("red",   2.0, (LAVFI, f"color=c=0xC0392B:s={W}x{H}"), "static"),
    ("bars",  3.5, (LAVFI, f"smptebars=s={W}x{H}"),        "static"),
    ("blue",  1.5, (LAVFI, f"color=c=0x1F4E79:s={W}x{H}"), "static"),
    ("zoom",  4.0, (TEXTURE, None),                        "push_in"),
    ("green", 2.5, (LAVFI, f"color=c=0x1E8449:s={W}x{H}"), "static"),
    ("pan",   3.0, (TEXTURE, None),                        "pan_right"),
]

BPM = 120.0
# The comma inside mod() must be escaped or ffmpeg reads it as an option break.
CLICK = f"0.7*sin(2*PI*880*t)*exp(-14*mod(t\\,{60.0 / BPM}))"

TEX_W, TEX_H = W * 2, H * 2


def truth() -> dict:
    cuts, t, shots = [], 0.0, []
    for label, dur, _, move in SEGMENTS:
        shots.append({"label": label, "t_in": round(t, 4),
                      "t_out": round(t + dur, 4), "camera": move})
        t += dur
        cuts.append(round(t, 4))
    return {
        "duration": round(t, 4),
        "fps": FPS,
        "width": W, "height": H,
        "bpm": BPM,
        "n_shots": len(SEGMENTS),
        "cuts": cuts[:-1],          # final boundary is EOF, not a cut
        "shots": shots,
    }


def _run(cmd: list[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"{what} failed:\n{res.stderr[:700]}")


def _make_texture(ffmpeg: str, dest: Path) -> None:
    """A static blurred-noise plate: dense 2-D corners, no animation."""
    _run(
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s={TEX_W}x{TEX_H}",
            "-vf", "geq=random(1)*255:128:128,boxblur=3:1,eq=contrast=1.6",
            "-frames:v", "1", str(dest),
        ],
        "texture",
    )


def build(out: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found on PATH")

    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "_segments"
    work.mkdir(exist_ok=True)

    texture = work / "texture.png"
    _make_texture(ffmpeg, texture)

    parts = []
    for i, (label, dur, (kind, src), move) in enumerate(SEGMENTS):
        seg = work / f"{i:02d}_{label}.mp4"
        frames = int(dur * FPS)

        if move == "push_in":
            # Continuous 1.0 -> 1.45 centred zoom. zoompan counts output
            # frames with `on`.
            vf = (
                f"zoompan=z='1+0.45*on/{frames}'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s={W}x{H}:fps={FPS},format=yuv420p"
            )
        elif move == "pan_right":
            # Slide a W-wide window left->right across the texture. Content
            # travels left in frame, which is what a rightward pan looks like.
            # crop uses `n` for frame number, not `on`.
            vf = (
                f"crop=w={TEX_W}:h={H}:x=0:y=0,"
                f"crop=w={W}:h={H}:x='(in_w-out_w)*n/{frames}':y=0,"
                f"fps={FPS},format=yuv420p"
            )
        else:
            vf = f"fps={FPS},format=yuv420p"

        if kind == TEXTURE:
            inputs = ["-loop", "1", "-framerate", str(FPS), "-i", str(texture)]
        else:
            inputs = ["-f", "lavfi", "-i", f"{src}:r={FPS}"]

        # Duration comes from -t, not a source option; `mandelbrot` and
        # friends reject `d=` outright.
        _run(
            [ffmpeg, "-v", "error", "-y", *inputs,
             "-vf", vf, "-t", str(dur),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(seg)],
            f"segment {label}",
        )
        parts.append(seg)

    listing = work / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts), encoding="utf-8"
    )

    total = sum(d for _, d, _, _ in SEGMENTS)
    silent = work / "_silent.mp4"
    _run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
          "-i", str(listing), "-c", "copy", str(silent)], "concat")

    _run([ffmpeg, "-v", "error", "-y",
          "-i", str(silent),
          "-f", "lavfi", "-i", f"aevalsrc={CLICK}:d={total}:s=44100",
          "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
          "-shortest", str(out)], "mux")

    shutil.rmtree(work, ignore_errors=True)

    gt = out.with_suffix(".truth.json")
    gt.write_text(json.dumps(truth(), indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/ref_6shot.mp4")
    built = build(dest)
    t = truth()
    print(f"built {built}  ({t['duration']}s, {t['n_shots']} shots)")
    print(f"truth {built.with_suffix('.truth.json')}")
    print(f"cuts  {t['cuts']}")
