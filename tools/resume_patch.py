"""Resume the already-paid BM1 patch job and finish the splice.

The generation was charged at submit; a dropped connection during polling
killed the run before the download. This re-polls that same job id - no new
spend - then assembles the final cut.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre import providers  # noqa: E402
from vre.probe import probe, run as ffrun  # noqa: E402

JOB_ID = "xAqj32xfAdH5W29UXPoI"
JOB = ROOT / "testbench" / "output" / "20260810_194720_h3"
ORIGINAL = JOB / "segment.mp4"
RESULT = JOB / "result.mp4"
OUT = JOB / "patched"
W, H, FPS = 1440, 2560, 24

EDIT = [("gen", 0.00, 6.00), ("result", 5.00, 9.11)]


def norm(src: Path, dest: Path, a: float, b: float) -> Path:
    r = ffrun([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(src),
        "-t", f"{b - a:.3f}",
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(dest),
    ])
    if r.returncode != 0:
        sys.exit(f"trim failed: {r.stderr[:300]}")
    return dest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    job = providers.VideoJob(
        id=JOB_ID,
        polling_url=f"{providers.BASE}/videos/{JOB_ID}",
        status="pending",
    )

    print(f"\n  resuming job {JOB_ID} (already paid)")
    seen = {"s": ""}

    def tick(status, _):
        if status != seen["s"]:
            seen["s"] = status
            print(f"    status: {status}", flush=True)

    payload = providers.poll_video(job, on_tick=tick)
    urls = providers.video_urls(payload)
    if not urls:
        sys.exit(f"no output url: {payload}")

    raw = OUT / "p0_raw.mp4"
    providers.download(urls[0], raw)
    print(f"    downloaded {raw.name} ({raw.stat().st_size / 1e6:.1f} MB)")

    pieces = [
        norm(raw, OUT / "p0_gen.mp4", 0.0, EDIT[0][2] - EDIT[0][1]),
        norm(RESULT, OUT / "p1_reuse.mp4", EDIT[1][1], EDIT[1][2]),
    ]

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
               "-i", str(ORIGINAL), "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)])
    if r.returncode != 0:
        shutil.copy2(silent, final)

    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(ORIGINAL), "-i", str(final),
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(OUT / "sidebyside.mp4")])

    s = probe(final)
    print(f"\n  final   {final}")
    print(f"          {s.width}x{s.height}  {s.duration:.2f}s  "
          f"audio={s.audio_codec or 'none'}")
    print(f"  compare {OUT / 'sidebyside.mp4'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
