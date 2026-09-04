"""Demonstrate — and measure — why per-frame image editing fails on video.

Extracts N consecutive frames, sends each one independently to an image model
with the same edit prompt, reassembles them, and scores the result.

The output is a number, not a vibe. Flicker ratio compares frame-to-frame
change in the edited sequence against the same measure on the source:

    ratio ~1.0   the edit is as temporally stable as the original
    ratio  2-4   visible boiling
    ratio  >4    unusable

Usage:
    python tools/flicker_test.py clip.mp4 "change the jacket to bright red"
    python tools/flicker_test.py clip.mp4 "..." --frames 16 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre import config  # noqa: E402
from vre.probe import probe, run  # noqa: E402

# Image models that accept an image and return an edited image.
# Cost is per 1024px image, derived from OpenRouter per-output-token rates.
EDIT_MODELS = {
    "nano-banana": ("google/gemini-2.5-flash-image", 0.0387),
    "gemini-pro": ("google/gemini-3-pro-image", 0.1548),
    "flux-klein": ("black-forest-labs/flux.2-klein-4b", 0.0044),
    "seedream": ("bytedance-seed/seedream-4.5", 0.0124),
}
DEFAULT_MODEL = "nano-banana"

TIMEOUT = httpx.Timeout(180.0, connect=15.0)


# ── frame IO ─────────────────────────────────────────────────────────────────


def extract_frames(video: Path, start: float, count: int, out_dir: Path) -> list[Path]:
    """Pull `count` consecutive frames starting at `start` seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"could not open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

    paths: list[Path] = []
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        dest = out_dir / f"src_{i:03d}.jpg"
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(dest)
    cap.release()

    if not paths:
        sys.exit(f"no frames read at t={start}s — is the clip shorter than that?")
    return paths


def assemble(frames: list[Path], fps: float, out: Path) -> bool:
    """Reassemble a frame sequence into a video at the source frame rate."""
    if not frames:
        return False
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    listing = out.parent / f"{out.stem}_list.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'\nduration {1 / fps:.6f}"
                  for p in frames)
        + f"\nfile '{frames[-1].resolve().as_posix()}'\n",
        encoding="utf-8",
    )
    res = run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-vf", f"scale={w}:{h},fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out),
    ])
    listing.unlink(missing_ok=True)
    return res.returncode == 0


def side_by_side(left: Path, right: Path, out: Path) -> bool:
    res = run([
        "ffmpeg", "-v", "error", "-y", "-i", str(left), "-i", str(right),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out),
    ])
    return res.returncode == 0


# ── the edit call ────────────────────────────────────────────────────────────


MAX_ATTEMPTS = 3


def edit_frame(client: httpx.Client, path: Path, prompt: str,
               model_id: str) -> bytes | None:
    """One independent image edit. No knowledge of any other frame.

    That isolation is the entire point of the experiment - it is exactly what
    happens if you try to edit a video with an image model.

    Image models refuse or reply with text often enough that a bare call has a
    meaningful failure rate, so this retries. That failure rate is itself a
    finding: it is why the pipeline needs a QC gate rather than trusting a
    single generation.
    """
    b64 = base64.b64encode(path.read_bytes()).decode()
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "modalities": ["image", "text"],
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = client.post("/chat/completions", json=payload)
            if r.status_code == 429:
                time.sleep(2.0 * attempt)
                continue
            if r.status_code != 200:
                print(f"    {path.name}: HTTP {r.status_code} {r.text[:160]}")
                time.sleep(1.0 * attempt)
                continue
            msg = r.json()["choices"][0]["message"]
        except Exception as exc:
            print(f"    {path.name}: {type(exc).__name__} {str(exc)[:120]}")
            time.sleep(1.0 * attempt)
            continue

        images = msg.get("images") or []
        if images:
            url = images[0].get("image_url", {}).get("url", "")
            if "," in url:
                return base64.b64decode(url.split(",", 1)[1])

        # No image came back. The text reply usually says why.
        said = (msg.get("content") or "").strip().replace("\n", " ")
        print(f"    {path.name}: attempt {attempt}/{MAX_ATTEMPTS} no image"
              + (f" - model said: {said[:110]!r}" if said else ""))
        time.sleep(1.0 * attempt)

    return None


def edit_all(frames: list[Path], prompt: str, model_id: str,
             out_dir: Path, workers: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = config.require("OPENROUTER_API_KEY")
    ref = cv2.imread(str(frames[0]))
    th, tw = ref.shape[:2]

    results: list[Path | None] = [None] * len(frames)

    with httpx.Client(
        base_url=config.OPENROUTER_BASE,
        headers={"Authorization": f"Bearer {key}",
                 "HTTP-Referer": "https://localhost/vre",
                 "X-Title": "vre flicker test"},
        timeout=TIMEOUT,
    ) as client:

        def work(i_path):
            i, path = i_path
            data = edit_frame(client, path, prompt, model_id)
            if data is None:
                return i, None
            dest = out_dir / f"edit_{i:03d}.jpg"
            dest.write_bytes(data)
            # Models often return a different aspect; normalise so the
            # comparison and the video assembly stay honest.
            img = cv2.imread(str(dest))
            if img is None:
                return i, None
            if img.shape[:2] != (th, tw):
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"    frame {i:03d} ok")
            return i, dest

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, dest in pool.map(work, list(enumerate(frames))):
                results[i] = dest

    return [p for p in results if p is not None]


# ── the measurement ──────────────────────────────────────────────────────────


def mean_frame_delta(frames: list[Path]) -> float:
    """Average absolute luminance change between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    grays = []
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16))
    if len(grays) < 2:
        return 0.0
    deltas = [float(np.abs(grays[i + 1] - grays[i]).mean())
              for i in range(len(grays) - 1)]
    return float(np.mean(deltas))


def verdict(ratio: float) -> str:
    if ratio < 1.5:
        return "stable - the edit held together"
    if ratio < 2.5:
        return "mild boiling, visible on close inspection"
    if ratio < 4.0:
        return "clear flicker - unusable as-is"
    return "severe flicker - the edit does not survive motion"


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("prompt", nargs="?", default="change the jacket to bright red")
    ap.add_argument("--frames", "-n", type=int, default=10)
    ap.add_argument("--start", "-s", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, choices=list(EDIT_MODELS))
    ap.add_argument("--workers", "-w", type=int, default=3,
                    help="parallel requests; lower if you hit rate limits")
    ap.add_argument("--out", "-o", type=Path, default=Path("jobs/flicker"))
    ap.add_argument("--dry-run", action="store_true",
                    help="extract and reassemble only, no API calls, $0")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"not found: {args.video}")

    model_id, per_image = EDIT_MODELS[args.model]
    src = probe(args.video)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  source    {args.video.name}  "
          f"{src.width}x{src.height} @ {src.fps:.2f}fps  {src.duration:.1f}s")
    print(f"  frames    {args.frames} from t={args.start}s "
          f"({args.frames / src.fps:.2f}s of video)")

    frames = extract_frames(args.video, args.start, args.frames, out / "src")
    print(f"  extracted {len(frames)}")

    orig_mp4 = out / "original.mp4"
    assemble(frames, src.fps, orig_mp4)

    if args.dry_run:
        print(f"\n  dry run - plumbing works, no API calls made")
        print(f"  wrote {orig_mp4}\n")
        return 0

    est = len(frames) * per_image
    print(f"  model     {model_id}")
    print(f"  est cost  ${est:.3f}  ({len(frames)} x ${per_image:.4f})")
    print(f"  prompt    {args.prompt!r}\n")

    t0 = time.perf_counter()
    edited = edit_all(frames, args.prompt, model_id, out / "edit", args.workers)
    elapsed = time.perf_counter() - t0

    if len(edited) < 2:
        print(f"\n  only {len(edited)} frames came back - cannot score\n")
        return 1

    edit_mp4 = out / "edited.mp4"
    assemble(edited, src.fps, edit_mp4)

    both = out / "sidebyside.mp4"
    side_by_side(orig_mp4, edit_mp4, both)

    src_delta = mean_frame_delta(frames[:len(edited)])
    edit_delta = mean_frame_delta(edited)
    ratio = edit_delta / src_delta if src_delta > 0.01 else float("inf")

    print(f"\n  {len(edited)}/{len(frames)} frames edited in {elapsed:.0f}s"
          f"  (${len(edited) * per_image:.3f})")
    print("\n  FLICKER MEASUREMENT")
    print(f"    source  frame-to-frame delta   {src_delta:7.3f}")
    print(f"    edited  frame-to-frame delta   {edit_delta:7.3f}")
    print(f"    ratio                          {ratio:7.2f}x")
    print(f"    verdict  {verdict(ratio)}")
    print(f"\n  watch: {both}")
    print(f"         left = original, right = per-frame edited\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
