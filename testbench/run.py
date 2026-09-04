"""Testbench: reference video + replacement elements -> edited video + score.

SAFETY: preflight is the default. Nothing is charged unless you pass --go,
and --go refuses to start if the estimate exceeds --budget.

    python testbench/run.py                    # preflight, $0
    python testbench/run.py --go               # runs, respects --budget
    python testbench/run.py --go --model h3    # cheapest model

Drop your files in testbench/input/ first - see testbench/README.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre import fidelity, providers, tunnel  # noqa: E402
from vre.decompile import decompile, save  # noqa: E402
from vre.probe import probe, run as ffrun  # noqa: E402

IN = Path(__file__).parent / "input"
OUT = Path(__file__).parent / "output"
ELEMENTS = IN / "elements"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v"}

# $/second, verified 2026-08-10 (Runway credit table + OpenRouter model pages)
MODELS = {
    "aleph": ("runway/aleph-2", 0.28, "purpose-built in-context editor"),
    "seedance": ("bytedance/seedance-2.5", 0.20, "video editing + extension"),
    "h3": ("minimax/hailuo-3", 0.13, "instruction-guided editing, cheapest"),
}
ANCHOR_MODEL = ("google/gemini-2.5-flash-image", 0.0387)


def die(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def find_reference() -> Path:
    vids = [p for p in IN.iterdir() if p.suffix.lower() in VIDEO_EXT] if IN.is_dir() else []
    if not vids:
        die(f"no reference video in {IN}\n  Drop one .mp4 there and re-run.")
    if len(vids) > 1:
        die(f"{len(vids)} videos in {IN} - keep exactly one:\n    "
            + "\n    ".join(v.name for v in vids))
    return vids[0]


def find_elements() -> list[Path]:
    if not ELEMENTS.is_dir():
        return []
    return sorted(p for p in ELEMENTS.iterdir() if p.suffix.lower() in IMAGE_EXT)


def read_brief() -> str:
    f = IN / "brief.txt"
    if not f.exists():
        die(f"missing {f}\n  Write one line describing the change you want.")
    text = "\n".join(
        ln for ln in f.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ).strip()
    if not text:
        die(f"{f} is empty - describe the change you want.")
    return text


def trim(src: Path, dest: Path, start: float, seconds: float) -> Path:
    """Cut a short segment. Re-encodes so the cut lands on an exact frame."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = ffrun([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest),
    ])
    if res.returncode != 0 or not dest.exists():
        die(f"trim failed: {res.stderr[:300]}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually spend money")
    ap.add_argument("--model", "-m", default="h3", choices=list(MODELS))
    ap.add_argument("--budget", "-b", type=float, default=5.0,
                    help="hard cap in dollars; refuses to start above this")
    ap.add_argument("--seconds", "-n", type=float, default=5.0,
                    help="length of the test segment; 0 = whole video")
    ap.add_argument("--start", "-s", type=float, default=0.0)
    ap.add_argument("--anchor", action="store_true",
                    help="also generate an anchor frame with the new element")
    ap.add_argument("--resolution", "-r", default=None, help="e.g. 480p, 720p")
    args = ap.parse_args()

    model_id, rate, blurb = MODELS[args.model]

    print("\n" + "=" * 68)
    print(f"  VRE TESTBENCH        {'LIVE RUN' if args.go else 'PREFLIGHT (no spend)'}")
    print("=" * 68)

    # ── 1. inputs ────────────────────────────────────────────────────────────
    ref = find_reference()
    elements = find_elements()
    brief = read_brief()
    src = probe(ref)

    print(f"\n  reference   {ref.name}")
    print(f"              {src.width}x{src.height} @ {src.fps:.2f}fps  "
          f"{src.duration:.2f}s  audio={'yes' if src.has_audio else 'no'}")
    print(f"  elements    {len(elements)}: "
          + (", ".join(p.name for p in elements) if elements else "none"))
    print(f"  brief       {brief[:90]}{'...' if len(brief) > 90 else ''}")
    print(f"  model       {model_id}  (${rate}/s - {blurb})")

    seconds = src.duration if args.seconds <= 0 else min(args.seconds,
                                                         src.duration - args.start)
    if seconds <= 0.5:
        die(f"segment too short ({seconds:.2f}s) - check --start/--seconds")

    # ── 2. cost ──────────────────────────────────────────────────────────────
    video_cost = seconds * rate
    anchor_cost = ANCHOR_MODEL[1] * 2 if args.anchor else 0.0
    est = round(video_cost + anchor_cost, 4)

    print(f"\n  segment     {args.start:.1f}s -> {args.start + seconds:.1f}s  "
          f"({seconds:.1f}s)")
    print(f"  est cost    ${est:.3f}   (video ${video_cost:.3f}"
          + (f" + anchor ${anchor_cost:.3f}" if args.anchor else "") + ")")
    print(f"  budget cap  ${args.budget:.2f}")

    try:
        left = providers.remaining()
        print(f"  credits     ${left:.2f} available")
    except Exception as exc:
        left = None
        print(f"  credits     could not read ({type(exc).__name__})")

    if est > args.budget:
        die(f"estimate ${est:.3f} exceeds budget ${args.budget:.2f}. "
            f"Lower --seconds, pick a cheaper --model, or raise --budget.")
    if left is not None and est > left:
        die(f"estimate ${est:.3f} exceeds remaining credits ${left:.2f}.")

    # ── 3. decompile the reference (free) ────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = OUT / f"{stamp}_{args.model}"
    job.mkdir(parents=True, exist_ok=True)

    seg = trim(ref, job / "segment.mp4", args.start, seconds)
    print(f"\n  [1/5] decompiling reference segment ...")
    bp_ref = decompile(seg, job / "ref")
    save(bp_ref, job / "ref")
    print(f"        {len(bp_ref.shots)} shots, {bp_ref.cut_count} cuts, "
          f"{bp_ref.source.duration:.2f}s")
    for s in bp_ref.shots:
        print(f"          {s.id}  {s.t_in:5.2f}-{s.t_out:5.2f}  {s.camera.move.value}")

    # ── 4. preflight stops here ──────────────────────────────────────────────
    upload = job / "upload"
    upload.mkdir(exist_ok=True)
    shutil.copy2(seg, upload / "reference.mp4")
    for p in elements:
        shutil.copy2(p, upload / p.name)

    if not args.go:
        print(f"\n  [2/5] verifying public HTTPS tunnel (required, still free) ...")
        try:
            with tunnel.serve_public(upload) as base:
                print(f"        OK - {tunnel.url_for(base, 'reference.mp4')}")
        except Exception as exc:
            die(f"tunnel failed: {exc}")

        print("\n" + "-" * 68)
        print(f"  PREFLIGHT PASSED - spent $0.00")
        print(f"  Everything is wired. To run for real:")
        print(f"      python testbench/run.py --go --model {args.model} "
              f"--seconds {seconds:.0f}")
        print(f"  Estimated cost: ${est:.3f}")
        print("-" * 68 + "\n")
        return 0

    # ── 5. live run ──────────────────────────────────────────────────────────
    before = left
    with tunnel.serve_public(upload) as base:
        ref_url = tunnel.url_for(base, "reference.mp4")
        img_urls = [tunnel.url_for(base, p.name) for p in elements]

        print(f"\n  [2/5] submitting to {model_id} ...")
        t0 = time.time()
        try:
            jb = providers.submit_video(
                model_id, brief,
                video_url=ref_url,
                image_urls=img_urls,
                duration=int(round(seconds)),
                resolution=args.resolution,
            )
        except providers.ProviderError as exc:
            die(str(exc))
        print(f"        job {jb.id}  status={jb.status}")

        print(f"  [3/5] polling (this takes minutes) ...")
        seen = {"s": ""}

        def tick(status, _):
            if status != seen["s"]:
                seen["s"] = status
                print(f"        {time.time() - t0:6.0f}s  {status}")

        try:
            payload = providers.poll_video(jb, on_tick=tick)
        except providers.ProviderError as exc:
            die(str(exc))

        urls = providers.video_urls(payload)
        if not urls:
            die(f"no output URL in payload: {json.dumps(payload)[:400]}")
        result = providers.download(urls[0], job / "result.mp4")
        print(f"        done in {time.time() - t0:.0f}s -> {result.name}")

    # ── 6. score ─────────────────────────────────────────────────────────────
    print(f"\n  [4/5] decompiling output ...")
    bp_out = decompile(result, job / "out")
    save(bp_out, job / "out")
    print(f"        {len(bp_out.shots)} shots, {bp_out.cut_count} cuts, "
          f"{bp_out.source.duration:.2f}s")

    print(f"\n  [5/5] scoring")
    rep = fidelity.compare(bp_ref, bp_out)
    print()
    print(fidelity.render(rep))

    spent = None
    try:
        after = providers.remaining()
        if before is not None:
            spent = round(before - after, 4)
    except Exception:
        after = None

    side = job / "sidebyside.mp4"
    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(seg), "-i", str(result),
           "-filter_complex",
           "[0:v]scale=-2:640[a];[1:v]scale=-2:640[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(side)])

    (job / "report.md").write_text(
        f"# {model_id}\n\n"
        f"- reference: `{ref.name}` segment {args.start:.1f}-"
        f"{args.start + seconds:.1f}s\n"
        f"- brief: {brief}\n"
        f"- elements: {', '.join(p.name for p in elements) or 'none'}\n"
        f"- estimated ${est:.3f}"
        + (f", actual ${spent:.3f}" if spent is not None else "") + "\n\n"
        f"## Fidelity {rep.overall:.2f} - {rep.verdict}\n\n"
        + "\n".join(f"- **{s.name}** {s.value:.2f} - {s.detail}" for s in rep.scores)
        + ("\n\n" + "\n".join(f"- NOTE {n}" for n in rep.notes) if rep.notes else ""),
        encoding="utf-8",
    )

    print("\n" + "-" * 68)
    if spent is not None:
        print(f"  actual spend  ${spent:.3f}   (estimated ${est:.3f})")
    if after is not None:
        print(f"  credits left  ${after:.2f}")
    print(f"  output        {result}")
    print(f"  compare       {side}")
    print(f"  report        {job / 'report.md'}")
    print("-" * 68 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
