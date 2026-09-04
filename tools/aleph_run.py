"""BM1 through Runway Aleph 2.0 - one careful pass.

Aleph is an in-context editor, not a generator. It preserves everything it is
not told to change, so the prompt is deliberately CHANGE-ONLY. Naming the
camera moves, cuts or pacing would be counterproductive: mentioning something
invites the model to touch it.

That is the opposite of hailuo-3, which regenerates the scene and had to be
told exactly when the bottle should turn. Three h3 passes proved that model
re-stages the choreography every time and traces the source's text no matter
how explicitly it is forbidden.

Verified free by probing the API:
  * input_references needs exactly one video_url; image_url may sit alongside
  * both must be HTTPS - data URIs are rejected
  * Aleph imposes no duration whitelist (h3 only allows 5-15s), so duration is
    omitted entirely and the edit matches the source length

    python tools/aleph_run.py          # preflight, $0
    python tools/aleph_run.py --go     # ~$2.83
"""

from __future__ import annotations

import argparse
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

REFERENCE = ROOT / "BM1" / "reference.mp4"
ELEMENT = ROOT / "BM1" / "rm2.jpeg"
OUT_ROOT = ROOT / "testbench" / "output"

MODEL = "runway/aleph-2"
RATE = 0.28

# Change-only. Everything unmentioned - camera moves, cuts, pacing, the pour's
# choreography - is preserved by the editor and must stay out of the prompt.
PROMPT = (
    "Replace the blue Parachute SkinPure body lotion bottle with the Parachute "
    "Advansed Onion Enriched Coconut Hair Oil bottle shown in the reference "
    "image: a tall slim amber translucent bottle with a magenta flip cap, its "
    "front label carrying the Parachute palm-tree logo, ADVANSED, ONION, "
    "ENRICHED COCONUT HAIR OIL, the Arabic text, HAIR FALL CONTROL and the "
    "coconut-and-onion artwork, sharp and legible.\n\n"
    "When the bottle turns to show its reverse, show the onion hair oil's own "
    "back panel: a dark magenta information panel carrying a block of fine "
    "ingredients text, a row of small white care-symbol icons and a barcode. "
    "The words HYALURON, BODY LOTION and SkinPure must never appear anywhere "
    "in the video.\n\n"
    "Remove the small SkinPure logo from the top-left corner of the frame.\n\n"
    "Change the background from blue to the soft pink gradient and glossy "
    "reflective surface of the reference image, with warm pink light on the "
    "glass and a clean reflection beneath the bottle. Remove the blue light "
    "rays and the water droplets.\n\n"
    "The pouring liquid stays thick and white."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually spend")
    ap.add_argument("--budget", type=float, default=3.50)
    args = ap.parse_args()

    for p in (REFERENCE, ELEMENT):
        if not p.exists():
            sys.exit(f"missing: {p}")

    src = probe(REFERENCE)
    est = round(src.duration * RATE, 4)

    print("\n" + "=" * 66)
    print(f"  ALEPH 2.0   {'LIVE' if args.go else 'PREFLIGHT (no spend)'}")
    print("=" * 66)
    print(f"\n  reference  {REFERENCE.name}  {src.width}x{src.height}  "
          f"{src.duration:.2f}s  {src.fps:.0f}fps")
    print(f"  element    {ELEMENT.name}")
    print(f"  model      {MODEL}  ${RATE}/s")
    print(f"  est cost   ${est:.2f}   budget ${args.budget:.2f}")

    try:
        left = providers.remaining()
        print(f"  credits    ${left:.2f}")
    except Exception:
        left = None
    if est > args.budget:
        sys.exit(f"\n  estimate ${est:.2f} exceeds budget ${args.budget:.2f}")
    if left is not None and est > left:
        sys.exit(f"\n  estimate ${est:.2f} exceeds credits ${left:.2f}")

    job_dir = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_aleph"
    job_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [1/4] decompiling the reference ...")
    bp_ref = decompile(REFERENCE, job_dir / "ref")
    save(bp_ref, job_dir / "ref")
    print(f"        {len(bp_ref.shots)} shots, {bp_ref.cut_count} cuts")

    stage = job_dir / "upload"
    stage.mkdir(exist_ok=True)
    shutil.copy2(REFERENCE, stage / "reference.mp4")
    shutil.copy2(ELEMENT, stage / ELEMENT.name)

    if not args.go:
        print("\n  [2/4] verifying tunnel ...")
        with tunnel.serve_public(stage) as base:
            print(f"        OK {tunnel.url_for(base, 'reference.mp4')}")
        print("\n" + "-" * 66)
        print("  PREFLIGHT PASSED - spent $0.00")
        print(f"  run:  python tools/aleph_run.py --go     (${est:.2f})")
        print("-" * 66 + "\n")
        return 0

    before = left
    with tunnel.serve_public(stage) as base:
        print(f"\n  [2/4] submitting to {MODEL} ...")
        t0 = time.time()
        job = providers.submit_video(
            MODEL, PROMPT,
            video_url=tunnel.url_for(base, "reference.mp4"),
            image_urls=[tunnel.url_for(base, ELEMENT.name)],
            # duration deliberately omitted so the edit matches the source
        )
        print(f"        job {job.id}")

        print("  [3/4] polling ...")
        seen = {"s": ""}

        def tick(status, _):
            if status != seen["s"]:
                seen["s"] = status
                print(f"        {time.time() - t0:6.0f}s  {status}", flush=True)

        payload = providers.poll_video(job, on_tick=tick)
        urls = providers.video_urls(payload)
        if not urls:
            sys.exit(f"no output url: {str(payload)[:300]}")
        result = providers.download(urls[0], job_dir / "result.mp4")
        print(f"        done in {time.time() - t0:.0f}s")

    print("\n  [4/4] scoring ...")
    bp_out = decompile(result, job_dir / "out")
    save(bp_out, job_dir / "out")
    rep = fidelity.compare(bp_ref, bp_out)
    print()
    print(fidelity.render(rep))

    ffrun(["ffmpeg", "-v", "error", "-y", "-i", str(REFERENCE), "-i", str(result),
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(job_dir / "sidebyside.mp4")])

    spent = None
    if before is not None:
        try:
            spent = round(before - providers.remaining(), 4)
        except Exception:
            pass

    s = probe(result)
    print("\n" + "-" * 66)
    if spent is not None:
        print(f"  spent      ${spent:.2f}  (estimated ${est:.2f})")
    print(f"  output     {result}")
    print(f"             {s.width}x{s.height}  {s.duration:.2f}s")
    print(f"  compare    {job_dir / 'sidebyside.mp4'}")
    print("-" * 66 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
