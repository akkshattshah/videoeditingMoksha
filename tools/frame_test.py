"""Generate a single keyframe before paying to animate it.

A still costs ~$0.04; a second of video costs $0.13. Any question about whether
the model *can* render a given product at a given angle should be answered here
first, not inside a finished video.

    python tools/frame_test.py --time 3.4 --variants 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre import providers  # noqa: E402

SOURCE = ROOT / "testbench" / "output" / "20260810_194720_h3" / "segment.mp4"
ELEMENT = ROOT / "BM1" / "rm2.jpeg"
OUT = ROOT / "testbench" / "output" / "frames"

MODEL = "google/gemini-2.5-flash-image"
COST = 0.0387

COMMON = (
    "The first image is a frame from a product film showing a blue Parachute "
    "SkinPure body lotion bottle. The second image shows a Parachute Advansed "
    "Onion Enriched Coconut Hair Oil bottle - a tall slim amber translucent "
    "bottle with a magenta flip cap - on a pink background.\n\n"
    "Redraw the first frame so it shows the ONION HAIR OIL bottle instead of "
    "the blue lotion bottle.\n\n"
)

FRAMING = (
    "\n\nCOMPOSITION IS FIXED. The new bottle must occupy exactly the same area "
    "of the frame as the blue bottle does, cropped the same way at the same "
    "edges, at the same scale and the same angle. Do not zoom out. Do not "
    "centre or straighten the bottle. Do not show the whole bottle if the "
    "original is cropped.\n\n"
    "CRITICAL: do not copy any wording from the blue bottle. The words "
    "HYALURON, BODY LOTION and SkinPure must not appear anywhere, and remove "
    "the SkinPure corner logo.\n\n"
    "Replace the blue background, blue light and water droplets with the soft "
    "pink gradient and glossy reflective surface from the second image. Keep "
    "the lighting direction identical."
)

# Each mode targets one beat the video model has failed on.
MODES: dict[str, tuple[float, str]] = {
    "back": (3.4, COMMON + (
        "The blue bottle is rotated to show its BACK panel. Show the onion oil "
        "bottle rotated to the same angle, its back carrying a dark magenta "
        "information panel with fine ingredients text, a row of small white "
        "care-symbol icons and a barcode."
    ) + FRAMING),
    "zoom": (1.8, COMMON + (
        "The blue bottle is shot LARGE, filling most of the frame, upright and "
        "front-on, cropped by the frame edges. Show the onion oil bottle at "
        "that same large scale, front label facing camera, filling the frame "
        "identically."
    ) + FRAMING),
    "slant": (4.4, COMMON + (
        "The blue bottle is TILTED, leaning diagonally across the frame in an "
        "extreme close-up, its label large and running at an angle rather than "
        "vertical. Show the onion oil bottle tilted at exactly the same "
        "diagonal lean, in the same extreme close-up, its ONION label large and "
        "angled the same way. The bottle must NOT be upright."
    ) + FRAMING),
}


def frame_at(video: Path, t: float, dest: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read {video} at {t}s")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", "-m", default="slant", choices=list(MODES))
    ap.add_argument("--variants", "-n", type=int, default=2)
    ap.add_argument("--go", action="store_true", default=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t, prompt = MODES[args.mode]
    src = frame_at(SOURCE, t, OUT / f"src_{args.mode}.jpg")
    est = args.variants * COST
    print(f"\n  source frame  {src.name}")
    print(f"  model         {MODEL}")
    print(f"  variants      {args.variants}  ->  ${est:.3f}\n")

    before = providers.remaining()
    made: list[Path] = []
    for i in range(args.variants):
        print(f"    generating variant {i + 1} ...", flush=True)
        data = providers.edit_image(src, prompt, MODEL, extra_images=[ELEMENT])
        if not data:
            print("      failed")
            continue
        dest = OUT / f"{args.mode}_v{i + 1}.jpg"
        dest.write_bytes(data)
        made.append(dest)
        print(f"      -> {dest.name}")

    if made:
        ref = cv2.imread(str(src))
        h = 620
        def fit(p):
            im = cv2.imread(str(p))
            return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        tiles = [fit(src)] + [fit(p) for p in made]
        labels = ["ORIGINAL frame"] + [f"variant {i+1}" for i in range(len(made))]
        for im, lb in zip(tiles, labels):
            cv2.rectangle(im, (0, 0), (230, 26), (0, 0, 0), -1)
            cv2.putText(im, lb, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        cv2.imwrite(str(OUT / f'{args.mode}_compare.jpg'), np.hstack(tiles),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    spent = round(before - providers.remaining(), 4)
    print(f"\n  spent ${spent:.3f}   compare: {OUT / (args.mode + '_compare.jpg')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
