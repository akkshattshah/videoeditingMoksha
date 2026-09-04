"""Preview the reference2 subject swap as stills before paying to animate it.

Three stills cost about $0.06; five seconds of video costs $0.65. Any doubt
about whether the model can put THIS man in THIS hoodie should be settled here.

The composition lock matters: on the BM1 job the image model reset framing to a
centred packshot on six attempts out of six, so the prompt has to nail the
frame down explicitly.

    python tools/frames_ref2.py
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

SOURCE = ROOT / "reference2.mp4"
ELEMENT = ROOT / "hoodie" / "desert-dune_01.jpg"
OUT = ROOT / "testbench" / "output" / "ref2_frames"

MODEL = "google/gemini-2.5-flash-image"

# The beats worth checking, one per distinct shot type in the 0-5s window.
SHOTS = [
    (0.40, "wide", "full body, static wide"),
    (1.70, "face", "face close-up, hood up"),
    (4.00, "mid", "full body, later shot"),
]

PROMPT = (
    "The first image is a frame from a fashion film showing a man in a bright "
    "blue hooded jacket and dark trousers in a grey studio. The second image "
    "shows a different man wearing a tan Desert Dune oversized hoodie.\n\n"
    "Redraw the first frame so the man from the SECOND image is the one in the "
    "scene: his face, his short swept brown hair, his build. Dress him in that "
    "same heavyweight tan camel-brown hoodie with ribbed cuffs and hem, a "
    "kangaroo front pocket, and the small rectangular graphic patch across the "
    "chest. The hoodie is loose and oversized, its thick fabric creasing "
    "naturally. His trousers become matching tan.\n\n"
    "COMPOSITION IS FIXED. He must stand in exactly the same place, in the same "
    "pose, at the same size in the frame, cropped identically at the same "
    "edges. Do not zoom out, do not re-centre him, do not turn him to face the "
    "camera differently, and do not show more of his body than the original "
    "frame shows.\n\n"
    "Keep the grey studio wall, the ladder, the floor, the lighting and the "
    "shadows exactly as they are. Keep any text or panel overlays exactly where "
    "they are."
)


def frame_at(video: Path, t: float, dest: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
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
    ap.add_argument("--variants", "-n", type=int, default=1)
    args = ap.parse_args()

    for p in (SOURCE, ELEMENT):
        if not p.exists():
            sys.exit(f"missing: {p}")
    OUT.mkdir(parents=True, exist_ok=True)

    before = providers.remaining()
    print(f"\n  element {ELEMENT.name}   credits ${before:.2f}\n")

    pairs = []
    for t, tag, desc in SHOTS:
        src = frame_at(SOURCE, t, OUT / f"src_{tag}.jpg")
        for v in range(args.variants):
            print(f"    {tag} @{t}s  ({desc}) ...", flush=True)
            data = providers.edit_image(src, PROMPT, MODEL, extra_images=[ELEMENT])
            if not data:
                print("      failed")
                continue
            dest = OUT / f"{tag}_v{v + 1}.jpg"
            dest.write_bytes(data)
            pairs.append((tag, src, dest))
            print(f"      -> {dest.name}")

    if pairs:
        h = 460
        def fit(p):
            im = cv2.imread(str(p))
            return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        tiles, labels = [], []
        for tag, src, dest in pairs:
            tiles += [fit(src), fit(dest)]
            labels += [f"{tag} BEFORE", f"{tag} AFTER"]
        for im, lb in zip(tiles, labels):
            cv2.rectangle(im, (0, 0), (170, 24), (0, 0, 0), -1)
            cv2.putText(im, lb, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 255), 2)
        cv2.imwrite(str(OUT / "compare.jpg"), np.hstack(tiles),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])

    spent = round(before - providers.remaining(), 4)
    print(f"\n  spent ${spent:.3f}   {OUT / 'compare.jpg'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
