"""Shot boundary detection.

Default backend is an HSV content detector equivalent to PySceneDetect's
ContentDetector, running on downscaled frames so it stays cheap on CPU. Every
run is cross-checked against ffmpeg's independent `scene` filter; agreement
between the two raises the confidence recorded in the blueprint.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..probe import ffmpeg_scene_scores
from ..schema import Provenance, Shot, SourceMeta, TransitionType

# PySceneDetect's default content threshold, on a 0..255 scale.
DEFAULT_THRESHOLD = 27.0
# Suppress boundaries closer together than this; ad cuts are rarely faster.
MIN_SHOT_SEC = 0.35
# Width frames are downscaled to before differencing.
ANALYSIS_WIDTH = 160
# A shot opening below this fraction of its own settled sharpness was cut
# mid-camera-move, not at a real boundary. Measured on real ad footage:
# genuine shots sit at 0.75-1.0, whip-pan artefacts at 0.05-0.14.
BLUR_MERGE_RATIO = 0.30


def _hsv_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-channel HSV difference, with circular hue handling.

    OpenCV packs hue into 0..179, so a naive absdiff reports ~179 for two
    nearly identical reds straddling the wrap point.
    """
    ah, as_, av = cv2.split(a.astype(np.int16))
    bh, bs, bv = cv2.split(b.astype(np.int16))

    dh = np.abs(ah - bh)
    dh = np.minimum(dh, 180 - dh)  # wrap

    return float((dh.mean() + np.abs(as_ - bs).mean() + np.abs(av - bv).mean()) / 3.0)


def _score_series(path: str | Path) -> tuple[list[float], float]:
    """Per-frame HSV delta for the whole clip, plus measured fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores: list[float] = []
    prev: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > ANALYSIS_WIDTH:
            scale = ANALYSIS_WIDTH / w
            frame = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if prev is not None:
            scores.append(_hsv_delta(prev, hsv))
        prev = hsv

    cap.release()
    return scores, fps


def _classify_transition(scores: list[float], idx: int, threshold: float) -> TransitionType:
    """Spike = cut. Sustained mid-level elevation = dissolve or fade."""
    lo = max(0, idx - 4)
    hi = min(len(scores), idx + 5)
    window = scores[lo:hi]
    if not window:
        return TransitionType.CUT

    peak = scores[idx]
    neighbours = [s for i, s in enumerate(scores[lo:hi], start=lo) if i != idx]
    if not neighbours:
        return TransitionType.CUT

    elevated = sum(1 for s in neighbours if s > threshold * 0.45)
    # A hard cut is one isolated spike; a dissolve smears across frames.
    if elevated >= 3 and peak < threshold * 2.2:
        return TransitionType.DISSOLVE
    return TransitionType.CUT


def detect_boundaries(
    path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_sec: float = MIN_SHOT_SEC,
) -> tuple[list[float], list[TransitionType], list[float]]:
    """Return (cut_times, transition_types, raw_scores)."""
    scores, fps = _score_series(path)
    if not scores:
        return [], [], []

    min_gap = max(1, int(min_shot_sec * fps))
    cuts: list[float] = []
    kinds: list[TransitionType] = []
    last_idx = -min_gap

    for i, s in enumerate(scores):
        if s <= threshold or (i - last_idx) < min_gap:
            continue
        # score[i] is the delta between frame i and i+1, so the new shot
        # starts at frame i+1
        cuts.append(round((i + 1) / fps, 4))
        kinds.append(_classify_transition(scores, i, threshold))
        last_idx = i

    return cuts, kinds, scores


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_at(cap: cv2.VideoCapture, t: float, fps: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def merge_blur_boundaries(path: str | Path, shots: list[Shot],
                          ratio: float = BLUR_MERGE_RATIO) -> int:
    """Merge boundaries that landed inside a whip-pan rather than on a cut.

    Content detectors cannot tell a hard cut from a fast camera move: both
    produce a large frame-to-frame delta. ffmpeg's detector makes the same
    mistake, so cross-check agreement does not catch it.

    The tell is asymmetric blur. A shot that opens far blurrier than its own
    middle and end did not start at a cut - the boundary was placed mid-move
    and the camera settled afterwards. A shot that *ends* blurry is just
    whipping out, which is legitimate and left alone.

    Returns the number of boundaries removed.
    """
    if len(shots) < 2:
        return 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    drop: list[int] = []
    for i in range(1, len(shots)):
        s = shots[i]
        dur = s.duration
        pad = min(0.06, dur * 0.2)
        first = _frame_at(cap, s.t_in + pad, fps)
        mid = _frame_at(cap, (s.t_in + s.t_out) / 2.0, fps)
        last = _frame_at(cap, max(s.t_in + pad, s.t_out - pad), fps)
        if first is None or (mid is None and last is None):
            continue

        settled = max(_sharpness(f) for f in (mid, last) if f is not None)
        if settled <= 1e-6:
            continue
        if _sharpness(first) / settled < ratio:
            drop.append(i)

    cap.release()
    if not drop:
        return 0

    # Fold each flagged shot into its predecessor, back to front so the
    # indices stay valid.
    for i in reversed(drop):
        prev = shots[i - 1]
        prev.t_out = shots[i].t_out
        prev.transition_out = shots[i].transition_out
        shots.pop(i)

    for idx, s in enumerate(shots):
        s.index = idx
        s.id = f"s{idx + 1:02d}"
    return len(drop)


def _agreement(mine: list[float], theirs: list[float], tol: float = 0.35) -> float:
    """Fraction of my cuts that ffmpeg also found, within `tol` seconds."""
    if not mine:
        return 1.0 if not theirs else 0.0
    matched = sum(1 for t in mine if any(abs(t - o) <= tol for o in theirs))
    return matched / len(mine)


def detect(
    source: SourceMeta,
    threshold: float = DEFAULT_THRESHOLD,
    cross_check: bool = True,
) -> tuple[list[Shot], dict]:
    """Build the shot list for a source. Returns (shots, diagnostics)."""
    cuts, kinds, scores = detect_boundaries(source.path, threshold=threshold)

    confidence = 0.75
    diag: dict = {"raw_cuts": len(cuts), "threshold": threshold}

    if cross_check:
        try:
            ffmpeg_cuts = ffmpeg_scene_scores(source.path, threshold=0.3)
            agree = _agreement(cuts, ffmpeg_cuts)
            diag["ffmpeg_cuts"] = len(ffmpeg_cuts)
            diag["agreement"] = round(agree, 3)
            # Two unrelated detectors agreeing is worth more than either alone.
            confidence = round(min(0.94, 0.62 + 0.32 * agree), 3)
        except Exception as exc:  # cross-check is advisory, never fatal
            diag["cross_check_error"] = str(exc)[:200]

    prov = Provenance(
        backend="opencv-content",
        confidence=confidence,
        note=f"threshold={threshold}, agreement={diag.get('agreement')}",
    )

    # Turn boundaries into closed [t_in, t_out) intervals.
    edges = [0.0, *cuts, source.duration]
    shots: list[Shot] = []
    for i in range(len(edges) - 1):
        t_in, t_out = edges[i], edges[i + 1]
        if t_out - t_in <= 0.01:
            continue
        trans_in = kinds[i - 1] if 0 < i <= len(kinds) else TransitionType.CUT
        trans_out = kinds[i] if i < len(kinds) else TransitionType.CUT
        shots.append(
            Shot(
                id=f"s{len(shots) + 1:02d}",
                index=len(shots),
                t_in=round(t_in, 4),
                t_out=round(t_out, 4),
                transition_in=trans_in,
                transition_out=trans_out,
                provenance=prov,
            )
        )

    merged = merge_blur_boundaries(source.path, shots)
    diag["blur_merged"] = merged

    diag["mean_score"] = round(float(np.mean(scores)), 3) if scores else 0.0
    diag["max_score"] = round(float(np.max(scores)), 3) if scores else 0.0
    return shots, diag
