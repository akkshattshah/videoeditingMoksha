"""Camera motion classification via sparse Lucas-Kanade optical flow.

Decomposes inter-frame point motion into translation and radial expansion,
which separates pans/tilts from push-ins/pull-outs. Frame-to-frame variance of
the translation vector gives a handheld/shake estimate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schema import Camera, CameraMove, Provenance, Shot

ANALYSIS_WIDTH = 320
MAX_PAIRS = 24          # sampled frame pairs per shot
MIN_TRACKED = 12        # below this the estimate is not trustworthy
# Mean inter-frame pixel delta below which a shot is treated as locked off.
STATIC_PIXEL_DELTA = 0.9

# Thresholds as a fraction of frame width, per frame.
T_TRANSLATE = 0.0016
T_ZOOM = 0.0011
T_SHAKE = 0.25
# Maps normalised jitter onto 0..1; ~0.007 of frame width per frame reads as
# fully handheld.
SHAKE_GAIN = 140.0

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_FEATURES = dict(maxCorners=220, qualityLevel=0.01, minDistance=8, blockSize=7)


def _pair_motion(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float, float] | None:
    """Return (tx, ty, zoom) normalised by frame width, or None if unreliable."""
    p0 = cv2.goodFeaturesToTrack(prev, mask=None, **_FEATURES)
    if p0 is None or len(p0) < MIN_TRACKED:
        return None

    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **_LK)
    if p1 is None or status is None:
        return None

    ok = status.ravel() == 1
    if ok.sum() < MIN_TRACKED:
        return None

    a = p0.reshape(-1, 2)[ok]
    b = p1.reshape(-1, 2)[ok]
    flow = b - a

    h, w = prev.shape[:2]
    centre = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    # Translation: median resists the handful of points that latch onto
    # moving subjects rather than the background.
    tx, ty = np.median(flow, axis=0)

    # Radial expansion: project residual flow onto the outward unit vector.
    radial = a - centre
    norms = np.linalg.norm(radial, axis=1)
    live = norms > (w * 0.06)          # points near the centre carry no zoom signal
    if live.sum() >= MIN_TRACKED // 2:
        unit = radial[live] / norms[live][:, None]
        residual = flow[live] - np.array([tx, ty])
        zoom = float(np.median(np.sum(residual * unit, axis=1)))
    else:
        zoom = 0.0

    return float(tx) / w, float(ty) / w, zoom / w


def _sample_grays(path: str | Path, t_in: float, t_out: float,
                  max_pairs: int = MAX_PAIRS) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(t_in * fps)
    end = max(start + 2, int(t_out * fps))
    total = end - start
    stride = max(1, total // (max_pairs + 1))

    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx < end and len(frames) <= max_pairs:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            h, w = frame.shape[:2]
            if w > ANALYSIS_WIDTH:
                scale = ANALYSIS_WIDTH / w
                frame = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1

    cap.release()
    return frames


def classify_shot(path: str | Path, shot: Shot) -> Camera:
    frames = _sample_grays(path, shot.t_in, shot.t_out)
    if len(frames) < 2:
        return Camera(move=CameraMove.UNKNOWN,
                      provenance=Provenance(backend="opencv-lk", confidence=0.2,
                                            note="too few frames"))

    # Check for a genuinely frozen frame before trying to track features. A
    # locked-off shot on a flat surface has nothing for LK to latch onto, and
    # would otherwise be reported as UNKNOWN when the honest answer is STATIC.
    diffs = [
        float(np.abs(frames[i].astype(np.int16) - frames[i + 1].astype(np.int16)).mean())
        for i in range(len(frames) - 1)
    ]
    if diffs and max(diffs) < STATIC_PIXEL_DELTA:
        return Camera(
            move=CameraMove.STATIC, speed=0.0, shake=0.0,
            provenance=Provenance(backend="frame-diff", confidence=0.9,
                                  note=f"max inter-frame delta {max(diffs):.2f}"),
        )

    measurements = [
        m for m in (_pair_motion(frames[i], frames[i + 1])
                    for i in range(len(frames) - 1))
        if m is not None
    ]
    if not measurements:
        return Camera(move=CameraMove.UNKNOWN,
                      provenance=Provenance(backend="opencv-lk", confidence=0.2,
                                            note="no trackable features"))

    arr = np.asarray(measurements)           # (n, 3) → tx, ty, zoom
    tx, ty, zoom = np.median(arr, axis=0)

    # Shake: absolute inconsistency of the translation, already normalised by
    # frame width. Deliberately NOT divided by translation magnitude - for a
    # pure zoom the translation is ~0, and the ratio would report a locked-off
    # tripod move as violently handheld.
    jitter = float(np.mean(np.std(arr[:, :2], axis=0)))
    shake = float(np.clip(jitter * SHAKE_GAIN, 0.0, 1.0)) if len(arr) > 2 else 0.0

    move = CameraMove.STATIC
    speed = 0.0

    if abs(zoom) > T_ZOOM and abs(zoom) > max(abs(tx), abs(ty)) * 0.7:
        move = CameraMove.PUSH_IN if zoom > 0 else CameraMove.PULL_OUT
        speed = abs(zoom)
    elif abs(tx) > T_TRANSLATE and abs(tx) >= abs(ty):
        # Positive tx means content moved right, i.e. the camera panned left.
        move = CameraMove.PAN_LEFT if tx > 0 else CameraMove.PAN_RIGHT
        speed = abs(tx)
    elif abs(ty) > T_TRANSLATE:
        move = CameraMove.TILT_UP if ty > 0 else CameraMove.TILT_DOWN
        speed = abs(ty)

    if move == CameraMove.STATIC and shake > T_SHAKE and jitter > T_TRANSLATE:
        move = CameraMove.HANDHELD
        speed = jitter

    # Confidence scales with how many pairs actually resolved.
    conf = round(min(0.82, 0.35 + 0.05 * len(measurements)), 3)

    return Camera(
        move=move,
        speed=round(float(np.clip(speed * 90.0, 0.0, 1.0)), 4),
        shake=round(shake, 4),
        provenance=Provenance(backend="opencv-lk", confidence=conf,
                              note=f"{len(measurements)} pairs"),
    )


def classify_all(path: str | Path, shots: list[Shot]) -> None:
    """Populate `shot.camera` in place."""
    for shot in shots:
        shot.camera = classify_shot(path, shot)
