"""Keyframe extraction.

Per shot: first, middle and last stills, plus a contact-sheet grid. The grid is
what gets sent to the VLM — one image per shot costs far fewer tokens than
three, and the model reads motion better from a strip than from isolated stills.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..probe import extract_frame
from ..schema import Keyframes, Shot

# Nudge away from the boundary so we never land on a transition frame.
EDGE_INSET = 0.06
GRID_COLS = 4
GRID_TILE_W = 320


def _inset_times(shot: Shot) -> tuple[float, float, float]:
    dur = shot.duration
    pad = min(EDGE_INSET, dur * 0.2)
    first = shot.t_in + pad
    last = max(first, shot.t_out - pad)
    mid = (shot.t_in + shot.t_out) / 2.0
    return first, mid, last


def _contact_sheet(path: str | Path, shot: Shot, out: Path,
                   cols: int = GRID_COLS) -> bool:
    """Sample `cols` evenly spaced frames into one horizontal strip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = shot.duration
    pad = min(EDGE_INSET, dur * 0.2)
    times = np.linspace(shot.t_in + pad, max(shot.t_in + pad, shot.t_out - pad), cols)

    tiles: list[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = GRID_TILE_W / w
        tiles.append(cv2.resize(frame, (GRID_TILE_W, max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA))
    cap.release()

    if not tiles:
        return False

    height = min(t.shape[0] for t in tiles)
    tiles = [t[:height] for t in tiles]
    sheet = np.hstack(tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]))


def extract(path: str | Path, shots: list[Shot], job_dir: Path,
            grid: bool = True) -> None:
    """Populate `shot.keyframes` in place, writing files under job_dir/frames."""
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for shot in shots:
        first_t, mid_t, last_t = _inset_times(shot)
        kf = Keyframes()

        for label, t in (("first", first_t), ("mid", mid_t), ("last", last_t)):
            dest = frames_dir / f"{shot.id}_{label}.jpg"
            if extract_frame(path, t, dest):
                # Store paths relative to the job dir so blueprints stay portable.
                setattr(kf, label, str(dest.relative_to(job_dir)).replace("\\", "/"))

        if grid:
            dest = frames_dir / f"{shot.id}_grid.jpg"
            if _contact_sheet(path, shot, dest):
                kf.grid = str(dest.relative_to(job_dir)).replace("\\", "/")

        shot.keyframes = kf
