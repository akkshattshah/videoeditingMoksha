"""Decompile orchestrator: reference video -> blueprint.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import backends
from ..probe import probe
from ..schema import AnalysisReport, Blueprint
from . import audio as audio_mod
from . import frames as frames_mod
from . import motion as motion_mod
from . import shots as shots_mod


class Timer:
    def __init__(self, report: AnalysisReport, key: str) -> None:
        self.report, self.key = report, key

    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        elapsed = (time.perf_counter() - self.t0) * 1000
        self.report.timings_ms[self.key] = int(elapsed)


def decompile(
    video: str | Path,
    job_dir: str | Path,
    *,
    threshold: float = shots_mod.DEFAULT_THRESHOLD,
    do_frames: bool = True,
    do_motion: bool = True,
    do_audio: bool = True,
    allow_remote: bool = False,
) -> Blueprint:
    """Run every locally available analyzer and assemble a Blueprint."""
    video = Path(video)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    report = AnalysisReport()

    with Timer(report, "probe"):
        source = probe(video)

    # Record what the registry selected, so the blueprint is self-describing.
    snapshot = backends.report(allow_remote=allow_remote)
    for key, info in snapshot.items():
        if info["selected"]:
            report.backends_used[key] = info["selected"]
        else:
            report.backends_missing.append(key)

    with Timer(report, "shots"):
        shot_list, diag = shots_mod.detect(source, threshold=threshold)
    if diag.get("agreement") is not None and diag["agreement"] < 0.6:
        report.warnings.append(
            f"opencv and ffmpeg detectors agree on only "
            f"{diag['agreement']:.0%} of cuts — inspect the shot list"
        )
    if diag.get("blur_merged"):
        report.warnings.append(
            f"merged {diag['blur_merged']} boundary(ies) that fell inside a "
            "whip-pan rather than on a cut"
        )
    if not shot_list:
        report.warnings.append("no shots detected; treating clip as a single shot")

    if do_motion and shot_list:
        with Timer(report, "motion"):
            motion_mod.classify_all(source.path, shot_list)

    if do_frames and shot_list:
        with Timer(report, "frames"):
            frames_mod.extract(source.path, shot_list, job_dir)

    track = None
    if do_audio:
        with Timer(report, "audio"):
            track = audio_mod.analyze(source.path, source.duration, source.has_audio)

    bp = Blueprint(source=source, shots=shot_list, analysis=report)
    if track is not None:
        bp.audio = track

    # Stages that need GPU or a VLM are declared but not run locally.
    for stage in ("ocr", "masks", "depth", "transcript"):
        if stage not in report.backends_used:
            report.warnings.append(f"{stage}: no local backend, slots will be incomplete")

    return bp


def save(bp: Blueprint, job_dir: str | Path) -> Path:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    out = job_dir / "blueprint.json"
    out.write_text(bp.model_dump_json(indent=2), encoding="utf-8")
    return out


def load(path: str | Path) -> Blueprint:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Blueprint.model_validate(data)
