"""ffprobe / ffmpeg wrappers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .schema import SourceMeta


class FFmpegMissing(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FFmpegMissing(
            f"{tool} not found on PATH. Install ffmpeg and re-run `vre doctor`."
        )
    return path


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
        errors="replace",
    )


def _aspect(w: int, h: int) -> str:
    g = math.gcd(w, h) or 1
    return f"{w // g}:{h // g}"


def probe(path: str | Path) -> SourceMeta:
    """Read container/stream metadata. Fast — no decoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ffprobe = _require("ffprobe")
    res = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()[:400]}")

    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")

    # avg_frame_rate is a rational string like "30000/1001"
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = video.get(key) or "0/0"
        try:
            num, _, den = raw.partition("/")
            if float(den or 0) > 0:
                fps = float(num) / float(den)
                break
        except (ValueError, ZeroDivisionError):
            continue

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    w, h = int(video["width"]), int(video["height"])

    return SourceMeta(
        path=str(path.resolve()),
        duration=round(duration, 4),
        fps=round(fps, 6),
        width=w,
        height=h,
        aspect=_aspect(w, h),
        has_audio=audio is not None,
        video_codec=video.get("codec_name"),
        audio_codec=(audio or {}).get("codec_name"),
        size_bytes=int(fmt["size"]) if fmt.get("size") else None,
    )


def decode_audio_mono(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Decode the audio track to a mono float32 array via ffmpeg stdout.

    Avoids a soundfile/librosa dependency — ffmpeg is already required.
    """
    ffmpeg = _require("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sr), "-",
        ],
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def integrated_lufs(path: str | Path) -> float | None:
    """Integrated loudness via ffmpeg's ebur128 filter."""
    ffmpeg = _require("ffmpeg")
    res = run([
        ffmpeg, "-v", "info", "-i", str(path),
        "-af", "ebur128=framelog=quiet", "-f", "null", "-",
    ])
    # ffmpeg prints the summary block to stderr
    marker = "I:"
    for line in reversed(res.stderr.splitlines()):
        stripped = line.strip()
        if stripped.startswith(marker) and "LUFS" in stripped:
            try:
                return float(stripped.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def extract_frame(path: str | Path, t: float, out: str | Path, width: int = 512) -> bool:
    """Write a single JPEG at timestamp `t`."""
    ffmpeg = _require("ffmpeg")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res = run([
        ffmpeg, "-v", "error", "-y",
        "-ss", f"{max(t, 0):.4f}", "-i", str(path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "3", str(out),
    ])
    return res.returncode == 0 and Path(out).exists()


def ffmpeg_scene_scores(path: str | Path, threshold: float = 0.3) -> list[float]:
    """Scene-change timestamps from ffmpeg's own detector.

    Used as an independent cross-check on the opencv detector — agreement
    between two unrelated methods is the cheapest confidence signal available.
    """
    ffmpeg = _require("ffmpeg")
    res = run([
        ffmpeg, "-v", "info", "-i", str(path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ])
    times: list[float] = []
    for line in res.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            chunk = line.split("pts_time:")[1].split()[0]
            times.append(round(float(chunk), 4))
        except (IndexError, ValueError):
            continue
    return sorted(times)
