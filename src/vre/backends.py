"""Backend capability registry.

Every analyzer declares the backends it can use, ordered best-first. At runtime
the pipeline picks the best available one and records the choice in the
blueprint's AnalysisReport. This is what lets the same code run on a CPU laptop
today and against GPU workers on Modal later without branching at the call site.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backend:
    name: str
    # base confidence assigned to values this backend produces
    confidence: float
    requires_modules: tuple[str, ...] = ()
    requires_binaries: tuple[str, ...] = ()
    # GPU backends are registered but never auto-selected locally
    remote_only: bool = False
    note: str = ""

    def available(self, allow_remote: bool = False) -> bool:
        if self.remote_only and not allow_remote:
            return False
        for mod in self.requires_modules:
            if importlib.util.find_spec(mod) is None:
                return False
        for exe in self.requires_binaries:
            if shutil.which(exe) is None:
                return False
        return True


@dataclass
class Stage:
    """One analysis step with a ranked list of possible backends."""

    key: str
    description: str
    backends: list[Backend] = field(default_factory=list)

    def select(self, allow_remote: bool = False) -> Backend | None:
        for b in self.backends:
            if b.available(allow_remote):
                return b
        return None

    def missing(self, allow_remote: bool = False) -> list[str]:
        return [b.name for b in self.backends if not b.available(allow_remote)]


# ── registry ─────────────────────────────────────────────────────────────────

STAGES: dict[str, Stage] = {
    "shots": Stage(
        key="shots",
        description="Shot boundary detection",
        backends=[
            Backend("transnetv2", 0.95, ("torch",), remote_only=True,
                    note="handles dissolves and fast cuts"),
            Backend("pyscenedetect", 0.85, ("scenedetect",), ("ffmpeg",)),
            Backend("opencv-content", 0.75, ("cv2", "numpy"), ("ffprobe",),
                    note="HSV content detector, hard cuts only"),
        ],
    ),
    "motion": Stage(
        key="motion",
        description="Camera motion classification",
        backends=[
            Backend("cotracker", 0.92, ("torch",), remote_only=True),
            Backend("opencv-lk", 0.70, ("cv2", "numpy"),
                    note="Lucas-Kanade sparse flow"),
        ],
    ),
    "audio": Stage(
        key="audio",
        description="Tempo, beat grid and loudness",
        backends=[
            Backend("librosa", 0.85, ("librosa",), ("ffmpeg",)),
            Backend("numpy-flux", 0.62, ("numpy",), ("ffmpeg",),
                    note="spectral-flux onset + autocorrelation tempo"),
        ],
    ),
    "transcript": Stage(
        key="transcript",
        description="VO transcription with word timings",
        backends=[
            Backend("whisperx", 0.93, ("whisperx",), remote_only=True),
            Backend("faster-whisper", 0.88, ("faster_whisper",), remote_only=True),
        ],
    ),
    "ocr": Stage(
        key="ocr",
        description="On-screen text detection",
        backends=[
            Backend("paddleocr", 0.88, ("paddleocr",), remote_only=True),
            Backend("vlm-openrouter", 0.75, ("httpx",),
                    note="needs OPENROUTER_API_KEY"),
        ],
    ),
    "masks": Stage(
        key="masks",
        description="Subject segmentation and tracking",
        backends=[
            Backend("sam2", 0.94, ("torch",), remote_only=True),
        ],
    ),
    "depth": Stage(
        key="depth",
        description="Temporally consistent depth",
        backends=[
            Backend("video-depth-anything", 0.90, ("torch",), remote_only=True),
        ],
    ),
}


def report(allow_remote: bool = False) -> dict[str, dict]:
    """Snapshot of what would run right now. Backs the `vre doctor` command."""
    out: dict[str, dict] = {}
    for key, stage in STAGES.items():
        chosen = stage.select(allow_remote)
        out[key] = {
            "description": stage.description,
            "selected": chosen.name if chosen else None,
            "confidence": chosen.confidence if chosen else 0.0,
            "note": chosen.note if chosen else "",
            "unavailable": stage.missing(allow_remote),
        }
    return out
