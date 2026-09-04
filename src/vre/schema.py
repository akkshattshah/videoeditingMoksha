"""Blueprint schema — the structured representation of a reference video's edit.

This is the contract every other module speaks. The decompiler produces one of
these; the router reads it; the recompiler consumes it. Keep it stable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

BLUEPRINT_VERSION = "1.0"


# ── enums ────────────────────────────────────────────────────────────────────


class CameraMove(str, Enum):
    STATIC = "static"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    HANDHELD = "handheld"
    UNKNOWN = "unknown"


class ShotScale(str, Enum):
    ECU = "extreme_close_up"
    CU = "close_up"
    MCU = "medium_close_up"
    MS = "medium_shot"
    MLS = "medium_long_shot"
    WS = "wide_shot"
    EWS = "extreme_wide_shot"
    UNKNOWN = "unknown"


class TransitionType(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE = "fade"
    UNKNOWN = "unknown"


class ShotRole(str, Enum):
    """Narrative function. Assigned by the VLM pass, not by CV."""

    HOOK = "hook"
    PROBLEM = "problem"
    DEMO = "demo"
    PROOF = "proof"
    CTA = "cta"
    BROLL = "broll"
    UNKNOWN = "unknown"


class SlotKind(str, Enum):
    PRODUCT = "product"
    MODEL = "model"
    BACKGROUND = "background"
    TEXT = "text"
    BRAND = "brand"


# ── provenance ───────────────────────────────────────────────────────────────


class Provenance(BaseModel):
    """Which backend produced a value, and how much to trust it.

    Every derived field carries one of these. Without it there is no way to
    tell a TransNetV2 shot boundary from a crude histogram guess, and the QC
    gate needs that distinction to decide what to re-verify.
    """

    backend: str
    confidence: float = Field(ge=0.0, le=1.0)
    note: str | None = None


# ── source ───────────────────────────────────────────────────────────────────


class SourceMeta(BaseModel):
    path: str
    duration: float
    fps: float
    width: int
    height: int
    aspect: str
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    size_bytes: int | None = None

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


# ── shot-level ───────────────────────────────────────────────────────────────


class Camera(BaseModel):
    move: CameraMove = CameraMove.UNKNOWN
    # normalised magnitude of the dominant motion, 0..1
    speed: float = 0.0
    # high-frequency jitter, 0..1; >0.35 reads as handheld
    shake: float = 0.0
    provenance: Provenance | None = None


class Subject(BaseModel):
    slot: str | None = None
    kind: SlotKind | None = None
    description: str | None = None
    action: str | None = None


class Overlay(BaseModel):
    text: str
    # normalised xyxy in 0..1 so it survives any resize
    bbox: tuple[float, float, float, float] | None = None
    t_in: float
    t_out: float
    anim: str | None = None
    provenance: Provenance | None = None


class Keyframes(BaseModel):
    """Extracted stills. Paths are relative to the job directory."""

    first: str | None = None
    mid: str | None = None
    last: str | None = None
    grid: str | None = None


class Shot(BaseModel):
    id: str
    index: int
    t_in: float
    t_out: float

    role: ShotRole = ShotRole.UNKNOWN
    shot_scale: ShotScale = ShotScale.UNKNOWN
    camera: Camera = Field(default_factory=Camera)
    subject: Subject = Field(default_factory=Subject)
    background: Subject = Field(default_factory=Subject)
    lighting: str | None = None

    overlays: list[Overlay] = Field(default_factory=list)
    transition_in: TransitionType = TransitionType.CUT
    transition_out: TransitionType = TransitionType.CUT

    keyframes: Keyframes = Field(default_factory=Keyframes)
    provenance: Provenance | None = None

    @property
    def duration(self) -> float:
        return round(self.t_out - self.t_in, 4)


# ── audio ────────────────────────────────────────────────────────────────────


class VOSegment(BaseModel):
    text: str
    t_in: float
    t_out: float
    speaker: str | None = None


class AudioTrack(BaseModel):
    bpm: float | None = None
    beat_grid: list[float] = Field(default_factory=list)
    vo: list[VOSegment] = Field(default_factory=list)
    integrated_lufs: float | None = None
    provenance: Provenance | None = None

    @property
    def wps(self) -> float | None:
        """Words per second across the VO. Drives pacing of the new script."""
        if not self.vo:
            return None
        words = sum(len(s.text.split()) for s in self.vo)
        span = sum(s.t_out - s.t_in for s in self.vo)
        return round(words / span, 3) if span > 0 else None


# ── slots ────────────────────────────────────────────────────────────────────


class Slot(BaseModel):
    """A swappable element. This is what the user fills in."""

    name: str
    kind: SlotKind
    description: str | None = None
    shot_ids: list[str] = Field(default_factory=list)
    # populated at recompile time
    filled_with: str | None = None


# ── analysis report ──────────────────────────────────────────────────────────


class AnalysisReport(BaseModel):
    """What ran, what didn't, and how long it took."""

    backends_used: dict[str, str] = Field(default_factory=dict)
    backends_missing: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ── blueprint ────────────────────────────────────────────────────────────────


class Blueprint(BaseModel):
    version: Literal["1.0"] = BLUEPRINT_VERSION
    source: SourceMeta
    shots: list[Shot] = Field(default_factory=list)
    audio: AudioTrack = Field(default_factory=AudioTrack)
    slots: list[Slot] = Field(default_factory=list)
    analysis: AnalysisReport = Field(default_factory=AnalysisReport)

    @property
    def cut_count(self) -> int:
        """Cuts, not shots. N shots means N-1 internal cuts."""
        return max(0, len(self.shots) - 1)

    @property
    def mean_shot_duration(self) -> float:
        if not self.shots:
            return 0.0
        return round(sum(s.duration for s in self.shots) / len(self.shots), 4)

    def shot_at(self, t: float) -> Shot | None:
        for s in self.shots:
            if s.t_in <= t < s.t_out:
                return s
        return None
