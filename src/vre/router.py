"""Path router.

Decides whether a reference video can go through the single-call in-context
editor (Aleph 2.0) or must be regenerated shot by shot. The thresholds encode
Aleph's published limits; the cut ceiling is deliberately conservative because
Runway documents multi-shot propagation without committing to a number.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from . import pricing
from .schema import Blueprint

# Aleph 2.0 published limits.
ALEPH_MAX_SECONDS = 30.0
ALEPH_MAX_HEIGHT = 1080
# Third-party reporting says ~10 cuts; Runway's own page gives no figure.
# Until benchmarked on real footage, treat this as unverified.
ALEPH_MAX_CUTS = 10
ALEPH_CUTS_UNVERIFIED = True

# Below this, boundary detection is too shaky to trust the cut count.
MIN_SHOT_CONFIDENCE = 0.55


class Path(str, Enum):
    ALEPH = "aleph"
    PER_SHOT = "per_shot"
    HYBRID = "hybrid"


class CostEstimate(BaseModel):
    path: Path
    model: str
    generation: float
    analysis: float
    with_retries: float
    detail: str


class RouterDecision(BaseModel):
    path: Path
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimates: list[CostEstimate] = Field(default_factory=list)

    @property
    def chosen(self) -> CostEstimate | None:
        return next((e for e in self.estimates if e.path == self.path), None)


def _aleph_estimate(bp: Blueprint) -> CostEstimate:
    model = pricing.MODELS[pricing.DEFAULT_EDITOR]
    gen = model.cost_for(bp.source.duration)
    analysis = pricing.COST_DECOMPILE + pricing.COST_UNDERSTAND
    return CostEstimate(
        path=Path.ALEPH,
        model=model.label,
        generation=round(gen, 4),
        analysis=round(analysis, 4),
        # A single call either lands or is re-run once; it does not fan out.
        with_retries=round(gen * 1.15 + analysis, 4),
        detail=f"{bp.source.duration:.1f}s x ${model.per_second}/s, one call",
    )


def _per_shot_estimate(bp: Blueprint, model_key: str, resolution: str) -> CostEstimate:
    model = pricing.MODELS[model_key]
    gen = sum(model.cost_for(s.duration, resolution) for s in bp.shots)
    billed = sum(max(s.duration, model.min_seconds) for s in bp.shots)
    analysis = pricing.COST_DECOMPILE + pricing.COST_UNDERSTAND
    return CostEstimate(
        path=Path.PER_SHOT,
        model=model.label,
        generation=round(gen, 4),
        analysis=round(analysis, 4),
        with_retries=round(gen * pricing.RETRY_FACTOR + analysis, 4),
        detail=(
            f"{len(bp.shots)} shots, {billed:.1f}s billed "
            f"(min {model.min_seconds:g}s/clip) @ {resolution}"
        ),
    )


def route(bp: Blueprint, resolution: str = "720p",
          generator: str = pricing.DEFAULT_GENERATOR) -> RouterDecision:
    reasons: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    dur = bp.source.duration
    cuts = bp.cut_count
    short_side = min(bp.source.width, bp.source.height)

    if dur <= ALEPH_MAX_SECONDS:
        reasons.append(f"duration {dur:.1f}s is within Aleph's {ALEPH_MAX_SECONDS:g}s limit")
    else:
        blockers.append(f"duration {dur:.1f}s exceeds Aleph's {ALEPH_MAX_SECONDS:g}s limit")

    if cuts <= ALEPH_MAX_CUTS:
        reasons.append(f"{cuts} cuts is within the {ALEPH_MAX_CUTS}-cut ceiling")
        if ALEPH_CUTS_UNVERIFIED and cuts > 4:
            warnings.append(
                f"{cuts}-cut propagation is unverified on real footage; "
                "the cut ceiling comes from third-party reporting, not Runway"
            )
    else:
        blockers.append(f"{cuts} cuts exceeds the {ALEPH_MAX_CUTS}-cut ceiling")

    if short_side <= ALEPH_MAX_HEIGHT:
        reasons.append(f"{bp.source.width}x{bp.source.height} is within 1080p")
    else:
        blockers.append(f"{bp.source.width}x{bp.source.height} exceeds 1080p")

    shot_conf = min(
        (s.provenance.confidence for s in bp.shots if s.provenance),
        default=0.0,
    )
    if bp.shots and shot_conf < MIN_SHOT_CONFIDENCE:
        warnings.append(
            f"shot detection confidence {shot_conf:.2f} is low — "
            "the cut count driving this decision may be wrong"
        )

    path = Path.ALEPH if not blockers else Path.PER_SHOT

    estimates = [
        _aleph_estimate(bp),
        _per_shot_estimate(bp, generator, resolution),
    ]
    if generator != pricing.BUDGET_GENERATOR:
        budget = _per_shot_estimate(bp, pricing.BUDGET_GENERATOR, resolution)
        budget.detail += "  [budget option]"
        estimates.append(budget)

    return RouterDecision(
        path=path,
        reasons=reasons,
        blockers=blockers,
        warnings=warnings,
        estimates=estimates,
    )
