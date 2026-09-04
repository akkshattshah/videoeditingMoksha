"""Unit economics and volume projections.

Two profiles:

  LOCAL  - pilot. Runs on your own machine. No cloud hosting, no managed
           services, free VLM, cheapest image model, 480p output. The point is
           to prove the machinery works, not to look good.
  CLOUD  - production. Modal workers, best-in-class models, 720p+, managed
           Postgres and object storage.

Rates verified 2026-08-10 where marked. Estimated line items are flagged so a
margin figure is never quoted as if every input were measured.

Sources:
  video      OpenRouter model pages / fal model pages
  image      OpenRouter /api/v1/models?output_modalities=image, billed per
             output token at ~1290 tokens per 1024px image
  vlm        OpenRouter /api/v1/models; 14 models carry a :free tier, five of
             them vision-capable (google/gemma-4-31b-it:free and others)
  compute    modal.com/pricing
  tts        elevenlabs.io/pricing/api
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import pricing

# ── verified third-party rates ───────────────────────────────────────────────

MODAL_L4_PER_SEC = 0.000222

# Per 1024px image, derived from OpenRouter per-output-token rates.
IMAGE_MODELS = {
    "flux.2-klein-4b": 0.0044,          # cheapest usable
    "krea-2-medium-turbo": 0.0046,
    "flux.2-pro": 0.0094,
    "seedream-4.5": 0.0124,
    "gemini-2.5-flash-image": 0.0387,   # "Nano Banana"
    "gemini-3-pro-image": 0.1548,
}

# Cost of one understand pass (~15k in / 3k out).
VLM_FREE = 0.0            # google/gemma-4-31b-it:free  (vision, 262k ctx)
VLM_CHEAP = 0.0014        # google/gemini-2.5-flash-lite:batch
VLM_BEST = 0.20           # Gemini perception + Claude structured authoring

TTS_FLASH_PER_1K = 0.05
CHARS_PER_SECOND = 13.5


class Tier(str, Enum):
    QUALITY = "quality"     # Aleph in-context edit
    BALANCED = "balanced"   # per-shot on Seedance 2.0
    BUDGET = "budget"       # per-shot on Seedance 1.5 Pro


TIER_MODEL = {
    Tier.BALANCED: "seedance-2.0",
    Tier.BUDGET: "seedance-1.5-pro",
}


@dataclass(frozen=True)
class Profile:
    name: str
    resolution: str
    image_model: str
    vlm_cost: float
    # decompile: own CPU is free; cloud bills GPU seconds
    local_compute: bool
    render_cost: float
    storage_cost: float
    include_vo: bool

    def decompile_cost(self, duration: float) -> tuple[float, str]:
        if self.local_compute:
            return 0.0, "own CPU, ~5s for 16s of video"
        gpu_sec = max(45.0, duration * 3.0)
        return round(gpu_sec * MODAL_L4_PER_SEC, 4), f"{gpu_sec:.0f}s on Modal L4"


LOCAL = Profile(
    name="local",
    resolution="480p",
    image_model="flux.2-klein-4b",
    vlm_cost=VLM_FREE,
    local_compute=True,
    render_cost=0.0,
    storage_cost=0.0,
    include_vo=False,   # VO adds nothing to a structural-fidelity test
)

CLOUD = Profile(
    name="cloud",
    resolution="720p",
    image_model="gemini-2.5-flash-image",
    vlm_cost=VLM_BEST,
    local_compute=False,
    render_cost=0.03,
    storage_cost=0.02,
    include_vo=True,
)

PROFILES = {"local": LOCAL, "cloud": CLOUD}


@dataclass
class LineItem:
    name: str
    cost: float
    verified: bool = True
    note: str = ""


@dataclass
class JobCost:
    tier: Tier
    profile: Profile
    duration: float
    shots: int
    items: list[LineItem] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(i.cost for i in self.items), 4)

    @property
    def dominant(self) -> LineItem:
        return max(self.items, key=lambda i: i.cost)


def job_cost(
    duration: float = 30.0,
    shots: int = 12,
    tier: Tier = Tier.QUALITY,
    profile: Profile = CLOUD,
    keyframe_attempts: float = 2.0,
) -> JobCost:
    """Full cost of one generated video."""
    items: list[LineItem] = []

    dec_cost, dec_note = profile.decompile_cost(duration)
    items.append(LineItem("decompile", dec_cost, True, dec_note))

    vlm_note = ("gemma-4-31b-it:free" if profile.vlm_cost == 0
                else "Gemini + Claude" if profile.vlm_cost >= 0.1
                else "gemini-2.5-flash-lite")
    items.append(LineItem("VLM understand", profile.vlm_cost,
                          profile.vlm_cost != VLM_BEST, vlm_note))

    img_rate = IMAGE_MODELS[profile.image_model]

    if tier is Tier.QUALITY:
        model = pricing.MODELS["aleph-2"]
        items.append(LineItem(
            f"anchor frame ({profile.image_model})",
            round(img_rate * keyframe_attempts, 4),
            True, f"{keyframe_attempts:g} attempts, one frame drives the edit",
        ))
        gen = model.cost_for(duration)
        items.append(LineItem("Aleph 2.0 edit", round(gen, 4), True,
                              f"{duration:.0f}s x ${model.per_second}/s, single call"))
        items.append(LineItem("retry allowance", round(gen * 0.15, 4), False,
                              "15% on the edit call"))
    else:
        model = pricing.MODELS[TIER_MODEL[tier]]
        per_shot = duration / max(shots, 1)
        items.append(LineItem(
            f"keyframes ({profile.image_model})",
            round(img_rate * keyframe_attempts * shots, 4),
            True, f"{shots} shots x {keyframe_attempts:g} attempts",
        ))
        gen = sum(model.cost_for(per_shot, profile.resolution) for _ in range(shots))
        billed = shots * max(per_shot, model.min_seconds)
        items.append(LineItem(
            f"{model.label} @ {profile.resolution}", round(gen, 4), True,
            f"{shots} clips, {billed:.0f}s billed (min {model.min_seconds:g}s/clip)",
        ))
        items.append(LineItem("retry allowance",
                              round(gen * (pricing.RETRY_FACTOR - 1.0), 4), False,
                              f"{(pricing.RETRY_FACTOR - 1) * 100:.0f}% fan-out"))

    if profile.include_vo:
        chars = duration * CHARS_PER_SECOND
        items.append(LineItem("VO (ElevenLabs Flash)",
                              round(chars / 1000 * TTS_FLASH_PER_1K, 4),
                              True, f"~{chars:.0f} chars"))
    if profile.render_cost or profile.storage_cost:
        items.append(LineItem("render + storage",
                              round(profile.render_cost + profile.storage_cost, 4),
                              False, "Remotion Lambda + R2"))
    else:
        items.append(LineItem("render + storage", 0.0, True, "local ffmpeg + disk"))

    return JobCost(tier=tier, profile=profile, duration=duration,
                   shots=shots, items=items)


# ── fixed monthly ────────────────────────────────────────────────────────────

FIXED_LOCAL: list[LineItem] = [
    LineItem("hosting", 0.0, True, "runs on your machine"),
    LineItem("database", 0.0, True, "SQLite file"),
    LineItem("storage", 0.0, True, "local disk"),
    LineItem("orchestration", 0.0, True, "plain Python, no Temporal"),
    LineItem("GPU compute", 0.0, True, "CPU analyzers only"),
]

FIXED_CLOUD: list[LineItem] = [
    LineItem("Modal (beyond $30 credit)", 80.0, False, "GPU workers"),
    LineItem("Postgres (Neon / Supabase)", 25.0, True, ""),
    LineItem("Cloudflare R2", 25.0, False, "no egress fee"),
    LineItem("Vercel Pro", 20.0, True, "frontend"),
    LineItem("Temporal Cloud", 100.0, False, "or $0 self-hosted"),
    LineItem("ElevenLabs Creator", 22.0, True, "120k chars/mo"),
    LineItem("domain, email, monitoring", 25.0, False, ""),
]


def fixed_monthly(profile: Profile) -> list[LineItem]:
    return FIXED_LOCAL if profile.local_compute else FIXED_CLOUD


def fixed_total(profile: Profile) -> float:
    return round(sum(i.cost for i in fixed_monthly(profile)), 2)


# ── validation spend ─────────────────────────────────────────────────────────


@dataclass
class BuildLine:
    name: str
    detail: str
    cost: float


def validation_plan(profile: Profile = LOCAL,
                    duration: float = 10.0, shots: int = 4) -> list[BuildLine]:
    """What it costs to answer 'does the machinery work?'

    Deliberately scoped to short references. A 10s clip with 4 cuts exercises
    every stage of the pipeline exactly as a 30s one does, at a third the cost.
    """
    aleph_runs = 20
    aleph_each = job_cost(duration, shots, Tier.QUALITY, profile).total
    aleph_cost = aleph_runs * aleph_each

    shootout = 6 * 8            # 6 models x 8 reference shots
    shootout_each = 5.0 * pricing.MODELS["seedance-1.5-pro"].cost_per_second(
        profile.resolution)
    shootout_cost = shootout * shootout_each

    iterations = 200
    iter_each = job_cost(duration, shots, Tier.BUDGET, profile).total
    iter_cost = iterations * iter_each

    subtotal = aleph_cost + shootout_cost + iter_cost
    return [
        BuildLine("Aleph cut-ceiling test",
                  f"{aleph_runs} x {duration:.0f}s edits @ ${aleph_each:.2f}",
                  round(aleph_cost, 2)),
        BuildLine("model shootout",
                  f"{shootout} clips @ {profile.resolution}", round(shootout_cost, 2)),
        BuildLine("pipeline iteration",
                  f"{iterations} budget jobs @ ${iter_each:.2f}", round(iter_cost, 2)),
        BuildLine("slack / failed runs", "20%", round(subtotal * 0.2, 2)),
    ]


def validation_total(profile: Profile = LOCAL, duration: float = 10.0,
                     shots: int = 4) -> float:
    return round(sum(l.cost for l in validation_plan(profile, duration, shots)), 2)


def margin(price_per_job: float, tier: Tier = Tier.QUALITY,
           duration: float = 30.0, shots: int = 12,
           profile: Profile = CLOUD) -> dict:
    cogs = job_cost(duration, shots, tier, profile).total
    gross = price_per_job - cogs
    return {
        "price": price_per_job,
        "cogs": cogs,
        "gross": round(gross, 2),
        "margin_pct": round(gross / price_per_job * 100, 1) if price_per_job else 0.0,
    }
