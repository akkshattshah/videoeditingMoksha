"""Provider pricing, verified 2026-08-10.

Seedance tiers are derived from OpenRouter's published token formula
    tokens/sec = (H * W * 24) / 1024
anchored on each model's quoted 480p floor, which resolves to a clean
$0.00700 / 1K tokens for Seedance 2.0.

Re-verify before trusting these for billing. Rates move monthly.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_VERIFIED = "2026-08-10"

# (width, height) per tier label
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


@dataclass(frozen=True)
class VideoModel:
    id: str
    label: str
    provider: str
    # flat per-second rate, or None if token-billed
    per_second: float | None = None
    # $ per 1000 tokens, for token-billed models
    per_1k_tokens: float | None = None
    min_seconds: float = 0.0
    max_seconds: float = 15.0
    # image-to-video arena Elo, where measured
    i2v_elo: int | None = None
    # in-context editor rather than a generator
    is_editor: bool = False
    note: str = ""

    def cost_per_second(self, resolution: str = "720p") -> float:
        if self.per_second is not None:
            return self.per_second
        if self.per_1k_tokens is None:
            raise ValueError(f"{self.id} has no pricing")
        w, h = RESOLUTIONS[resolution]
        tokens_per_sec = (w * h * 24) / 1024
        return tokens_per_sec * self.per_1k_tokens / 1000.0

    def cost_for(self, seconds: float, resolution: str = "720p") -> float:
        billed = max(seconds, self.min_seconds)
        return round(billed * self.cost_per_second(resolution), 4)


MODELS: dict[str, VideoModel] = {
    "aleph-2": VideoModel(
        id="runway/aleph-2",
        label="Runway Aleph 2.0",
        provider="openrouter",
        per_second=0.28,
        max_seconds=30.0,
        is_editor=True,
        note="in-context editor; preserves untouched regions across shots",
    ),
    "seedance-2.0": VideoModel(
        id="bytedance/seedance-2.0",
        label="Seedance 2.0",
        provider="openrouter",
        per_1k_tokens=0.00700,
        min_seconds=4.0,
        i2v_elo=1198,
        note="#1 image-to-video; strong reference conditioning",
    ),
    "seedance-2.0-fast": VideoModel(
        id="bytedance/seedance-2.0-fast",
        label="Seedance 2.0 Fast",
        provider="openrouter",
        per_1k_tokens=0.00560,
        min_seconds=4.0,
    ),
    "seedance-1.5-pro": VideoModel(
        id="bytedance/seedance-1-5-pro",
        label="Seedance 1.5 Pro",
        provider="openrouter",
        per_1k_tokens=0.00240,
        min_seconds=4.0,
        i2v_elo=1000,
        note="cheapest usable tier",
    ),
    "wan-2.7": VideoModel(
        id="alibaba/wan-2.7",
        label="Wan 2.7",
        provider="openrouter",
        per_second=0.10,
        min_seconds=3.0,
        i2v_elo=1094,
        note="multi-reference-image conditioning; value pick",
    ),
    "kling-v3-pro": VideoModel(
        id="kwaivgi/kling-v3.0-pro",
        label="Kling v3.0 Pro",
        provider="openrouter",
        per_second=0.168,
        min_seconds=3.0,
        i2v_elo=1075,
        note="OpenRouter exposes audio-on only; fal is $0.112/s audio-off",
    ),
    "kling-v3-pro-fal": VideoModel(
        id="fal-ai/kling-video/v3/pro",
        label="Kling v3.0 Pro (fal, audio off)",
        provider="fal",
        per_second=0.112,
        min_seconds=3.0,
        i2v_elo=1075,
    ),
    "minimax-h3": VideoModel(
        id="minimax/hailuo-3",
        label="MiniMax H3",
        provider="openrouter",
        per_second=0.13,
        min_seconds=3.0,
        i2v_elo=1190,
    ),
}

DEFAULT_EDITOR = "aleph-2"
DEFAULT_GENERATOR = "seedance-2.0"
BUDGET_GENERATOR = "wan-2.7"

# Multiplier applied to generation cost to account for QC-gate retries.
RETRY_FACTOR = 1.4

# Analysis stages, roughly. Decompile is local; understand is API-billed.
COST_DECOMPILE = 0.02
COST_UNDERSTAND = 0.15
