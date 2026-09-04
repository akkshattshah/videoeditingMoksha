"""Environment configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get(key: str, default: str | None = None) -> str | None:
    _load()
    val = os.environ.get(key, default)
    return val.strip() if isinstance(val, str) else val


def require(key: str) -> str:
    val = get(key)
    if not val:
        raise RuntimeError(f"{key} is not set. Add it to {PROJECT_ROOT / '.env'}")
    return val


def has(key: str) -> bool:
    return bool(get(key))


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def credentials_status() -> dict[str, bool]:
    return {
        "OPENROUTER_API_KEY": has("OPENROUTER_API_KEY"),
        "FAL_KEY": has("FAL_KEY"),
        "REPLICATE_API_TOKEN": has("REPLICATE_API_TOKEN"),
        "KLING_ACCESS_KEY": has("KLING_ACCESS_KEY"),
        "ARK_API_KEY": has("ARK_API_KEY"),
    }
