"""OpenRouter client for video editing and image editing.

Video generation uses POST /api/v1/videos - a different endpoint and shape
from chat/completions, and asynchronous. Contract verified 2026-08-10 by
probing the API with deliberately invalid requests (which cost nothing):

  * input_references[].video_url.url  - the source footage for v2v models
  * input_references[].image_url.url  - reference images, allowed alongside
  * both must be HTTPS; data URIs are rejected outright
  * runway/aleph-2 requires exactly one video_url reference
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import config

BASE = "https://openrouter.ai/api/v1"
POLL_INTERVAL = 5.0
POLL_TIMEOUT = 900.0


class ProviderError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.require('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/vre",
        "X-Title": "vre testbench",
    }


# ── account ──────────────────────────────────────────────────────────────────


def credits() -> tuple[float, float]:
    """(total_credits, total_usage) in dollars."""
    with httpx.Client(timeout=30) as c:
        r = c.get(f"{BASE}/credits", headers=_headers())
    if r.status_code != 200:
        raise ProviderError(f"credits: HTTP {r.status_code} {r.text[:200]}")
    d = r.json()["data"]
    return float(d["total_credits"]), float(d["total_usage"])


def remaining() -> float:
    total, used = credits()
    return round(total - used, 4)


# ── video ────────────────────────────────────────────────────────────────────


@dataclass
class VideoJob:
    id: str
    polling_url: str
    status: str
    raw: dict = field(default_factory=dict)


def submit_video(
    model: str,
    prompt: str,
    *,
    video_url: str | None = None,
    image_urls: list[str] | None = None,
    duration: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    seed: int | None = None,
) -> VideoJob:
    refs: list[dict] = []
    if video_url:
        refs.append({"type": "video_url", "video_url": {"url": video_url}})
    for u in image_urls or []:
        refs.append({"type": "image_url", "image_url": {"url": u}})

    body: dict = {"model": model, "prompt": prompt}
    if refs:
        body["input_references"] = refs
    if duration is not None:
        body["duration"] = duration
    if resolution:
        body["resolution"] = resolution
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if seed is not None:
        body["seed"] = seed

    with httpx.Client(timeout=120) as c:
        r = c.post(f"{BASE}/videos", headers=_headers(), json=body)
    if r.status_code not in (200, 201, 202):
        raise ProviderError(f"submit {model}: HTTP {r.status_code} {r.text[:400]}")

    d = r.json()
    d = d.get("data", d)
    job_id = d.get("id") or ""
    poll = d.get("polling_url") or (f"{BASE}/videos/{job_id}" if job_id else "")
    if not poll:
        raise ProviderError(f"no polling url in response: {r.text[:300]}")
    return VideoJob(id=job_id, polling_url=poll, status=d.get("status", "pending"),
                    raw=d)


def poll_video(job: VideoJob, timeout: float = POLL_TIMEOUT,
               on_tick=None) -> dict:
    """Block until the job finishes. Returns the completed payload.

    Network errors are retried rather than raised: the generation is already
    paid for by this point, and a dropped connection during a ten-minute poll
    must not throw the result away.
    """
    deadline = time.time() + timeout
    fails = 0
    with httpx.Client(timeout=60) as c:
        while time.time() < deadline:
            try:
                r = c.get(job.polling_url, headers=_headers())
            except Exception as exc:
                fails += 1
                if fails > 20:
                    raise ProviderError(
                        f"poll: {type(exc).__name__} x{fails}; job {job.id} may "
                        f"still be valid - re-poll {job.polling_url}") from exc
                time.sleep(min(POLL_INTERVAL * fails, 30))
                continue
            fails = 0
            if r.status_code >= 500 or r.status_code == 429:
                time.sleep(POLL_INTERVAL)
                continue
            if r.status_code != 200:
                raise ProviderError(f"poll: HTTP {r.status_code} {r.text[:250]}")
            d = r.json()
            d = d.get("data", d)
            status = (d.get("status") or "").lower()
            if on_tick:
                on_tick(status, int(time.time() - (deadline - timeout)))
            if status in ("completed", "succeeded", "success"):
                return d
            if status in ("failed", "error", "cancelled", "canceled"):
                raise ProviderError(f"job {status}: {str(d)[:400]}")
            time.sleep(POLL_INTERVAL)
    raise ProviderError(f"timed out after {timeout:.0f}s (job {job.id})")


def video_urls(payload: dict) -> list[str]:
    for key in ("unsigned_urls", "urls", "output", "videos"):
        v = payload.get(key)
        if isinstance(v, list) and v:
            out = []
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    u = item.get("url") or item.get("video_url")
                    if isinstance(u, dict):
                        u = u.get("url")
                    if u:
                        out.append(u)
            if out:
                return out
        if isinstance(v, str):
            return [v]
    return []


def download(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # The field is called `unsigned_urls` for a reason: results are served
    # from /api/v1/videos/{id}/content, which is an authenticated endpoint
    # rather than a pre-signed link. Fetching it bare returns 401.
    headers = _headers() if url.startswith(BASE) else {}

    with httpx.Client(timeout=300, follow_redirects=True) as c:
        with c.stream("GET", url, headers=headers) as r:
            if r.status_code != 200:
                raise ProviderError(f"download: HTTP {r.status_code} for {url[:90]}")
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(65536):
                    f.write(chunk)
    return dest


# ── image ────────────────────────────────────────────────────────────────────


def _data_uri(p: str | Path) -> str:
    p = Path(p)
    b64 = base64.b64encode(p.read_bytes()).decode()
    suffix = p.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


# Tried in order; the first that answers wins.
#
# Two traps learned the hard way: `:free` tiers 429 constantly from the shared
# upstream pool, and *reasoning* models (qwen3.7-flash) spend the entire token
# budget thinking and return finish_reason=length with empty content. These
# are cheap, non-reasoning, and reliable - about $0.002 a call.
VISION_MODELS = [
    "google/gemini-2.5-flash-lite",
    "google/gemma-3-27b-it",
    "amazon/nova-lite-v1",
]


def chat_vision(prompt: str, images: list[str | Path] | None = None,
                models: list[str] | None = None,
                max_tokens: int = 2500,
                temperature: float = 0.2) -> tuple[str, str]:
    """Ask a vision model a question about images. Returns (text, model_used)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in images or []:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})

    last = ""
    with httpx.Client(timeout=180) as c:
        for model in (models or VISION_MODELS):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Suppress chain-of-thought so the budget goes to the answer.
                "reasoning": {"exclude": True},
            }
            try:
                r = c.post(f"{BASE}/chat/completions", headers=_headers(), json=body)
            except Exception as exc:
                last = f"{model}: {type(exc).__name__}"
                continue
            if r.status_code != 200:
                last = f"{model}: HTTP {r.status_code} {r.text[:130]}"
                continue
            try:
                choice = r.json()["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason", "")
            except Exception:
                last = f"{model}: unparseable envelope {r.text[:120]}"
                continue
            if text:
                return text, model
            last = f"{model}: empty content (finish_reason={finish!r})"

    raise ProviderError(f"all vision models failed. last: {last}")


def edit_image(image: str | Path, prompt: str, model: str,
               extra_images: list[str | Path] | None = None,
               attempts: int = 3) -> bytes | None:
    """Edit an image via chat/completions. Used to build the anchor frame."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in [image, *(extra_images or [])]:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        suffix = Path(p).suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"}})

    body = {"model": model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"]}

    with httpx.Client(timeout=240) as c:
        for attempt in range(1, attempts + 1):
            try:
                r = c.post(f"{BASE}/chat/completions", headers=_headers(), json=body)
                if r.status_code == 429:
                    time.sleep(2.0 * attempt)
                    continue
                if r.status_code != 200:
                    print(f"      image HTTP {r.status_code}: {r.text[:160]}")
                    time.sleep(1.0 * attempt)
                    continue
                msg = r.json()["choices"][0]["message"]
            except Exception as exc:
                print(f"      image {type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(1.0 * attempt)
                continue

            imgs = msg.get("images") or []
            if imgs:
                url = imgs[0].get("image_url", {}).get("url", "")
                if "," in url:
                    return base64.b64decode(url.split(",", 1)[1])
            said = (msg.get("content") or "").strip().replace("\n", " ")
            print(f"      attempt {attempt}/{attempts} no image"
                  + (f" - {said[:100]!r}" if said else ""))
    return None
