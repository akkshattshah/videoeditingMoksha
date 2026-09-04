"""Turn a reference blueprint + chosen elements into a plan the user approves.

The plan exists so the user can see the pipeline understood them *before* any
money is spent. Its most important field is `generation_prompt`: that string is
what actually gets sent to the video model, so the user approves the real
instruction rather than a summary of it.

Plans are revisable. The user replies in plain English ("no, keep the
background, just make it sunset") and the plan is rewritten.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import providers
from .schema import Blueprint

MAX_CONTACT_SHOTS = 8


@dataclass
class Change:
    target: str = ""        # "hair", "top and leggings"
    from_desc: str = ""     # what it is now
    to_desc: str = ""       # what it becomes
    source: str = ""        # element filename, or "instruction"
    shots: list[str] = field(default_factory=list)


@dataclass
class Plan:
    scene: str = ""
    changes: list[Change] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    generation_prompt: str = ""
    model_used: str = ""
    revision: int = 0
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "changes": [c.__dict__ for c in self.changes],
            "preserved": self.preserved,
            "concerns": self.concerns,
            "generation_prompt": self.generation_prompt,
            "model_used": self.model_used,
            "revision": self.revision,
        }


SCHEMA_HINT = """Reply with ONLY a JSON object, no prose, no markdown fence:

{
  "scene": "one sentence describing the reference footage as shot",
  "changes": [
    {"target":"what changes","from_desc":"how it looks now",
     "to_desc":"how it will look","source":"element filename or 'instruction'",
     "shots":["s01","s03"]}
  ],
  "preserved": ["things that must NOT change"],
  "concerns": ["risks worth warning the user about, [] if none"],
  "generation_prompt": "the exact instruction to send to the video editing model"
}"""

RULES = """Rules for generation_prompt:
- Describe ONLY what changes. Anything not mentioned is preserved automatically.
- Never describe the camera, cuts, pacing or edit structure.
- Name the physical qualities that make a change look real (fabric texture,
  how hair falls, how light behaves on the new surface).
- End by stating that everything else stays exactly as it is.
- Two or three sentences. No lists, no markdown."""


def _contact_sheet(bp: Blueprint, job_dir: Path) -> Path | None:
    """One strip of shot keyframes - cheaper and clearer than N separate images."""
    import cv2
    import numpy as np

    frames = []
    for s in bp.shots[:MAX_CONTACT_SHOTS]:
        if not s.keyframes.mid:
            continue
        p = job_dir / s.keyframes.mid
        if not p.exists():
            continue
        img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (180, 320))
        cv2.rectangle(img, (0, 0), (58, 22), (0, 0, 0), -1)
        cv2.putText(img, s.id, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        frames.append(img)

    if not frames:
        return None
    out = job_dir / "contact.jpg"
    cv2.imwrite(str(out), np.hstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def _shot_table(bp: Blueprint) -> str:
    rows = [
        f"  {s.id}  {s.t_in:5.2f}-{s.t_out:5.2f}s  camera={s.camera.move.value}"
        for s in bp.shots
    ]
    return "\n".join(rows)


def _elements_block(elements: list[dict]) -> str:
    if not elements:
        return "  (none - the user gave instructions only)"
    return "\n".join(
        f"  - {e['kind']}: '{e['name']}'"
        + (f"  [image {i + 1} after the shot strip]" if e.get("path") else "")
        for i, e in enumerate(elements)
    )


def _parse(text: str) -> dict:
    """Pull JSON out of a model reply that may be fenced or padded with prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"no JSON object in reply: {text[:220]}")


def _build(data: dict, model: str, revision: int, raw: str) -> Plan:
    changes = []
    for c in data.get("changes") or []:
        if not isinstance(c, dict):
            continue
        changes.append(Change(
            target=str(c.get("target", ""))[:120],
            from_desc=str(c.get("from_desc", ""))[:200],
            to_desc=str(c.get("to_desc", ""))[:200],
            source=str(c.get("source", ""))[:120],
            shots=[str(s) for s in (c.get("shots") or [])][:20],
        ))
    return Plan(
        scene=str(data.get("scene", ""))[:400],
        changes=changes,
        preserved=[str(x)[:160] for x in (data.get("preserved") or [])][:10],
        concerns=[str(x)[:240] for x in (data.get("concerns") or [])][:6],
        generation_prompt=str(data.get("generation_prompt", "")).strip()[:1200],
        model_used=model,
        revision=revision,
        raw=raw,
    )


def make_plan(bp: Blueprint, job_dir: Path, elements: list[dict],
              user_prompt: str = "") -> Plan:
    """First pass: read the reference and the elements, propose a plan."""
    sheet = _contact_sheet(bp, job_dir)
    images: list[Path] = ([sheet] if sheet else []) + [
        Path(e["path"]) for e in elements if e.get("path") and Path(e["path"]).exists()
    ]

    prompt = f"""You are planning an edit to an existing video. The edit will be
performed by a video model that preserves everything it is not told to change.

THE FIRST IMAGE is a strip of keyframes from the reference video, one per shot,
labelled s01, s02, ... {"THE IMAGES AFTER IT are the replacement elements the user uploaded, in the order listed below." if len(images) > 1 else ""}

REFERENCE: {bp.source.duration:.1f}s, {bp.source.width}x{bp.source.height}, \
{len(bp.shots)} shots, {bp.cut_count} cuts
{_shot_table(bp)}

ELEMENTS THE USER UPLOADED:
{_elements_block(elements)}

THE USER'S OWN INSTRUCTIONS:
  {user_prompt.strip() or "(none given - infer intent from the elements above)"}

Write a plan describing exactly what you will change and what you will leave
alone. Be specific about what you actually see in the reference - name the
person, clothing, setting and lighting as they appear. Say which shots each
change affects, using the s01/s02 labels. Raise a concern if a requested change
looks risky or if an uploaded element never appears in the footage.

{RULES}

{SCHEMA_HINT}"""

    text, model = providers.chat_vision(prompt, images)
    return _build(_parse(text), model, 0, text)


def revise_plan(plan: Plan, feedback: str, bp: Blueprint, job_dir: Path,
                elements: list[dict], user_prompt: str = "") -> Plan:
    """Rewrite a plan from the user's correction, in their words."""
    sheet = job_dir / "contact.jpg"
    images: list[Path] = ([sheet] if sheet.exists() else []) + [
        Path(e["path"]) for e in elements if e.get("path") and Path(e["path"]).exists()
    ]

    prompt = f"""You previously produced this plan for editing a video:

{json.dumps(plan.to_dict(), indent=2)}

The user has corrected you:

  "{feedback.strip()}"

Rewrite the plan so it matches what they actually want. Apply their correction
literally - if they say something must stay unchanged, remove it from `changes`
and add it to `preserved`. Keep every other part of the plan intact; do not
invent new changes they did not ask for.

The reference is {bp.source.duration:.1f}s, {len(bp.shots)} shots:
{_shot_table(bp)}

{RULES}

{SCHEMA_HINT}"""

    text, model = providers.chat_vision(prompt, images)
    return _build(_parse(text), model, plan.revision + 1, text)
