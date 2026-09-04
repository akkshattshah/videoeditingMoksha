"""Local UI for the reference-video testbench.

    python ui/server.py          ->  http://127.0.0.1:7860

Upload a reference video, attach typed elements (clothing, model, background,
...), optionally add custom instructions. Analysis is free; generation is
gated behind an explicit confirm and always shows the cost first.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

from vre import config, pricing  # noqa: E402
from vre.decompile import decompile  # noqa: E402
from vre.probe import probe  # noqa: E402
from vre.router import route  # noqa: E402

HERE = Path(__file__).parent
UPLOADS = HERE / "uploads"
ELEMENTS = UPLOADS / "elements"
WORK = UPLOADS / "work"

VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

ELEMENT_TYPES = [
    {"id": "clothing", "label": "Clothing / Outfit"},
    {"id": "product", "label": "Product"},
    {"id": "model", "label": "Model (human)"},
    {"id": "hair", "label": "Hair / Styling"},
    {"id": "background", "label": "Background / Location"},
    {"id": "prop", "label": "Prop"},
    {"id": "logo", "label": "Logo / Brand mark"},
    {"id": "other", "label": "Other"},
]

# $/second. Verified 2026-08-10 against the Runway credit table and
# OpenRouter model pages.
MODELS = [
    {"id": "h3", "model": "minimax/hailuo-3", "rate": 0.13,
     "label": "MiniMax H3", "note": "cheapest; proven on colour + hair swaps"},
    {"id": "seedance", "model": "bytedance/seedance-2.5", "rate": 0.20,
     "label": "Seedance 2.5", "note": "video editing + extension"},
    {"id": "aleph", "model": "runway/aleph-2", "rate": 0.28,
     "label": "Runway Aleph 2.0", "note": "purpose-built in-context editor"},
]

app = FastAPI(title="vre")

# In-memory session. Single user, single machine - a dict is enough.
STATE: dict[str, Any] = {"reference": None, "elements": {}, "prompt": "",
                         "analysis": None, "plan": None, "plan_history": []}
JOB: dict[str, Any] = {"status": "idle", "log": [], "result": None,
                       "error": None, "cost": None, "model": None}


def _reset_dirs() -> None:
    for d in (UPLOADS, ELEMENTS, WORK):
        d.mkdir(parents=True, exist_ok=True)


def _public_state() -> dict:
    return {
        "reference": STATE["reference"],
        "elements": list(STATE["elements"].values()),
        "prompt": STATE["prompt"],
        "analysis": STATE["analysis"],
        "plan": STATE.get("plan"),
        "element_types": ELEMENT_TYPES,
        "models": MODELS,
        "has_key": config.has("OPENROUTER_API_KEY"),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/state")
def get_state() -> dict:
    return _public_state()


@app.get("/api/credits")
def get_credits() -> dict:
    try:
        from vre import providers

        total, used = providers.credits()
        return {"ok": True, "total": total, "used": round(used, 4),
                "remaining": round(total - used, 4)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


@app.post("/api/reference")
async def upload_reference(file: UploadFile = File(...)) -> dict:
    _reset_dirs()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported video type '{ext}'")

    dest = UPLOADS / f"reference{ext}"
    for old in UPLOADS.glob("reference.*"):
        old.unlink(missing_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        src = probe(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read video: {exc}") from exc

    STATE["reference"] = {
        "name": file.filename,
        "path": str(dest),
        "url": f"/api/file/reference{ext}",
        "duration": src.duration,
        "width": src.width,
        "height": src.height,
        "fps": round(src.fps, 2),
        "aspect": src.aspect,
        "has_audio": src.has_audio,
        "size_mb": round(dest.stat().st_size / 1e6, 2),
    }
    STATE["analysis"] = None
    return _public_state()


@app.post("/api/element")
async def add_element(kind: str = Form(...), file: UploadFile = File(...)) -> dict:
    _reset_dirs()
    if kind not in {t["id"] for t in ELEMENT_TYPES}:
        raise HTTPException(400, f"unknown element type '{kind}'")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXT:
        raise HTTPException(400, f"unsupported image type '{ext}'")

    eid = uuid.uuid4().hex[:8]
    dest = ELEMENTS / f"{eid}_{kind}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    STATE["elements"][eid] = {
        "id": eid, "kind": kind, "name": file.filename,
        "path": str(dest), "url": f"/api/file/elements/{dest.name}",
        "size_kb": round(dest.stat().st_size / 1024),
    }
    return _public_state()


@app.delete("/api/element/{eid}")
def remove_element(eid: str) -> dict:
    el = STATE["elements"].pop(eid, None)
    if el:
        Path(el["path"]).unlink(missing_ok=True)
    return _public_state()


@app.post("/api/prompt")
async def set_prompt(prompt: str = Form("")) -> dict:
    STATE["prompt"] = prompt.strip()
    return _public_state()


@app.get("/api/file/{path:path}")
def serve_file(path: str) -> FileResponse:
    target = (UPLOADS / path).resolve()
    if not str(target).startswith(str(UPLOADS.resolve())) or not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.post("/api/analyze")
def analyze(seconds: float = Form(5.0), start: float = Form(0.0),
            model: str = Form("h3")) -> dict:
    """Decompile the reference and price the job. Costs nothing."""
    ref = STATE["reference"]
    if not ref:
        raise HTTPException(400, "upload a reference video first")

    src_path = Path(ref["path"])
    seg = src_path
    if seconds > 0 and seconds < ref["duration"]:
        from vre.probe import run as ffrun

        seg = WORK / "segment.mp4"
        seg.parent.mkdir(parents=True, exist_ok=True)
        r = ffrun(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                   "-i", str(src_path), "-t", f"{seconds:.3f}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", str(seg)])
        if r.returncode != 0:
            raise HTTPException(500, f"trim failed: {r.stderr[:200]}")

    # Keyframes are required: the planner builds its contact sheet from them.
    bp = decompile(seg, WORK / "ref", do_frames=True)
    dec = route(bp)
    spec = next(m for m in MODELS if m["id"] == model)
    billed = bp.source.duration
    cost = round(billed * spec["rate"], 4)

    STATE["_blueprint"] = bp
    STATE["plan"] = None
    STATE["_plan"] = None
    STATE["plan_history"] = []
    STATE["analysis"] = {
        "duration": bp.source.duration,
        "shots": [
            {"id": s.id, "t_in": round(s.t_in, 2), "t_out": round(s.t_out, 2),
             "camera": s.camera.move.value,
             "transition": s.transition_out.value}
            for s in bp.shots
        ],
        "cut_count": bp.cut_count,
        "route": dec.path.value,
        "warnings": dec.warnings + bp.analysis.warnings[:2],
        "bpm": bp.audio.bpm,
        "model": spec,
        "cost": cost,
        "segment_seconds": round(billed, 2),
    }
    return _public_state()


@app.post("/api/plan")
def build_plan(feedback: str = Form("")) -> dict:
    """Propose a plan, or revise the current one from user feedback. ~$0.001."""
    if not STATE["reference"]:
        raise HTTPException(400, "upload a reference video first")
    if not STATE["analysis"]:
        raise HTTPException(400, "run Analyse first")

    from vre import planner
    from vre.decompile import decompile as _dc

    bp = STATE.get("_blueprint")
    if bp is None:
        bp = _dc(WORK / "segment.mp4" if (WORK / "segment.mp4").exists()
                 else Path(STATE["reference"]["path"]), WORK / "ref")
        STATE["_blueprint"] = bp

    elements = list(STATE["elements"].values())
    try:
        if feedback.strip() and STATE.get("_plan"):
            plan = planner.revise_plan(STATE["_plan"], feedback, bp,
                                       WORK / "ref", elements, STATE["prompt"])
        else:
            plan = planner.make_plan(bp, WORK / "ref", elements, STATE["prompt"])
    except Exception as exc:
        raise HTTPException(500, f"planner failed: {exc}"[:300]) from exc

    STATE["_plan"] = plan
    STATE["plan"] = plan.to_dict()
    if feedback.strip():
        STATE.setdefault("plan_history", []).append(feedback.strip())
    STATE["plan"]["history"] = STATE.get("plan_history", [])
    return _public_state()


@app.post("/api/plan/prompt")
async def override_prompt(generation_prompt: str = Form(...)) -> dict:
    """Let the user hand-edit the exact instruction before it is sent."""
    plan = STATE.get("_plan")
    if not plan:
        raise HTTPException(400, "no plan yet")
    plan.generation_prompt = generation_prompt.strip()[:1200]
    STATE["plan"] = plan.to_dict()
    STATE["plan"]["history"] = STATE.get("plan_history", [])
    return _public_state()


@app.post("/api/generate")
def start_generate(model: str = Form("h3")) -> dict:
    """Kick off the real edit in a worker thread. This spends money."""
    if not STATE.get("_plan"):
        raise HTTPException(400, "approve a plan first")
    if JOB.get("status") == "running":
        raise HTTPException(409, "a generation is already running")

    spec = next(m for m in MODELS if m["id"] == model)
    seg = WORK / "segment.mp4"
    if not seg.exists():
        seg = Path(STATE["reference"]["path"])

    JOB.update({"status": "running", "log": [], "result": None, "error": None,
                "cost": None, "model": spec["label"]})
    t = threading.Thread(target=_run_generation,
                         args=(spec, seg, STATE["_plan"].generation_prompt,
                               list(STATE["elements"].values())), daemon=True)
    t.start()
    return {"started": True}


@app.get("/api/generate")
def generate_status() -> dict:
    return {k: JOB.get(k) for k in
            ("status", "log", "result", "error", "cost", "model")}


def _log(msg: str) -> None:
    JOB.setdefault("log", []).append(msg)


def _run_generation(spec: dict, segment: Path, prompt: str,
                    elements: list[dict]) -> None:
    from vre import providers, tunnel

    out = WORK / "gen"
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "upload"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(segment, stage / "reference.mp4")
    for e in elements:
        p = Path(e["path"])
        if p.exists():
            shutil.copy2(p, stage / p.name)

    try:
        before = providers.remaining()
        _log("opening public tunnel")
        with tunnel.serve_public(stage, verbose=False) as base:
            _log("submitting to " + spec["model"])
            job = providers.submit_video(
                spec["model"], prompt,
                video_url=tunnel.url_for(base, "reference.mp4"),
                image_urls=[tunnel.url_for(base, Path(e["path"]).name)
                            for e in elements if Path(e["path"]).exists()],
            )
            _log(f"job {job.id} accepted")
            seen = {"s": ""}

            def tick(status, _):
                if status != seen["s"]:
                    seen["s"] = status
                    _log(f"status: {status}")

            payload = providers.poll_video(job, on_tick=tick)
            urls = providers.video_urls(payload)
            if not urls:
                raise RuntimeError("no output URL returned")
            _log("downloading result")
            dest = out / "result.mp4"
            providers.download(urls[0], dest)

        after = providers.remaining()
        JOB["cost"] = round(before - after, 4)
        JOB["result"] = "/api/file/work/gen/result.mp4"
        JOB["status"] = "done"
        _log(f"complete - spent ${JOB['cost']:.3f}")
    except Exception as exc:
        JOB["error"] = f"{type(exc).__name__}: {exc}"[:400]
        JOB["status"] = "failed"
        _log("failed: " + JOB["error"])


@app.post("/api/reset")
def reset() -> dict:
    for d in (ELEMENTS, WORK):
        shutil.rmtree(d, ignore_errors=True)
    for old in UPLOADS.glob("reference.*"):
        old.unlink(missing_ok=True)
    STATE.update({"reference": None, "elements": {}, "prompt": "",
                  "analysis": None, "plan": None, "plan_history": [],
                  "_plan": None, "_blueprint": None})
    JOB.update({"status": "idle", "log": [], "result": None,
                "error": None, "cost": None, "model": None})
    _reset_dirs()
    return _public_state()


@app.exception_handler(HTTPException)
def http_error(request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    _reset_dirs()
    # Bind 0.0.0.0 in a container: a process listening only on loopback is
    # unreachable from outside it. PORT is injected by the host in production
    # and falls back to 7860 for local runs.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    print(f"\n  vre UI  ->  http://127.0.0.1:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
