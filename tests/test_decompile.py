"""Score the decompiler against the fixture's known ground truth.

Runs standalone (`python tests/test_decompile.py`) so it needs no pytest.
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vre.decompile import decompile  # noqa: E402
from vre.router import Path as RoutePath  # noqa: E402
from vre.router import route  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ref_6shot.mp4"
TRUTH = FIXTURE.with_suffix(".truth.json")

CUT_TOLERANCE = 0.20   # seconds
BPM_TOLERANCE = 4.0

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, f"{label}{'  ' + detail if detail else ''}"))


def main() -> int:
    if not FIXTURE.exists():
        print(f"fixture missing: {FIXTURE}\nrun: python tests/make_fixture.py")
        return 2

    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="vre_test_"))

    try:
        bp = decompile(FIXTURE, work)

        # ── source ───────────────────────────────────────────────────────────
        check(abs(bp.source.duration - truth["duration"]) < 0.1,
              "duration", f"{bp.source.duration} vs {truth['duration']}")
        check(bp.source.width == truth["width"] and bp.source.height == truth["height"],
              "resolution", f"{bp.source.width}x{bp.source.height}")

        # ── shot count ───────────────────────────────────────────────────────
        check(len(bp.shots) == truth["n_shots"],
              "shot count", f"{len(bp.shots)} vs {truth['n_shots']}")

        # ── cut placement ────────────────────────────────────────────────────
        found = [s.t_in for s in bp.shots[1:]]
        expected = truth["cuts"]
        matched = 0
        for exp in expected:
            nearest = min((abs(exp - f), f) for f in found) if found else (99, None)
            if nearest[0] <= CUT_TOLERANCE:
                matched += 1
        check(matched == len(expected),
              "cut placement", f"{matched}/{len(expected)} within {CUT_TOLERANCE}s")

        if found:
            worst = max(
                min(abs(exp - f) for f in found) for exp in expected
            )
            check(worst <= CUT_TOLERANCE, "worst cut error", f"{worst:.3f}s")

        # ── camera motion ────────────────────────────────────────────────────
        truth_moves = [s["camera"] for s in truth["shots"]]
        if len(bp.shots) == len(truth_moves):
            got = [s.camera.move.value for s in bp.shots]
            correct = sum(1 for a, b in zip(got, truth_moves) if a == b)
            check(correct == len(truth_moves),
                  "camera motion", f"{correct}/{len(truth_moves)}  got={got}")
        else:
            check(False, "camera motion", "skipped, shot count mismatch")

        # ── audio ────────────────────────────────────────────────────────────
        if bp.audio.bpm:
            err = abs(bp.audio.bpm - truth["bpm"])
            check(err <= BPM_TOLERANCE,
                  "bpm", f"{bp.audio.bpm:.1f} vs {truth['bpm']}  (err {err:.1f})")
        else:
            check(False, "bpm", "not detected")

        check(len(bp.audio.beat_grid) > 8,
              "beat grid", f"{len(bp.audio.beat_grid)} beats")

        # ── keyframes ────────────────────────────────────────────────────────
        missing = [s.id for s in bp.shots if not s.keyframes.first]
        check(not missing, "keyframes", f"missing for {missing}" if missing else "all shots")

        # ── router ───────────────────────────────────────────────────────────
        decision = route(bp)
        check(decision.path == RoutePath.ALEPH,
              "route", f"{decision.path.value} (16.5s / 5 cuts should be aleph)")
        check(not decision.blockers, "no blockers", str(decision.blockers))

        est = decision.chosen
        check(est is not None and est.generation > 0, "cost estimate",
              f"${est.generation:.2f}" if est else "none")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    for ok, label in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n  {len(results) - failed}/{len(results)} passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
