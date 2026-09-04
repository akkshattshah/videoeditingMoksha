"""Score how faithfully an output preserved the reference's structure.

Decompile the output, diff its blueprint against the reference's. Cut count,
cut placement, duration and camera moves are all objective and measurable, so
"did it hold structure?" stops being a matter of opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Blueprint

CUT_TOLERANCE = 0.25    # seconds; a cut this close counts as preserved


@dataclass
class Score:
    name: str
    value: float          # 0..1
    detail: str
    weight: float = 1.0


@dataclass
class FidelityReport:
    scores: list[Score] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        if not self.scores:
            return 0.0
        total_w = sum(s.weight for s in self.scores)
        return round(sum(s.value * s.weight for s in self.scores) / total_w, 4)

    @property
    def verdict(self) -> str:
        o = self.overall
        if o >= 0.85:
            return "PASS - structure preserved"
        if o >= 0.65:
            return "MARGINAL - structure mostly held, inspect by eye"
        if o >= 0.40:
            return "WEAK - noticeable structural drift"
        return "FAIL - structure not preserved"


def _cut_times(bp: Blueprint) -> list[float]:
    return [s.t_in for s in bp.shots[1:]]


def compare(ref: Blueprint, out: Blueprint) -> FidelityReport:
    rep = FidelityReport()

    # ── duration ─────────────────────────────────────────────────────────────
    rd, od = ref.source.duration, out.source.duration
    drift = abs(rd - od)
    rep.scores.append(Score(
        "duration", max(0.0, 1.0 - drift / max(rd, 0.01)),
        f"{od:.2f}s vs {rd:.2f}s (drift {drift:.2f}s)", weight=1.5,
    ))

    # ── shot count ───────────────────────────────────────────────────────────
    rc, oc = len(ref.shots), len(out.shots)
    rep.scores.append(Score(
        "shot_count", 1.0 if rc == oc else max(0.0, 1.0 - abs(rc - oc) / max(rc, 1)),
        f"{oc} vs {rc}", weight=2.0,
    ))

    # ── cut placement ────────────────────────────────────────────────────────
    rcuts, ocuts = _cut_times(ref), _cut_times(out)
    if not rcuts:
        rep.scores.append(Score("cut_placement", 1.0 if not ocuts else 0.5,
                                "reference has no cuts", weight=1.0))
    else:
        matched, errors = 0, []
        for t in rcuts:
            if not ocuts:
                break
            err = min(abs(t - o) for o in ocuts)
            errors.append(err)
            if err <= CUT_TOLERANCE:
                matched += 1
        worst = max(errors) if errors else 99.0
        rep.scores.append(Score(
            "cut_placement", matched / len(rcuts),
            f"{matched}/{len(rcuts)} within {CUT_TOLERANCE}s (worst {worst:.2f}s)",
            weight=3.0,
        ))

    # ── camera moves ─────────────────────────────────────────────────────────
    if rc == oc and rc > 0:
        same = sum(1 for a, b in zip(ref.shots, out.shots)
                   if a.camera.move == b.camera.move)
        rep.scores.append(Score(
            "camera_moves", same / rc, f"{same}/{rc} matched", weight=1.5,
        ))
    else:
        rep.notes.append("camera moves not scored - shot counts differ")

    # ── resolution / aspect ──────────────────────────────────────────────────
    same_aspect = ref.source.aspect == out.source.aspect
    rep.scores.append(Score(
        "aspect", 1.0 if same_aspect else 0.0,
        f"{out.source.aspect} vs {ref.source.aspect}", weight=0.5,
    ))

    if oc < rc:
        rep.notes.append(
            f"output lost {rc - oc} shot(s) - the model may have merged or "
            "dropped cuts"
        )
    elif oc > rc:
        rep.notes.append(
            f"output gained {oc - rc} shot(s) - likely new visual "
            "discontinuities the reference did not have"
        )
    return rep


def render(rep: FidelityReport) -> str:
    lines = ["  STRUCTURAL FIDELITY", ""]
    for s in rep.scores:
        bar = "#" * min(20, int(round(s.value * 20)))
        lines.append(f"    {s.name.ljust(15)} {s.value:5.2f}  {bar.ljust(20)}  {s.detail}")
    lines.append("")
    lines.append(f"    OVERALL         {rep.overall:5.2f}   {rep.verdict}")
    for n in rep.notes:
        lines.append(f"    ! {n}")
    return "\n".join(lines)
