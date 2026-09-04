"""vre command line."""

from __future__ import annotations

from pathlib import Path

import typer

from . import backends, config, pricing
from .decompile import decompile as run_decompile
from .decompile import load, save
from .router import route

app = typer.Typer(add_completion=False, help="Video reference engine.")


def _echo_kv(label: str, value: str, pad: int = 22) -> None:
    typer.echo(f"  {label.ljust(pad)} {value}")


@app.command()
def doctor(remote: bool = typer.Option(False, "--remote", help="Include GPU backends")) -> None:
    """Show which analysis backends and credentials are active."""
    typer.secho("\nBackends", fg=typer.colors.CYAN, bold=True)
    for key, info in backends.report(allow_remote=remote).items():
        if info["selected"]:
            mark, colour = "OK ", typer.colors.GREEN
            detail = f"{info['selected']} (conf {info['confidence']:.2f})"
        else:
            mark, colour = "-- ", typer.colors.YELLOW
            detail = f"none available; wants {', '.join(info['unavailable'])}"
        typer.secho(f"  {mark}{key.ljust(12)} {detail}", fg=colour)
        if info["selected"] and info["note"]:
            typer.echo(f"     {info['note']}")

    typer.secho("\nCredentials", fg=typer.colors.CYAN, bold=True)
    for key, present in config.credentials_status().items():
        mark = "OK " if present else "-- "
        colour = typer.colors.GREEN if present else typer.colors.YELLOW
        typer.secho(f"  {mark}{key}", fg=colour)
    typer.echo("")


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("jobs"), "--out", "-o", help="Job directory root"),
    threshold: float = typer.Option(27.0, "--threshold", "-t",
                                    help="Shot detection sensitivity; lower finds more cuts"),
    resolution: str = typer.Option("720p", "--resolution", "-r"),
    no_frames: bool = typer.Option(False, "--no-frames"),
    no_motion: bool = typer.Option(False, "--no-motion"),
) -> None:
    """Decompile a reference video and route it."""
    job_dir = out / video.stem
    typer.secho(f"\nAnalysing {video.name}", fg=typer.colors.CYAN, bold=True)

    bp = run_decompile(
        video, job_dir,
        threshold=threshold,
        do_frames=not no_frames,
        do_motion=not no_motion,
    )
    path = save(bp, job_dir)

    s = bp.source
    typer.secho("\nSource", fg=typer.colors.CYAN, bold=True)
    _echo_kv("duration", f"{s.duration:.2f}s")
    _echo_kv("resolution", f"{s.width}x{s.height} ({s.aspect})")
    _echo_kv("fps", f"{s.fps:.3f}")
    _echo_kv("audio", s.audio_codec or "none")

    typer.secho(f"\nShots ({len(bp.shots)}, {bp.cut_count} cuts)", fg=typer.colors.CYAN, bold=True)
    for sh in bp.shots:
        cam = sh.camera
        typer.echo(
            f"  {sh.id}  {sh.t_in:7.2f} -> {sh.t_out:7.2f}  "
            f"{sh.duration:5.2f}s  {cam.move.value:<10} "
            f"speed={cam.speed:.2f} shake={cam.shake:.2f}  out={sh.transition_out.value}"
        )

    if bp.audio.bpm:
        typer.secho("\nAudio", fg=typer.colors.CYAN, bold=True)
        _echo_kv("bpm", f"{bp.audio.bpm:.1f}")
        _echo_kv("beats", str(len(bp.audio.beat_grid)))
        if bp.audio.integrated_lufs is not None:
            _echo_kv("loudness", f"{bp.audio.integrated_lufs:.1f} LUFS")

    decision = route(bp, resolution=resolution)
    typer.secho(f"\nRoute: {decision.path.value.upper()}", fg=typer.colors.GREEN, bold=True)
    for r in decision.reasons:
        typer.echo(f"  + {r}")
    for b in decision.blockers:
        typer.secho(f"  x {b}", fg=typer.colors.RED)
    for w in decision.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)

    typer.secho("\nCost estimates", fg=typer.colors.CYAN, bold=True)
    for est in decision.estimates:
        marker = ">" if est.path == decision.path else " "
        typer.echo(
            f"  {marker} {est.model.ljust(28)} "
            f"${est.generation:6.2f} gen   ${est.with_retries:6.2f} w/ retries"
        )
        typer.echo(f"      {est.detail}")

    if bp.analysis.warnings:
        typer.secho("\nWarnings", fg=typer.colors.YELLOW, bold=True)
        for w in bp.analysis.warnings:
            typer.echo(f"  ! {w}")

    timings = ", ".join(f"{k} {v}ms" for k, v in bp.analysis.timings_ms.items())
    typer.echo(f"\n  timings: {timings}")
    typer.secho(f"  blueprint: {path}\n", fg=typer.colors.GREEN)


@app.command("route")
def route_cmd(
    blueprint: Path = typer.Argument(..., exists=True, dir_okay=False),
    resolution: str = typer.Option("720p", "--resolution", "-r"),
    generator: str = typer.Option(pricing.DEFAULT_GENERATOR, "--generator", "-g"),
) -> None:
    """Re-route an existing blueprint without re-analysing."""
    bp = load(blueprint)
    decision = route(bp, resolution=resolution, generator=generator)
    typer.secho(f"\nRoute: {decision.path.value.upper()}", fg=typer.colors.GREEN, bold=True)
    for r in decision.reasons:
        typer.echo(f"  + {r}")
    for b in decision.blockers:
        typer.secho(f"  x {b}", fg=typer.colors.RED)
    for w in decision.warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    for est in decision.estimates:
        marker = ">" if est.path == decision.path else " "
        typer.echo(f"  {marker} {est.model.ljust(28)} ${est.with_retries:6.2f}  {est.detail}")
    typer.echo("")


@app.command()
def models(resolution: str = typer.Option("720p", "--resolution", "-r")) -> None:
    """List known video models with per-second cost and arena rank."""
    typer.secho(f"\nVideo models @ {resolution}  (verified {pricing.PRICING_VERIFIED})",
                fg=typer.colors.CYAN, bold=True)
    rows = sorted(pricing.MODELS.values(), key=lambda m: m.cost_per_second(resolution))
    for m in rows:
        elo = f"elo {m.i2v_elo}" if m.i2v_elo else ("editor" if m.is_editor else "-")
        typer.echo(
            f"  {m.label.ljust(30)} ${m.cost_per_second(resolution):.4f}/s  "
            f"min {m.min_seconds:g}s  {elo.ljust(9)} {m.provider}"
        )
        if m.note:
            typer.echo(f"      {m.note}")
    typer.echo("")


@app.command()
def batch(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Directory of reference videos"),
    out: Path = typer.Option(Path("jobs/batch"), "--out", "-o"),
    threshold: float = typer.Option(27.0, "--threshold", "-t"),
    frames: bool = typer.Option(False, "--frames", help="Also extract keyframes (slower)"),
) -> None:
    """Decompile a folder of real reference ads and report the distribution.

    Answers the question that gates everything else: what do real references
    actually look like, and how many are even eligible for the Aleph path?
    """
    import json
    import statistics as stats

    vids = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v", ".avi"}
    )
    if not vids:
        raise typer.BadParameter(f"no video files in {folder}")

    typer.secho(f"\nAnalysing {len(vids)} references\n", fg=typer.colors.CYAN, bold=True)
    rows, failures = [], []

    for i, v in enumerate(vids, 1):
        typer.echo(f"  [{i}/{len(vids)}] {v.name[:48].ljust(48)}", nl=False)
        try:
            bp = run_decompile(v, out / v.stem, threshold=threshold,
                               do_frames=frames, do_motion=False)
            d = route(bp)
            conf = min((s.provenance.confidence for s in bp.shots if s.provenance),
                       default=0.0)
            rows.append({
                "file": v.name,
                "duration": bp.source.duration,
                "cuts": bp.cut_count,
                "shots": len(bp.shots),
                "mean_shot": bp.mean_shot_duration,
                "resolution": f"{bp.source.width}x{bp.source.height}",
                "route": d.path.value,
                "confidence": round(conf, 3),
            })
            colour = (typer.colors.GREEN if d.path.value == "aleph"
                      else typer.colors.YELLOW)
            typer.secho(f" {bp.source.duration:6.1f}s  {bp.cut_count:3d} cuts  "
                        f"conf {conf:.2f}  -> {d.path.value}", fg=colour)
        except Exception as exc:
            failures.append((v.name, str(exc)[:120]))
            typer.secho(f" FAILED: {str(exc)[:60]}", fg=typer.colors.RED)

    if not rows:
        typer.secho("\nnothing analysed successfully\n", fg=typer.colors.RED)
        raise typer.Exit(1)

    durs = [r["duration"] for r in rows]
    cuts = [r["cuts"] for r in rows]
    confs = [r["confidence"] for r in rows]
    aleph = [r for r in rows if r["route"] == "aleph"]

    typer.secho("\nDISTRIBUTION", fg=typer.colors.CYAN, bold=True)
    for label, vals in (("duration (s)", durs), ("cuts", cuts),
                        ("shot confidence", confs)):
        typer.echo(f"  {label.ljust(18)} min {min(vals):6.1f}   "
                   f"median {stats.median(vals):6.1f}   max {max(vals):6.1f}")
    typer.echo(f"  {'mean shot (s)'.ljust(18)} "
               f"{stats.median([r['mean_shot'] for r in rows]):6.1f} (median)")

    pct = len(aleph) / len(rows) * 100
    colour = (typer.colors.GREEN if pct >= 60 else
              typer.colors.YELLOW if pct >= 30 else typer.colors.RED)
    typer.secho(f"\nALEPH ELIGIBLE: {len(aleph)}/{len(rows)}  ({pct:.0f}%)",
                fg=colour, bold=True)
    over_dur = sum(1 for r in rows if r["duration"] > 30)
    over_cuts = sum(1 for r in rows if r["cuts"] > 10)
    typer.echo(f"  blocked by duration >30s   {over_dur}")
    typer.echo(f"  blocked by cuts >10        {over_cuts}")

    low = [r for r in rows if r["confidence"] < 0.7]
    if low:
        typer.secho(f"\n  ! {len(low)} refs had shot confidence <0.70 - "
                    "inspect those cut lists by hand", fg=typer.colors.YELLOW)
    if failures:
        typer.secho(f"\n  ! {len(failures)} failed to analyse", fg=typer.colors.RED)
        for name, err in failures[:5]:
            typer.echo(f"      {name}: {err}")

    summary = out / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(
        {"rows": rows, "failures": failures,
         "aleph_eligible_pct": round(pct, 1)}, indent=2), encoding="utf-8")
    typer.secho(f"\n  {summary}\n", fg=typer.colors.GREEN)


@app.command()
def cost(
    profile: str = typer.Option("local", "--profile", "-P", help="local | cloud"),
    duration: float = typer.Option(0.0, "--duration", "-d",
                                   help="Output seconds; 0 uses the profile default"),
    shots: int = typer.Option(0, "--shots", "-s", help="0 uses the profile default"),
    price: float = typer.Option(0.0, "--price", "-p", help="Price per video, for margin"),
) -> None:
    """Unit economics and validation spend for a profile."""
    from . import costing as c

    if profile not in c.PROFILES:
        raise typer.BadParameter(f"profile must be one of {list(c.PROFILES)}")
    prof = c.PROFILES[profile]

    # A pilot proves the machinery on short references; production ships 30s.
    if duration <= 0:
        duration = 10.0 if prof.local_compute else 30.0
    if shots <= 0:
        shots = 4 if prof.local_compute else 12

    typer.secho(f"\nPROFILE: {prof.name.upper()}   "
                f"{prof.resolution}, {prof.image_model}, "
                f"{'free VLM' if prof.vlm_cost == 0 else f'VLM ${prof.vlm_cost}'}",
                fg=typer.colors.MAGENTA, bold=True)

    typer.secho(f"\nPER-JOB COST  ({duration:.0f}s, {shots} shots)",
                fg=typer.colors.CYAN, bold=True)
    for tier in c.Tier:
        jc = c.job_cost(duration, shots, tier, prof)
        typer.secho(f"\n  {tier.value.upper()}", bold=True)
        for item in jc.items:
            flag = " " if item.verified else "~"
            typer.echo(f"   {flag} {item.name.ljust(34)} ${item.cost:8.4f}"
                       + (f"   {item.note}" if item.note else ""))
        typer.secho(f"     {'TOTAL'.ljust(34)} ${jc.total:8.4f}", fg=typer.colors.GREEN)
        if jc.total > 0:
            typer.echo(f"     dominant: {jc.dominant.name} "
                       f"({jc.dominant.cost / jc.total * 100:.0f}% of job)")

    typer.secho("\n  (~) = estimated, not verified", fg=typer.colors.YELLOW)

    typer.secho("\nFIXED MONTHLY", fg=typer.colors.CYAN, bold=True)
    for item in c.fixed_monthly(prof):
        flag = " " if item.verified else "~"
        typer.echo(f"   {flag} {item.name.ljust(34)} ${item.cost:8.2f}"
                   + (f"   {item.note}" if item.note else ""))
    typer.secho(f"     {'TOTAL'.ljust(34)} ${c.fixed_total(prof):8.2f}/mo",
                fg=typer.colors.GREEN)

    typer.secho("\nVALIDATION SPEND (one time)", fg=typer.colors.CYAN, bold=True)
    for line in c.validation_plan(prof, duration, shots):
        typer.echo(f"     {line.name.ljust(34)} ${line.cost:8.2f}   {line.detail}")
    typer.secho(f"     {'TOTAL'.ljust(34)} ${c.validation_total(prof, duration, shots):8.2f}",
                fg=typer.colors.GREEN)

    if price > 0:
        typer.secho(f"\nMARGIN AT ${price:.0f}/VIDEO", fg=typer.colors.CYAN, bold=True)
        for tier in c.Tier:
            m = c.margin(price, tier, duration, shots, prof)
            colour = typer.colors.GREEN if m["margin_pct"] > 50 else (
                typer.colors.YELLOW if m["margin_pct"] > 0 else typer.colors.RED)
            typer.secho(f"     {tier.value.ljust(10)} cogs ${m['cogs']:6.2f}  "
                        f"gross ${m['gross']:6.2f}  margin {m['margin_pct']:5.1f}%",
                        fg=colour)
    typer.echo("")


if __name__ == "__main__":
    app()
