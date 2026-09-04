# Testbench

Drop a reference video and the elements you want swapped in. Get an edited
video plus an objective structural-fidelity score.

## Drop your files here

```
testbench/input/
  <anything>.mp4      <- exactly ONE reference video
  brief.txt           <- one line: what should change
  elements/
    product.jpg       <- the new product / outfit / object
    model.jpg         <- optional
    background.jpg    <- optional
```

## Run it

```bash
python testbench/run.py            # PREFLIGHT - validates everything, spends $0
python testbench/run.py --go       # actually runs
```

**Preflight is the default.** It checks your files, decompiles the reference,
prices the job, reads your credit balance, and proves the HTTPS tunnel works —
without spending anything. Only `--go` costs money.

## Options

| Flag | Default | |
|---|---|---|
| `--model` | `h3` | `h3` $0.13/s · `seedance` $0.20/s · `aleph` $0.28/s |
| `--seconds` | `5` | length of the test segment; `0` = whole video |
| `--start` | `0` | where to cut the segment from |
| `--budget` | `5.00` | hard cap — refuses to start above it |
| `--resolution` | model default | e.g. `480p` |
| `--anchor` | off | pre-build an anchor frame (+$0.08) |

Start cheap: `--model h3 --seconds 5` is about **$0.65**.

## What you get

```
testbench/output/<timestamp>_<model>/
  segment.mp4        the exact input that was sent
  result.mp4         the edited output
  sidebyside.mp4     original | result
  ref/blueprint.json decompiled reference
  out/blueprint.json decompiled output
  report.md          fidelity scores and actual spend
```

## How it's scored

The output is decompiled and its blueprint diffed against the reference's:

| Metric | Weight | |
|---|---|---|
| cut_placement | 3.0 | cuts within 0.25s of the original |
| shot_count | 2.0 | same number of shots |
| duration | 1.5 | length drift |
| camera_moves | 1.5 | per-shot move classification matches |
| aspect | 0.5 | aspect ratio preserved |

**≥0.85** structure preserved · **0.65–0.85** mostly held, check by eye ·
**<0.40** failed.

A high score means the *edit* held the structure. It says nothing about whether
the swap looks good — watch `sidebyside.mp4` for that.

## Notes

- Video models need a public HTTPS URL; data URIs are rejected. `run.py` uses a
  free anonymous **cloudflared quick tunnel** to serve your local files, and
  verifies it's reachable before spending. The tunnel dies with the process.
- Everything under `input/` and `output/` is gitignored.
