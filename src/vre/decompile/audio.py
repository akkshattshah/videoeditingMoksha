"""Tempo, beat grid and loudness.

Uses a numpy spectral-flux onset envelope with autocorrelation tempo estimation
so the default path needs no librosa. If librosa is installed it is preferred —
its beat tracker is materially better on music with a weak downbeat.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ..probe import decode_audio_mono, integrated_lufs
from ..schema import AudioTrack, Provenance

SR = 22050
N_FFT = 2048
HOP = 512
BPM_MIN, BPM_MAX = 60.0, 190.0


def _onset_envelope(y: np.ndarray) -> np.ndarray:
    """Spectral flux: summed positive frame-to-frame magnitude increase."""
    if y.size < N_FFT * 2:
        return np.zeros(0, dtype=np.float32)

    n_frames = 1 + (len(y) - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)

    # Strided view avoids materialising a copy per frame.
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, N_FFT),
        strides=(y.strides[0] * HOP, y.strides[0]),
        writeable=False,
    )
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)

    # Log compression keeps loud transients from dominating the envelope.
    spec = np.log1p(spec * 8.0)
    flux = np.diff(spec, axis=0)
    env = np.maximum(flux, 0.0).sum(axis=1)

    if env.size and env.max() > 0:
        env = env / env.max()
    return env.astype(np.float32)


def _estimate_tempo(env: np.ndarray, env_fps: float) -> float | None:
    if env.size < 16:
        return None

    centred = env - env.mean()
    ac = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    if ac.size < 4 or ac[0] <= 0:
        return None
    ac = ac / ac[0]

    lag_min = max(2, int(round(60.0 / BPM_MAX * env_fps)))
    lag_max = min(len(ac) - 1, int(round(60.0 / BPM_MIN * env_fps)))
    if lag_max <= lag_min:
        return None

    window = ac[lag_min:lag_max + 1]
    best = int(np.argmax(window)) + lag_min
    if ac[best] < 0.05:
        return None

    # Octave correction. Autocorrelation peaks just as strongly at 2x and 3x
    # the true period, so a 120 BPM track reads as 60 unless the sub-multiples
    # are checked explicitly. Prefer the faster reading when it is nearly as
    # strong, since the slow harmonic is the more common failure.
    for divisor in (2, 3):
        candidate = best // divisor
        if candidate < lag_min:
            continue
        if ac[candidate] >= 0.75 * ac[best]:
            best = candidate
            break

    return float(60.0 * env_fps / best)


def _beat_grid(env: np.ndarray, env_fps: float, bpm: float,
               duration: float) -> list[float]:
    """Phase-align a constant-tempo pulse train to the onset envelope."""
    period = 60.0 / bpm * env_fps
    if period < 1:
        return []

    best_offset, best_score = 0.0, -np.inf
    for offset in np.linspace(0, period, num=24, endpoint=False):
        idx = np.arange(offset, len(env), period).astype(int)
        idx = idx[idx < len(env)]
        if idx.size == 0:
            continue
        score = float(env[idx].sum() / idx.size)
        if score > best_score:
            best_score, best_offset = score, float(offset)

    times = np.arange(best_offset, len(env), period) / env_fps
    return [round(float(t), 4) for t in times if t <= duration]


def _librosa_analyze(path: str | Path) -> tuple[float, list[float]] | None:
    if importlib.util.find_spec("librosa") is None:
        return None
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=SR, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        return bpm, [round(float(t), 4) for t in beats]
    except Exception:
        return None


def analyze(path: str | Path, duration: float, has_audio: bool) -> AudioTrack:
    if not has_audio:
        return AudioTrack(
            provenance=Provenance(backend="none", confidence=1.0,
                                  note="source has no audio stream")
        )

    lufs = None
    try:
        lufs = integrated_lufs(path)
    except Exception:
        pass

    via_librosa = _librosa_analyze(path)
    if via_librosa is not None:
        bpm, beats = via_librosa
        return AudioTrack(
            bpm=round(bpm, 2),
            beat_grid=beats,
            integrated_lufs=lufs,
            provenance=Provenance(backend="librosa", confidence=0.85),
        )

    y = decode_audio_mono(path, sr=SR)
    if y.size == 0:
        return AudioTrack(
            integrated_lufs=lufs,
            provenance=Provenance(backend="numpy-flux", confidence=0.1,
                                  note="audio decode produced no samples"),
        )

    env = _onset_envelope(y)
    env_fps = SR / HOP
    bpm = _estimate_tempo(env, env_fps)

    beats: list[float] = []
    confidence = 0.35
    if bpm is not None:
        beats = _beat_grid(env, env_fps, bpm, duration)
        # A dense, plausible grid is a decent proxy for a real musical bed.
        confidence = 0.62 if len(beats) > 8 else 0.45

    return AudioTrack(
        bpm=round(bpm, 2) if bpm else None,
        beat_grid=beats,
        integrated_lufs=lufs,
        provenance=Provenance(
            backend="numpy-flux",
            confidence=confidence,
            note="install librosa for better beat tracking",
        ),
    )
