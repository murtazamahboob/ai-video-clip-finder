"""Per-second feature track for a source video.

One row per second of source: audio energy (0-1), laughter/applause flag,
speech rate (words/sec), face centre-x (0-1, None when no face is visible)
and the active speaker label. A real ingest fills this from ffmpeg +
Whisper + a face detector; the bundled demo synthesises it with a fixed
seed so every run is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class FeatureTrack:
    duration: float
    energy: np.ndarray          # (n_seconds,) 0..1 audio energy
    laughter: np.ndarray        # (n_seconds,) 0/1
    speech_rate: np.ndarray     # (n_seconds,) words per second
    face_x: np.ndarray          # (n_seconds,) 0..1 or NaN when no face
    speaker: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.energy)

    def mean_energy(self, start: float, end: float) -> float:
        i0, i1 = int(start), min(len(self), int(math.ceil(end)))
        if i1 <= i0:
            return 0.0
        return float(self.energy[i0:i1].mean())

    def laughter_seconds(self, start: float, end: float) -> int:
        i0, i1 = int(start), min(len(self), int(math.ceil(end)))
        if i1 <= i0:
            return 0
        return int(self.laughter[i0:i1].sum())

    def speaker_changes(self, start: float, end: float) -> int:
        seg = self.speaker[int(start):int(math.ceil(end))]
        return sum(1 for a, b in zip(seg, seg[1:]) if a != b)

    def mean_speech_rate(self, start: float, end: float) -> float:
        i0, i1 = int(start), min(len(self), int(math.ceil(end)))
        if i1 <= i0:
            return 0.0
        return float(self.speech_rate[i0:i1].mean())

    def face_coverage(self, start: float, end: float) -> float:
        i0, i1 = int(start), min(len(self), int(math.ceil(end)))
        if i1 <= i0:
            return 0.0
        return float(np.isfinite(self.face_x[i0:i1]).mean())


def neutral_track(duration: float) -> FeatureTrack:
    """Flat features for a bare SRT with no audio/vision analysis."""
    n = max(1, int(math.ceil(duration)))
    return FeatureTrack(
        duration=duration,
        energy=np.full(n, 0.5),
        laughter=np.zeros(n, dtype=int),
        speech_rate=np.full(n, 2.5),
        face_x=np.full(n, np.nan),
        speaker=["A"] * n,
    )


def make_synthetic_track(duration: float, seed: int = 42,
                         hot_seconds: tuple[int, ...] = ()) -> FeatureTrack:
    """Seeded synthetic track: baseline energy with spikes on hot seconds."""
    rng = np.random.default_rng(seed)
    n = max(1, int(math.ceil(duration)))
    energy = np.clip(0.45 + 0.15 * np.sin(np.arange(n) / 23.0)
                     + rng.normal(0, 0.07, n), 0.05, 1.0)
    laughter = np.zeros(n, dtype=int)
    for s in hot_seconds:
        if 0 <= s < n:
            energy[s:s + 3] = np.clip(energy[s:s + 3] + 0.35, 0, 1)
            laughter[s:s + 2] = 1
    # Face wanders slowly (random walk), drops out for ~8% of seconds.
    face = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, n)), 0.15, 0.85)
    face[rng.random(n) < 0.08] = np.nan
    speaker = ["A" if (i // 47) % 2 == 0 else "B" for i in range(n)]
    return FeatureTrack(duration, energy, laughter,
                        np.clip(rng.normal(2.5, 0.4, n), 1.0, 4.5),
                        face, speaker)
