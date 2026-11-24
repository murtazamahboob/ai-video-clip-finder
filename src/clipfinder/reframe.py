"""9:16 reframe — face-tracked vertical crop with smooth camera motion.

Blueprint §1.4: 1080×1920 output, face kept centred, exponential-moving-
average smoothing (alpha 0.15–0.25), max crop shift limited per frame, and
a blurred-background fit when no face is visible. The dominant speaker's
crop is locked for at least 1.5 s so overlapping speech cannot make the
frame jump (§1.8 failure mode).

The math is resolution-independent: it works in normalised 0..1 x-coords
and converts to pixels for a given source size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

OUT_W, OUT_H = 1080, 1920
ALPHA = 0.20            # EMA smoothing, blueprint range 0.15–0.25
LOCK_SECONDS = 1.5      # minimum hold on the dominant speaker
MAX_SHIFT_PER_FRAME = 0.02  # normalised x per frame at 30 fps


@dataclass
class CropPlan:
    """One crop per second of the clip: centre-x and zoom for that second."""
    seconds: list[float]
    centre_x: list[float]
    zoom: list[float]
    fallback_blur: list[bool]

    def crop_pixels(self, src_w: int = 1920, src_h: int = 1080,
                    second: float = 0.0) -> tuple[int, int, int, int]:
        """(x, y, w, h) source rect for the given second — 9:16 aspect."""
        i = min(len(self.seconds) - 1, max(0, int(second - self.seconds[0])))
        cx, zoom = self.centre_x[i], self.zoom[i]
        h = src_h / zoom
        w = h * 9 / 16
        x = min(max(cx * src_w - w / 2, 0), src_w - w)
        return int(round(x)), int(round((src_h - h) / 2)), int(round(w)), int(round(h))


def plan_reframe(face_x: np.ndarray, start: float, end: float,
                 energy: np.ndarray | None = None,
                 alpha: float = ALPHA) -> CropPlan:
    seconds = [start + i for i in range(int(math.ceil(end - start)))]
    centres: list[float] = []
    zooms: list[float] = []
    fallback: list[bool] = []
    smooth: float | None = None
    locked_until = -1.0
    locked_centre = 0.5
    for t in seconds:
        idx = min(len(face_x) - 1, int(t))
        face = face_x[idx] if idx < len(face_x) else np.nan
        if np.isfinite(face):
            if t >= locked_until:
                locked_centre = float(face)
                locked_until = t + LOCK_SECONDS
            target = locked_centre
            is_fallback = False
        else:
            target = smooth if smooth is not None else 0.5
            is_fallback = True
        smooth = target if smooth is None else smooth + alpha * (target - smooth)
        # Limit per-frame shift so the camera never jumps.
        if centres:
            shift = max(-MAX_SHIFT_PER_FRAME * 30,
                        min(MAX_SHIFT_PER_FRAME * 30, smooth - centres[-1]))
            smooth = centres[-1] + shift
        centres.append(round(float(smooth), 4))
        # Smart zoom (Part 3 upgrade): punch in 15% on slow moments.
        slow = energy is not None and idx < len(energy) and energy[idx] < 0.4
        zooms.append(1.15 if slow else 1.0)
        fallback.append(is_fallback)
    return CropPlan(seconds, centres, zooms, fallback)
