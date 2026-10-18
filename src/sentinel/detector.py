"""Classical OpenCV 5 wildfire smoke detector (long-baseline residual design).

Every detection decision here is made by classical computer vision. No learned
model participates in scoring; the agent may later ask a vision model for a
second opinion on an already-scored candidate, but that opinion never changes
this score.

Design history, kept deliberately
---------------------------------
The first version of this file compared frames 60 seconds apart and scored
texture drop, optical-flow rise and saturation drop between them. Measured on
real FIgLib sequences it did not work, and the failure was informative: between
consecutive frames a developing plume changes the image about as much as wind in
the chaparral, so pre- and post-ignition windows were statistically
indistinguishable on every feature (texture drop means -0.005 vs -0.003,
saturation -0.002 vs +0.002, motion cells 117 vs 103 of 256).

The physics says why. A plume takes 10-40 minutes to grow. Sampled at 60 s it is
nearly static; sampled against a reference 15-45 minutes old it is obvious. This
version therefore scores the *residual against a long-baseline median reference*
(see ``background.py``), which is the primitive fixed-camera monitoring actually
calls for.

What makes a residual smoke-like
--------------------------------
1. **Positive luminance residual** - smoke scatters light, so it is usually
   brighter than the terrain it hides (and it desaturates).
2. **Texture suppression** - a translucent medium erases the ridgeline behind
   it. Local gradient energy drops where smoke covers structure.
3. **Desaturation** - grey-white plume over coloured terrain.
4. **Spatial compactness** - a plume is a connected blob with sane aspect, not
   a scatter of unrelated pixels (which is what haze, exposure error and sensor
   noise produce).
5. **Growth** - the blob's area increases over successive frames.

Each is individually defeated by a different nuisance (cloud shadow, glare,
haze), so the score requires several to hold at once.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import cv2
import numpy as np

WORK_WIDTH = 960


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    """Tunable parameters. ``sensitivity`` is the only knob the agent may move."""

    sensitivity: float = 0.5
    work_width: int = WORK_WIDTH
    horizon_blur: int = 31
    clahe_clip: float = 2.0
    min_area_frac: float = 0.0008   # ignore blobs smaller than this fraction of the band
    max_area_frac: float = 0.45     # a "blob" covering half the band is weather, not a plume
    sky_margin_frac: float = 0.55   # band starts well above the ridge: plumes straddle it

    @property
    def residual_threshold(self) -> float:
        """Grey-level residual that counts as changed (high sens -> lower bar)."""
        return float(np.clip(14.0 - 7.0 * self.sensitivity, 5.0, 14.0))

    @property
    def score_threshold(self) -> float:
        return float(np.clip(0.55 - 0.35 * self.sensitivity, 0.12, 0.55))


@dataclasses.dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    area_frac: float
    score: float
    luminance_residual: float
    texture_drop: float
    desaturation: float
    compactness: float
    growth: float

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@dataclasses.dataclass
class DetectionResult:
    score: float
    candidates: list[Candidate]
    horizon_y: int
    threshold: float
    detected: bool
    has_reference: bool
    mask: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "detected": self.detected,
            "threshold": round(self.threshold, 4),
            "horizon_y": self.horizon_y,
            "has_reference": self.has_reference,
            "candidates": [c.as_dict() for c in self.candidates],
        }


def _to_working(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def estimate_horizon(gray: np.ndarray, blur: int = 31) -> int:
    """Row of the strongest step in row-wise gradient energy (terrain vs sky)."""
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    profile = mag.mean(axis=1).astype(np.float32)
    if blur % 2 == 0:
        blur += 1
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, blur), 0).ravel()
    limit = int(len(profile) * 0.85)
    if limit < 8:
        return len(profile) // 3
    return int(np.argmax(np.diff(profile[:limit]))) + 1


def _texture_energy(gray: np.ndarray) -> np.ndarray:
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    return cv2.boxFilter(mag, cv2.CV_32F, (15, 15))


def detect(
    frame: np.ndarray,
    reference: np.ndarray | None,
    config: DetectorConfig | None = None,
    prev_mask: np.ndarray | None = None,
) -> DetectionResult:
    """Score one frame against a long-baseline reference.

    ``reference`` comes from :func:`background.build_reference`. When it is
    ``None`` (not enough history yet) the result is an explicit non-detection
    with ``has_reference=False`` rather than a guess.
    """
    cfg = config or DetectorConfig()
    cur = _to_working(frame, cfg.work_width)
    gray_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    horizon = estimate_horizon(gray_cur, cfg.horizon_blur)

    if reference is None:
        return DetectionResult(0.0, [], int(horizon), cfg.score_threshold, False, False, None)

    ref = _to_working(reference, cfg.work_width)
    if ref.shape != cur.shape:
        ref = cv2.resize(ref, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)

    from . import background as _bg
    cur_a = _bg.align_to_reference(cur, ref)
    cur_m = _bg.photometric_match(cur_a, ref)

    gray_c = cv2.cvtColor(cur_m, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray_c - gray_r                      # signed: smoke is usually brighter

    top = max(0, horizon - int(cfg.sky_margin_frac * gray_c.shape[0]))
    band = np.zeros_like(residual, dtype=np.uint8)
    band[top:, :] = 255

    # Candidate smoke pixels. Polarity is NOT fixed: a white plume over dark
    # terrain raises luminance, while a dense grey plume against bright hazy sky
    # lowers it. Measured on FIgLib both occur, so take absolute deviation and
    # record which side it fell on.
    thr = cfg.residual_threshold
    mask = ((np.abs(residual) > thr) & (gray_c < 250) & (gray_c > 6) & (band > 0)).astype(np.uint8) * 255
    # Morphology: close gaps inside a diffuse plume, drop speckle.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    tex_c = _texture_energy(cv2.cvtColor(cur_m, cv2.COLOR_BGR2GRAY))
    tex_r = _texture_energy(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    tex_drop_map = (tex_r - tex_c) / np.maximum(tex_r, 8.0)

    sat_c = cv2.cvtColor(cur_m, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    sat_r = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    desat_map = (sat_r - sat_c) / 255.0

    band_area = float((band > 0).sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cands: list[Candidate] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        af = area / max(band_area, 1.0)
        if af < cfg.min_area_frac or af > cfg.max_area_frac:
            continue
        sel = labels == i
        lum_signed = float(residual[sel].mean())
        lum = abs(lum_signed)
        td = float(tex_drop_map[sel].mean())
        ds = float(desat_map[sel].mean())
        # Compactness: filled fraction of the bounding box. Plumes are blobby;
        # exposure artefacts and horizon bands are long thin slivers.
        comp = float(area) / float(max(w * h, 1))
        aspect = w / max(h, 1)
        if aspect > 12.0:      # a full-width sliver is a horizon/exposure artefact
            continue
        growth = 0.0
        if prev_mask is not None and prev_mask.shape == mask.shape:
            prev_area = float((prev_mask[sel] > 0).sum())
            growth = float(np.clip((area - prev_area) / max(area, 1.0), 0.0, 1.0))

        score = (
            float(np.clip(lum / 22.0, 0.0, 1.0))
            * (0.30 + 0.70 * float(np.clip(td * 3.0, 0.0, 1.0)))
            * (0.35 + 0.65 * float(np.clip(ds * 6.0, 0.0, 1.0)))
            * (0.40 + 0.60 * float(np.clip(comp * 1.6, 0.0, 1.0)))
            * (0.65 + 0.35 * growth)
        )
        cands.append(Candidate(int(x), int(y), int(w), int(h), af, score, lum_signed, td, ds, comp, growth))

    cands.sort(key=lambda c: c.score, reverse=True)
    frame_score = cands[0].score if cands else 0.0
    return DetectionResult(
        score=float(frame_score),
        candidates=cands[:8],
        horizon_y=int(horizon),
        threshold=cfg.score_threshold,
        detected=bool(frame_score >= cfg.score_threshold),
        has_reference=True,
        mask=mask,
    )


def annotate(frame: np.ndarray, result: DetectionResult, config: DetectorConfig | None = None) -> np.ndarray:
    """Overlay horizon, candidates and score, for the UI and the report."""
    cfg = config or DetectorConfig()
    img = _to_working(frame, cfg.work_width).copy()
    cv2.line(img, (0, result.horizon_y), (img.shape[1], result.horizon_y), (90, 180, 255), 1, cv2.LINE_AA)
    for i, c in enumerate(result.candidates[:4]):
        colour = (40, 90, 255) if (result.detected and i == 0) else (60, 200, 255)
        cv2.rectangle(img, (c.x, c.y), (c.x + c.w, c.y + c.h), colour, 2)
        cv2.putText(img, f"{c.score:.2f}", (c.x + 4, max(c.y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    state = "SMOKE" if result.detected else ("clear" if result.has_reference else "warming up")
    cv2.putText(img, f"score {result.score:.3f}  thr {result.threshold:.2f}  {state}",
                (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return img
