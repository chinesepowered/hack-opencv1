"""Per-frame feature extraction for the fitted smoke classifier.

Why a fitted rule on top of classical CV
----------------------------------------
Earlier versions of this project combined five hand-chosen terms multiplicatively.
Measured on real sequences that structure was worse than a single feature: signed
luminance residual alone separated pre/post at Cohen's d = -1.05, while the
product score separated at roughly nothing. Hand-weighting was the problem, not
the imagery.

So the features below stay entirely classical OpenCV measurements - no learned
representation anywhere - and a small interpretable model (logistic regression or
a shallow tree) decides how to weigh them. The pipeline is still an OpenCV
pipeline; only the final decision boundary is fitted, on the design split only.

Physics the features are meant to capture
-----------------------------------------
* **Signed luminance residual.** A plume may be brighter than terrain or darker
  than bright hazy sky. The *sign matters* and must not be folded away with
  abs(); measurement showed folding it destroyed the strongest feature.
* **Joint dark-and-smooth.** Smoke is a scattering medium: it both shifts
  luminance and erases the texture behind it. Cloud shadow shifts luminance and
  keeps texture; haze smooths without a localised luminance step. The *joint*
  region is far more specific than either gate alone.
* **Directional persistence.** A plume drifts coherently over minutes, so its
  residual mass moves in a consistent direction. Cloud churn reverses direction;
  sensor noise has no direction at all.
* **Multi-minute temporal slope.** Plume area grows over 10-40 minutes. Sampling
  that slope over a multi-frame baseline is meaningful, whereas the 60-second
  "growth" measured earlier ran backwards (d = -0.89) because it tracked cloud
  churn.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from . import background as _bg
from .detector import _texture_energy, _to_working, estimate_horizon


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    work_width: int = 960
    sky_margin_frac: float = 0.55
    residual_threshold: float = 10.0
    min_area_frac: float = 0.0006
    max_area_frac: float = 0.45


def _band_top(gray: np.ndarray, cfg: FeatureConfig) -> tuple[int, int]:
    hz = estimate_horizon(gray)
    return max(0, hz - int(cfg.sky_margin_frac * gray.shape[0])), hz


def extract(
    frame: np.ndarray,
    reference: np.ndarray | None,
    history: list[tuple[int, np.ndarray]] | None = None,
    now_s: int = 0,
    cfg: FeatureConfig | None = None,
) -> dict | None:
    """Measure one frame against its long-baseline reference.

    Returns ``None`` when there is no reference yet (camera still warming up) or
    when no region deviates enough to be worth scoring - both are honest
    non-answers rather than zeros that would pollute a fit.
    """
    cfg = cfg or FeatureConfig()
    cur = _to_working(frame, cfg.work_width)
    gray_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if reference is None:
        return None

    ref = _to_working(reference, cfg.work_width)
    if ref.shape != cur.shape:
        ref = cv2.resize(ref, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)

    cur_m = _bg.photometric_match(_bg.align_to_reference(cur, ref), ref)
    gc = cv2.cvtColor(cur_m, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gc - gr

    top, horizon = _band_top(gray_cur, cfg)
    band = np.zeros(residual.shape, np.uint8)
    band[top:, :] = 255

    tex_c = _texture_energy(gc.astype(np.uint8))
    tex_r = _texture_energy(gr.astype(np.uint8))
    tex_drop = (tex_r - tex_c) / np.maximum(tex_r, 8.0)

    sat_c = cv2.cvtColor(cur_m, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    sat_r = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    desat = (sat_r - sat_c) / 255.0

    # Joint "changed luminance AND lost texture": the specific smoke signature.
    joint = ((np.abs(residual) > cfg.residual_threshold) & (tex_drop > 0.05) & (band > 0))
    joint_u8 = (joint.astype(np.uint8)) * 255
    joint_u8 = cv2.morphologyEx(joint_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    joint_u8 = cv2.morphologyEx(joint_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    band_area = float((band > 0).sum()) or 1.0
    n, labels, stats, cents = cv2.connectedComponentsWithStats(joint_u8, 8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        af = area / band_area
        if af < cfg.min_area_frac or af > cfg.max_area_frac:
            continue
        if w / max(h, 1) > 12.0:
            continue
        if area > best_area:
            best, best_area = (i, x, y, w, h, area, af), area
    if best is None:
        return None
    i, x, y, w, h, area, af = best
    sel = labels == i

    lum = float(residual[sel].mean())
    feat = {
        "lum_signed": lum,
        "lum_dark_frac": float((residual[sel] < 0).mean()),
        "tex_drop": float(tex_drop[sel].mean()),
        "desat": float(desat[sel].mean()),
        "area_frac": float(af),
        "compactness": float(area) / float(max(w * h, 1)),
        "aspect": float(w) / float(max(h, 1)),
        "centroid_y_rel": float((cents[i][1] - top) / max(gc.shape[0] - top, 1)),
        "band_cover": float(joint.sum()) / band_area,
        "edge_density": float(cv2.Canny(cur_m, 60, 160)[sel].mean() / 255.0),
    }

    # --- temporal features over a multi-minute baseline -----------------------
    slope = 0.0
    drift_coh = 0.0
    if history:
        recent = [(t, im) for t, im in history if 0 < now_s - t <= 900]
        recent.sort(key=lambda p: p[0])
        areas, cxs, cys, ts = [], [], [], []
        for t, im in recent[-5:]:
            w_im = _to_working(im, cfg.work_width)
            if w_im.shape != ref.shape:
                w_im = cv2.resize(w_im, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            m_im = _bg.photometric_match(w_im, ref)
            g_im = cv2.cvtColor(m_im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            r_im = g_im - gr
            t_im = (tex_r - _texture_energy(g_im.astype(np.uint8))) / np.maximum(tex_r, 8.0)
            j = ((np.abs(r_im) > cfg.residual_threshold) & (t_im > 0.05) & (band > 0))
            a = float(j.sum()) / band_area
            areas.append(a); ts.append(t)
            ys, xs = np.nonzero(j)
            if xs.size:
                cxs.append(float(xs.mean())); cys.append(float(ys.mean()))
        if len(areas) >= 3:
            tt = np.array(ts, np.float64); aa = np.array(areas, np.float64)
            tt = (tt - tt.mean()) / 60.0
            slope = float(np.polyfit(tt, aa, 1)[0])          # area growth per minute
        if len(cxs) >= 3:
            dx = np.diff(cxs); dy = np.diff(cys)
            ang = np.arctan2(dy, dx)
            # Resultant length: 1 = perfectly consistent drift, 0 = churn.
            drift_coh = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    feat["area_slope_per_min"] = slope
    feat["drift_coherence"] = drift_coh
    feat["bbox"] = [int(x), int(y), int(w), int(h)]
    feat["horizon_y"] = int(horizon)
    return feat


FEATURE_NAMES = [
    "lum_signed", "lum_dark_frac", "tex_drop", "desat", "area_frac",
    "compactness", "aspect", "centroid_y_rel", "band_cover", "edge_density",
    "area_slope_per_min", "drift_coherence",
]


def vector(feat: dict) -> list[float]:
    return [float(feat.get(k, 0.0)) for k in FEATURE_NAMES]
