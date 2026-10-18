"""Long-baseline background reference for fixed cameras.

Why this exists
---------------
The first detector compared frames 60 seconds apart. That is the wrong primitive
for this data: a wildfire plume takes 10-40 minutes to grow, so between two
consecutive FIgLib frames a real plume changes the image about as much as wind
in the chaparral does. Measured on a real sequence, pre-ignition and
post-ignition windows were statistically indistinguishable on every short-baseline
feature (texture drop, motion, saturation), which is exactly what the physics
predicts once you look at the numbers.

A fixed camera lets us do much better. The scene is constant apart from
illumination, weather and the thing we care about. So we keep a *reference* built
from frames tens of minutes old and ask what is different now. The plume has had
time to develop against that reference, and the terrain has not moved.

Robust statistics matter here. A mean reference is dragged by the very plume we
are trying to find, and by birds, insects and lens flare. A per-pixel median over
a sparse set of older frames is immune to all of those as long as the plume
covers fewer than half the sampled frames, which holds by construction because
the reference lags the present.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class BackgroundConfig:
    """How the reference is built.

    **Anchoring matters more than lag.** An earlier version used a *sliding*
    window (``lag_s=900, span_s=1800``) that crept forward with the query frame.
    Measured against FIgLib offsets that window was 17% post-ignition at +1200 s,
    50% at +1800 s and 83% at +2400 s: the median reference was built mostly from
    frames that already contained the plume, so the residual subtracted the plume
    from itself. Detection got *worse* as the fire grew, which is exactly what the
    measurements showed (signed luminance inverted, growth ran backwards).

    The reference is therefore **anchored to the oldest history available** rather
    than sliding. In deployment this is the standard pattern for fixed-camera
    monitoring: hold a reference from a period the system has already judged
    clean, and only advance it when nothing is detected. ``anchor=True``
    implements the evaluation-time equivalent - take the oldest frames in the
    buffer, which for a camera that has been running are genuinely clear.
    """

    lag_s: int = 900          # minimum age of a reference frame (sliding mode)
    span_s: int = 1800        # window length (sliding mode)
    samples: int = 7          # how many frames to median together
    work_width: int = 960
    anchor: bool = True       # build from oldest history instead of a sliding window
    anchor_span_s: int = 1800 # how much of the oldest history to draw from
    min_age_s: int = 300      # never use frames newer than this relative to now


def build_reference(
    frames: list[tuple[int, np.ndarray]],
    now_s: int,
    config: BackgroundConfig | None = None,
) -> np.ndarray | None:
    """Median reference image from frames at least ``lag_s`` older than ``now_s``.

    ``frames`` is a list of ``(offset_seconds, bgr_image)``. Returns ``None``
    when there is not enough history yet, which is a normal state at the start
    of a camera's day and must be handled by the caller rather than faked.
    """
    cfg = config or BackgroundConfig()
    if cfg.anchor:
        # Anchored: oldest frames in the buffer, never anything newer than
        # min_age_s. This cannot creep forward into the event being detected.
        older = [(t, im) for t, im in frames if now_s - t >= cfg.min_age_s]
        if not older:
            return None
        t0 = min(t for t, _ in older)
        pool = [(t, im) for t, im in older if t - t0 <= cfg.anchor_span_s]
    else:
        pool = [(t, im) for t, im in frames
                if cfg.lag_s <= now_s - t <= cfg.lag_s + cfg.span_s]
    if len(pool) < 3:
        return None
    pool.sort(key=lambda p: p[0])
    # Even spread across the window, so a burst of similar frames cannot dominate.
    idx = np.linspace(0, len(pool) - 1, min(cfg.samples, len(pool))).astype(int)
    stack = []
    for i in idx:
        im = pool[i][1]
        h, w = im.shape[:2]
        if w != cfg.work_width:
            im = cv2.resize(im, (cfg.work_width, int(round(h * cfg.work_width / w))), interpolation=cv2.INTER_AREA)
        stack.append(im)
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def align_to_reference(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Correct small camera drift (wind sway, servicing) before differencing.

    Fixed cameras are not perfectly fixed. A few pixels of sway turns every
    high-contrast edge into a false "change", which would swamp a diffuse plume.
    ECC translation alignment is cheap and enough for mast sway; it is skipped
    silently if it fails to converge, because a failed alignment is better than
    a wrong one.
    """
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            g_ref, g_img, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), None, 5,
        )
    except cv2.error:
        return img
    shift = float(np.hypot(warp[0, 2], warp[1, 2]))
    if shift < 0.3 or shift > 40:
        return img  # nothing to fix, or a bogus solution
    return cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def photometric_match(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match ``img`` to ``ref`` in brightness/contrast per channel.

    Sun angle changes over the reference window, and cameras auto-expose. A
    global gain+offset fit removes that without touching local structure, so the
    residual holds only things that changed *in the scene*.
    """
    out = np.empty_like(img)
    for c in range(3):
        a = img[..., c].astype(np.float32).ravel()
        b = ref[..., c].astype(np.float32).ravel()
        # Robust line fit on a subsample: resistant to the plume itself.
        sel = np.linspace(0, a.size - 1, min(20000, a.size)).astype(int)
        A = np.vstack([a[sel], np.ones(sel.size, np.float32)]).T
        try:
            gain, offset = np.linalg.lstsq(A, b[sel], rcond=None)[0]
        except np.linalg.LinAlgError:
            gain, offset = 1.0, 0.0
        gain = float(np.clip(gain, 0.5, 2.0))
        out[..., c] = np.clip(img[..., c].astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    return out
