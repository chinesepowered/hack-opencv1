"""FIgLib dataset access.

Source: HPWREN Fire Ignition images Library (FIgLib).
Terms: "While we are making the data publicly available, its use requires a
credit reference to https://www.hpwren.ucsd.edu/ in derivative work."
Disclaimer: "This data is provided as-is, no guarantees for anything is included."

See docs/DATA.md. We cache frames locally (git-ignored) and never redistribute
the originals in this repository; the report and UI show derived visualisations
with the required credit.

Ground truth comes free with the filenames. FIgLib names each frame
``<origin_timestamp>_<offset_seconds_from_visible_plume_appearance>.jpg`` so a
negative offset is before the plume was visible to a human annotator and a
non-negative offset is after. That gives us both binary labels and, because the
offsets are real seconds, an exact detection-latency measurement.
"""

from __future__ import annotations

import dataclasses
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CDN = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
CREDIT = "Imagery: HPWREN FIgLib (https://www.hpwren.ucsd.edu/), used with the required credit reference."
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
UA = "smoke-sentinel/0.1 (OpenCV AI Competition 2026 entry; contact via GitHub)"

# FIgLib URL-encodes the leading "+" of non-negative offsets as %2B, e.g.
# 1500763593_%2B02280.jpg. A naive r"_(-?\d+)\.jpg$" silently matches only the
# pre-ignition half of every sequence and throws away all the positive labels,
# so the encoded form is handled explicitly here.
_OFFSET_RE = re.compile(r"_(%2B|\+|-)?(\d+)\.jpg$", re.IGNORECASE)


def _parse_offset(name: str) -> int | None:
    m = _OFFSET_RE.search(name)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    return sign * int(m.group(2))


@dataclasses.dataclass(frozen=True)
class Frame:
    """One FIgLib frame with its label derived from the filename."""

    sequence: str
    name: str
    offset_s: int

    @property
    def is_smoke(self) -> bool:
        """True once the plume is visible (offset >= 0 by FIgLib's convention)."""
        return self.offset_s >= 0

    @property
    def url(self) -> str:
        return f"{CDN}/{self.sequence}/{self.name}"

    @property
    def cache_path(self) -> Path:
        return CACHE / self.sequence / self.name


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_sequences(limit: int | None = None) -> list[str]:
    """All sequence directory names on the FIgLib CDN."""
    html = _get(f"{CDN}/index.html").decode("utf-8", "ignore")
    names = re.findall(r'href=["\']?(\d{8}_[^/"\' >]+)/index\.html', html)
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen[:limit] if limit else seen


def list_frames(sequence: str) -> list[Frame]:
    """Frames of one sequence, ordered by time offset."""
    html = _get(f"{CDN}/{sequence}/index.html").decode("utf-8", "ignore")
    out: list[Frame] = []
    for name in re.findall(r'href=["\']?([^"\' >]+\.jpg)', html):
        offset = _parse_offset(name)
        if offset is None:
            continue
        out.append(Frame(sequence=sequence, name=name, offset_s=offset))
    out.sort(key=lambda f: f.offset_s)
    return out


def fetch(frame: Frame) -> Path:
    """Download one frame into the local cache if not already present."""
    p = frame.cache_path
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_get(frame.url))
    return p


def load(frame: Frame) -> np.ndarray:
    """Fetch (if needed) and decode a frame as BGR."""
    img = cv2.imread(str(fetch(frame)), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to decode {frame.cache_path}")
    return img


def window(frames: list[Frame], index: int, size: int = 3) -> list[Frame]:
    """The ``size`` frames ending at ``index`` (a detector input buffer)."""
    start = max(0, index - size + 1)
    return frames[start : index + 1]
