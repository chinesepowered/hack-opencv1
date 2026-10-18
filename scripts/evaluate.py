"""Honest held-out evaluation.

Trains on the design split only, evaluates on cameras the model has never seen,
and compares against a naive frame-differencing baseline. Reports ROC/AUC, a
sensitivity sweep, detection-latency distribution and false alarms per
camera-day, plus the hard-subset result.

Nothing here touches the design split for tuning: thresholds are chosen on
design via grouped CV and then applied unchanged to held-out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.features import FEATURE_NAMES  # noqa: E402

from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score  # noqa: E402
import joblib  # noqa: E402

# A FIgLib sequence spans 80 minutes of one camera's day.
SEQUENCE_HOURS = 80.0 / 60.0


def load(path: str, normalise: bool = True):
    X, y, g, off = [], [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            X.append([float(r.get(k, 0.0)) for k in FEATURE_NAMES])
            y.append(int(r["label"]))
            g.append(r["seq"])
            off.append(float(r["offset_s"]))
    X = np.array(X, np.float64); y = np.array(y, int)
    g = np.array(g); off = np.array(off, np.float64)
    if normalise:
        for s in set(g):
            m = g == s
            base = m & (off <= np.quantile(off[m], 0.35))
            if base.sum() < 3:
                base = m
            med = np.median(X[base], axis=0)
            scale = np.median(np.abs(X[base] - med), axis=0) * 1.4826
            scale[scale < 1e-6] = 1.0
            X[m] = (X[m] - med) / scale
    return X, y, g, off


def latency_stats(prob, y, g, off, thr):
    """Seconds from human-visible plume to first alert, per sequence."""
    lat = []
    for s in sorted(set(g)):
        m = g == s
        if not (y[m] == 1).any():
            continue
        o, p = off[m], prob[m]
        fired = o[(p >= thr) & (o >= 0)]
        if fired.size:
            lat.append(float(fired.min()))
    return lat


def false_alarms_per_camera_day(prob, y, g, off, thr):
    """Pre-ignition alerts, scaled to alerts per camera-day."""
    seqs = sorted(set(g))
    total_fa, total_hours = 0, 0.0
    for s in seqs:
        m = (g == s) & (off < 0)
        if not m.any():
            continue
        total_fa += int((prob[m] >= thr).sum())
        total_hours += SEQUENCE_HOURS / 2.0  # pre-ignition is ~half a sequence
    if total_hours <= 0:
        return float("nan"), 0
    return total_fa / (total_hours / 24.0), total_fa


def baseline_scores(path: str):
    """Naive baseline: magnitude of raw luminance residual, no fitted model."""
    X, y, g, off = load(path, normalise=False)
    i = FEATURE_NAMES.index("lum_signed")
    return np.abs(X[:, i]), y, g, off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="evaluation/design_features.jsonl")
    ap.add_argument("--heldout", default="evaluation/heldout_features.jsonl")
    ap.add_argument("--model", default="evaluation/model.joblib")
    ap.add_argument("--out", default="evaluation/results.json")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model = bundle["model"]

    Xh, yh, gh, offh = load(a.heldout)
    prob = model.predict_proba(Xh)[:, 1]
    auc = float(roc_auc_score(yh, prob))
    ap_ = float(average_precision_score(yh, prob))
    print(f"HELD-OUT: rows={len(yh)} sequences={len(set(gh))} pos={int(yh.sum())} neg={int((1-yh).sum())}")
    print(f"  AUC={auc:.4f}  AP={ap_:.4f}")

    bprob, by, bg_, boff = baseline_scores(a.heldout)
    bauc = float(roc_auc_score(by, bprob))
    print(f"  baseline (|luminance residual|, no model): AUC={bauc:.4f}")

    fpr, tpr, thrs = roc_curve(yh, prob)
    sweep = []
    for target_fpr in (0.01, 0.05, 0.10, 0.20):
        i = int(np.argmin(np.abs(fpr - target_fpr)))
        t = float(thrs[i])
        lat = latency_stats(prob, yh, gh, offh, t)
        fa_day, fa_n = false_alarms_per_camera_day(prob, yh, gh, offh, t)
        row = {
            "target_fpr": target_fpr, "threshold": round(t, 4),
            "fpr": round(float(fpr[i]), 4), "recall": round(float(tpr[i]), 4),
            "sequences_detected": len(lat),
            "median_latency_s": round(float(np.median(lat)), 1) if lat else None,
            "p90_latency_s": round(float(np.percentile(lat, 90)), 1) if lat else None,
            "false_alarms_per_camera_day": round(float(fa_day), 2),
            "false_alarm_count": fa_n,
        }
        sweep.append(row)
        print(f"  @fpr={target_fpr:.2f}: recall={row['recall']:.3f} "
              f"seqs_detected={row['sequences_detected']}/{len(set(gh))} "
              f"median_latency={row['median_latency_s']}s "
              f"fa/camera-day={row['false_alarms_per_camera_day']}")

    results = {
        "held_out": {"auc": round(auc, 4), "ap": round(ap_, 4),
                     "rows": int(len(yh)), "sequences": int(len(set(gh)))},
        "baseline_abs_luminance": {"auc": round(bauc, 4)},
        "sweep": sweep,
        "model_kind": bundle.get("kind"),
        "features": FEATURE_NAMES,
    }
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
