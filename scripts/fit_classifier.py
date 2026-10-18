"""Fit an interpretable decision rule over classical OpenCV features.

Design-set only, grouped cross-validation by sequence. Two candidate models:
logistic regression (linear, coefficients readable) and a depth-3 tree (rules
readable). Whichever generalises better under grouped CV is saved.

Grouping matters: frames within one fire sequence are highly correlated, so a
plain k-fold would leak near-duplicate frames across folds and report a score
far above reality. Folds are split by *sequence*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.features import FEATURE_NAMES  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
import joblib  # noqa: E402


def load(path: str, normalise: bool = True):
    """Load features, optionally normalised per sequence.

    Per-sequence normalisation is not a statistical nicety here, it is the main
    finding. Measured per-feature AUC was 0.67-0.73 *within* a sequence but
    collapsed to ~0.53 when pooled across sequences: absolute values of
    ``area_frac`` or ``band_cover`` on a hazy valley camera are nothing like the
    same numbers on a clear ridge camera, so between-camera variance drowns the
    within-camera discrimination the detector actually has.

    Each camera is therefore referenced to its own quiet baseline - the median
    and spread of its earliest frames - which is exactly what a deployed
    per-camera calibration would do.
    """
    X, y, g, meta = [], [], [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            X.append([float(r.get(k, 0.0)) for k in FEATURE_NAMES])
            y.append(int(r["label"]))
            g.append(r["seq"])
            meta.append({"seq": r["seq"], "offset_s": r["offset_s"]})
    X = np.array(X, np.float64); y = np.array(y, int); g = np.array(g)
    if normalise:
        offs = np.array([m["offset_s"] for m in meta], np.float64)
        for s in set(g):
            m = g == s
            # Baseline = this camera's own earliest frames (always pre-ignition).
            base = m & (offs <= np.quantile(offs[m], 0.35))
            if base.sum() < 3:
                base = m
            med = np.median(X[base], axis=0)
            scale = np.median(np.abs(X[base] - med), axis=0) * 1.4826
            scale[scale < 1e-6] = 1.0
            X[m] = (X[m] - med) / scale
    return X, y, g, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="evaluation/design_features.jsonl")
    ap.add_argument("--out", default="evaluation/model.joblib")
    ap.add_argument("--report", default="evaluation/fit_report.json")
    a = ap.parse_args()

    X, y, g, _ = load(a.features)
    n_seq = len(set(g))
    print(f"rows={len(y)} pos={int(y.sum())} neg={int((1-y).sum())} sequences={n_seq}")
    if n_seq < 4 or len(set(y)) < 2:
        print("insufficient data to fit"); return

    folds = min(5, n_seq)
    cv = GroupKFold(n_splits=folds)

    models = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
        "tree3": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=25, class_weight="balanced", random_state=0
        ),
    }

    results = {}
    for name, model in models.items():
        prob = cross_val_predict(model, X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, prob))
        ap_ = float(average_precision_score(y, prob))
        results[name] = {"cv_auc": round(auc, 4), "cv_ap": round(ap_, 4)}
        print(f"{name:8s} grouped-CV AUC={auc:.4f}  AP={ap_:.4f}")

    best = max(results, key=lambda k: results[k]["cv_auc"])
    model = models[best]
    model.fit(X, y)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "kind": best}, a.out)
    print(f"\nbest={best} -> {a.out}")

    explain = {}
    if best == "logreg":
        coefs = model.named_steps["clf"].coef_[0]
        order = np.argsort(-np.abs(coefs))
        explain["coefficients"] = [
            {"feature": FEATURE_NAMES[i], "weight": round(float(coefs[i]), 4)} for i in order
        ]
        print("\nweights (standardised, by |magnitude|):")
        for e in explain["coefficients"]:
            print(f"  {e['feature']:>20}: {e['weight']:+.3f}")
    else:
        explain["tree"] = export_text(model, feature_names=FEATURE_NAMES)
        print("\n" + explain["tree"])

    json.dump({"results": results, "best": best, "explain": explain,
               "n_rows": int(len(y)), "n_sequences": int(n_seq)},
              open(a.report, "w"), indent=1)
    print(f"report -> {a.report}")


if __name__ == "__main__":
    main()
