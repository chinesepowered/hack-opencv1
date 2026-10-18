"""Smoke test: do the committed artefacts still reproduce the reported numbers?

This is the "test instructions" entry point. It needs no imagery and no network:
it re-derives the headline metrics from the committed per-frame feature tables
and the committed model, and fails loudly if any of them has drifted from what
README.md and docs/report.md claim.

    uv run python scripts/selftest.py

Exit code 0 means every reported number was reproduced on this machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, got, want, tol: float = 0.0005) -> None:
    global CHECKS
    CHECKS += 1
    ok = abs(float(got) - float(want)) <= tol if isinstance(want, (int, float)) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, expected {want!r}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import cv2
    import joblib
    from sklearn.metrics import average_precision_score, roc_auc_score

    from sentinel.features import FEATURE_NAMES

    print("environment")
    check("OpenCV major version", cv2.__version__.split(".")[0], "5")
    print(f"  info  OpenCV {cv2.__version__}, numpy {np.__version__}")

    res = json.load(open(ROOT / "evaluation/results.json"))

    # --- re-derive the held-out metrics from the committed feature table ----
    rows = [json.loads(line) for line in open(ROOT / "evaluation/heldout_features.jsonl")]
    X = np.array([[float(r.get(k, 0.0)) for k in FEATURE_NAMES] for r in rows], float)
    y = np.array([int(r["label"]) for r in rows], int)
    g = np.array([r["seq"] for r in rows])
    off = np.array([float(r["offset_s"]) for r in rows], float)

    # Per-camera median/MAD calibration, exactly as evaluate.py does it.
    for s in set(g):
        m = g == s
        base = m & (off <= np.quantile(off[m], 0.35))
        if base.sum() < 3:
            base = m
        med = np.median(X[base], axis=0)
        sc = np.median(np.abs(X[base] - med), axis=0) * 1.4826
        sc[sc < 1e-6] = 1.0
        X[m] = (X[m] - med) / sc

    model = joblib.load(ROOT / "evaluation/model.joblib")["model"]
    prob = model.predict_proba(X)[:, 1]

    print("\nheld-out metrics, recomputed from committed artefacts")
    check("rows", len(rows), res["held_out"]["rows"])
    check("sequences", len(set(g)), res["held_out"]["sequences"])
    check("AUC", round(float(roc_auc_score(y, prob)), 4), res["held_out"]["auc"])
    check("average precision", round(float(average_precision_score(y, prob)), 4),
          res["held_out"]["ap"])

    # --- the operating point quoted everywhere -----------------------------
    row = min(res["sweep"], key=lambda r: abs(r["target_fpr"] - 0.10))
    thr = float(row["threshold"])
    recall = float((prob[y == 1] >= thr).mean())
    print("\noperating point (FPR 0.10)")
    check("threshold", thr, 0.7668)
    check("recall", round(recall, 4), row["recall"])
    check("sequences detected", len({s for s, p, l in zip(g, prob, y) if l == 1 and p >= thr}),
          row["sequences_detected"])
    check("false alarms / camera-day", row["false_alarms_per_camera_day"], 86.4, tol=0.051)

    # --- the agent experiment, both baselines ------------------------------
    print("\nagent layer (measured, not enabled)")
    for baseline, fa_red, recall_cost, pool in (("sliding", 8.33, 28.16, 46.0),
                                                ("anchored", 6.95, 9.34, 22.4)):
        s = json.load(open(ROOT / f"evaluation/suppression_{baseline}.json"))
        check(f"{baseline}: detector FA/camera-day",
              s["before"]["false_alarms_per_camera_day"], 86.4, tol=0.051)
        check(f"{baseline}: false-alarm reduction %", s["false_alarm_reduction_pct"], fa_red,
              tol=0.011)
        check(f"{baseline}: recall cost %", s["recall_cost_pct"], recall_cost, tol=0.011)
        check(f"{baseline}: history pool post-ignition %",
              s["history_pool_post_ignition_pct"], pool, tol=0.051)

    anchored = json.load(open(ROOT / "evaluation/suppression_anchored.json"))
    if anchored["recall_cost_pct"] <= 5.0:
        # If a future change ever clears the bar, this test should force the
        # write-up to be updated rather than silently keep calling it a failure.
        FAILURES.append("the agent layer now clears its 5% bar: update the write-up")
        print("  FAIL  the agent layer now clears its 5% bar: update the write-up")

    # --- the demo pack the UI replays --------------------------------------
    print("\ndemo pack")
    pack_root = ROOT / "web/pack"
    if not (pack_root / "index.json").exists():
        check("pack index present", False, True)
    else:
        index = json.load(open(pack_root / "index.json"))
        check("pack threshold matches operating point", index["threshold"], 0.7668)
        check("hpwren credit present", "hpwren.ucsd.edu" in index["credit"], True)
        for entry in index["sequences"]:
            seq = entry["sequence"]
            meta = json.load(open(pack_root / seq / "pack.json"))
            missing = [f["image"] for f in meta["frames"]
                       if not (pack_root / seq / f["image"]).exists()]
            check(f"{seq}: frame images present", len(missing), 0)
            # Every probability shown in the UI must be the one in the
            # committed table, or the interface is telling a different story
            # from the report.
            table = {(r["seq"], float(r["offset_s"])): i for i, r in enumerate(rows)}
            bad = 0
            for f in meta["frames"]:
                k = table.get((seq, float(f["offset_s"])))
                if k is None:
                    continue
                if f["prob"] is None or abs(f["prob"] - round(float(prob[k]), 4)) > 0.0001:
                    bad += 1
            check(f"{seq}: UI scores match the evaluation table", bad, 0)

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES))
        return 1
    print("OK: every reported number reproduced from the committed artefacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
