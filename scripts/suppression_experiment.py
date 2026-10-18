"""Does the agentic verification loop actually suppress false alarms?

This is a hypothesis under test, not a claim. The detector alone fires ~86
alerts per camera-day at the operating point where every held-out fire is
caught, which is operationally unusable. The agent layer is supposed to fix
that. This script measures whether it does, on the held-out split only, and
reports recall and detection latency alongside so that any recall sacrificed to
verification is visible rather than hidden.

Honesty notes on what the verification steps actually do
-------------------------------------------------------
Features were extracted once per frame at a single detector sensitivity.
Re-extracting the whole split at further sensitivities would cost about forty
minutes per pass, so the loop's "look again" steps use real additional data
rather than a simulated re-run:

* **Temporal re-examination** scores the *adjacent real frames* (+/-120 s from
  the same camera) and takes their agreement. A genuine plume persists across
  neighbouring minutes; transient clutter does not.
* **Neighbour corroboration** scores a *different real camera* that observed the
  same event on the same date, at the nearest available offset.

Both are genuine second looks at genuine imagery. Neither is described as a
sensitivity re-run, because that is not what they are.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.agent import VerificationPolicy, guardrails  # noqa: E402
from sentinel.features import FEATURE_NAMES  # noqa: E402

import joblib  # noqa: E402

SEQUENCE_HOURS = 80.0 / 60.0


def load(path: str):
    X, y, g, off = [], [], [], []
    for line in open(path):
        r = json.loads(line)
        X.append([float(r.get(k, 0.0)) for k in FEATURE_NAMES])
        y.append(int(r["label"])); g.append(r["seq"]); off.append(float(r["offset_s"]))
    X = np.array(X, float); y = np.array(y, int); g = np.array(g); off = np.array(off, float)
    for s in set(g):
        m = g == s
        base = m & (off <= np.quantile(off[m], 0.35))
        if base.sum() < 3:
            base = m
        med = np.median(X[base], axis=0)
        sc = np.median(np.abs(X[base] - med), axis=0) * 1.4826
        sc[sc < 1e-6] = 1.0
        X[m] = (X[m] - med) / sc
    return X, y, g, off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="evaluation/heldout_features.jsonl")
    ap.add_argument("--model", default="evaluation/model.joblib")
    ap.add_argument("--neighbours", default="evaluation/neighbours.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating point; default = the fpr~0.10 point from results.json")
    ap.add_argument("--out", default="evaluation/suppression.json")
    ap.add_argument("--traces", default="evaluation/traces.json")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model = bundle["model"]
    X, y, g, off = load(a.heldout)
    prob = model.predict_proba(X)[:, 1]
    nbrs = json.load(open(a.neighbours))

    thr = a.threshold
    if thr is None:
        res = json.load(open("evaluation/results.json"))
        row = min(res["sweep"], key=lambda r: abs(r["target_fpr"] - 0.10))
        thr = float(row["threshold"])
    print(f"operating point threshold = {thr:.4f}")

    # Fast lookup: (sequence, offset) -> model probability
    pmap: dict[tuple[str, float], float] = {}
    for s, o, p in zip(g, off, prob):
        pmap[(s, float(o))] = float(p)
    seq_offsets: dict[str, list[float]] = {}
    for s, o in zip(g, off):
        seq_offsets.setdefault(s, []).append(float(o))
    for s in seq_offsets:
        seq_offsets[s].sort()

    def rescore(seq: str, offset_s: int, sensitivity: float) -> dict:
        """Temporal re-examination: agreement of real adjacent frames."""
        offs = seq_offsets.get(seq)
        if not offs:
            return {"confidence": 0.0}
        near = [o for o in offs if abs(o - offset_s) <= 120.0]
        if not near:
            k = min(offs, key=lambda o: abs(o - offset_s))
            return {"confidence": pmap.get((seq, k), 0.0)}
        vals = [pmap.get((seq, o), 0.0) for o in near]
        # Persistence: the median across the local window, not the peak, so a
        # single spiking frame cannot carry a candidate through.
        return {"confidence": float(np.median(vals))}

    def neighbours_of(seq: str) -> list[str]:
        return list(nbrs.get(seq, []))

    def history(seq: str, offset_s: int) -> list[float]:
        """This camera's own earlier frames, as its routine-noise baseline."""
        offs = seq_offsets.get(seq, [])
        return [pmap.get((seq, o), 0.0) for o in offs if o < offset_s - 300]

    policy = VerificationPolicy(rescore=rescore, neighbours=neighbours_of, history=history)

    raw_alerts, verified_alerts = [], []
    traces = []
    blocked = 0
    for s, o, p, lab in zip(g, off, prob, y):
        if p < thr:
            continue
        raw_alerts.append((s, float(o), float(p), int(lab)))
        outcome = policy.run(s, int(o), float(p))
        ok, why = guardrails(outcome)
        if not ok:
            blocked += 1
            # A guardrail states what the policy may NOT do; the correction is
            # the opposite of the rejected action, not a blanket suppression.
            # An earlier version forced `suppress` on every violation, which
            # inverted the "cannot suppress a very high confidence candidate"
            # rule and silently discarded 145 candidates at confidence ~0.98 -
            # most of the apparent recall collapse was this bug, not the loop.
            outcome.action = "suppress" if outcome.action == "escalate" else "escalate"
            outcome.reason = f"guardrail correction ({why})"
        if len(traces) < 400:
            traces.append({"sequence": s, "offset_s": float(o), "label": int(lab),
                           "detector_confidence": round(float(p), 4), **outcome.as_dict()})
        if outcome.action == "escalate":
            verified_alerts.append((s, float(o), float(p), int(lab)))

    def summarise(alerts, name):
        seqs = sorted({s for s, _, _, _ in alerts})
        fa = [(s, o) for s, o, _, lab in alerts if lab == 0]
        tp = [(s, o) for s, o, _, lab in alerts if lab == 1]
        all_seqs = sorted(set(g))
        pre_hours = len(all_seqs) * SEQUENCE_HOURS / 2.0
        fa_day = len(fa) / (pre_hours / 24.0) if pre_hours else float("nan")
        detected = sorted({s for s, _, _, lab in alerts if lab == 1})
        lat = []
        for s in detected:
            fired = [o for ss, o, _, lab in alerts if ss == s and lab == 1 and o >= 0]
            if fired:
                lat.append(min(fired))
        pos_total = int((y == 1).sum())
        return {
            "name": name,
            "alerts_total": len(alerts),
            "true_positive_alerts": len(tp),
            "false_alarm_alerts": len(fa),
            "false_alarms_per_camera_day": round(float(fa_day), 2),
            "sequences_detected": len(detected),
            "sequences_total": len(all_seqs),
            "recall_frames": round(len(tp) / max(pos_total, 1), 4),
            "median_latency_s": round(float(np.median(lat)), 1) if lat else None,
            "p90_latency_s": round(float(np.percentile(lat, 90)), 1) if lat else None,
        }

    before = summarise(raw_alerts, "detector_only")
    after = summarise(verified_alerts, "detector_plus_agent")
    reduction = (1 - after["false_alarms_per_camera_day"] /
                 max(before["false_alarms_per_camera_day"], 1e-9)) * 100

    print(f"\n{'metric':<34}{'detector':>12}{'+ agent':>12}")
    for k in ("alerts_total", "false_alarm_alerts", "false_alarms_per_camera_day",
              "true_positive_alerts", "sequences_detected", "recall_frames",
              "median_latency_s", "p90_latency_s"):
        print(f"{k:<34}{str(before[k]):>12}{str(after[k]):>12}")
    print(f"\nfalse-alarm reduction: {reduction:.1f}%")
    print(f"guardrail blocks: {blocked}")

    json.dump({"threshold": thr, "before": before, "after": after,
               "false_alarm_reduction_pct": round(float(reduction), 2),
               "guardrail_blocks": blocked},
              open(a.out, "w"), indent=1)
    json.dump(traces, open(a.traces, "w"), indent=1)
    print(f"-> {a.out} and {a.traces} ({len(traces)} traces)")


if __name__ == "__main__":
    main()
