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

The two history baselines
-------------------------
``--baseline sliding`` (the original) asks "is this candidate stronger than
anything this camera has shown recently", using every frame older than the
query. On a FIgLib sequence that pool is **66% post-ignition at +2400 s**, so a
growing plume raises its own baseline and then suppresses itself. This is the
same failure as the detector's reference-contamination bug, in a second place.

``--baseline anchored`` restricts the pool to the *oldest* span of available
history for that camera, which a later plume cannot contaminate, mirroring a
deployment that would compare against yesterday's frames at the same hour. It
uses only frame age, never labels.
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

# Anchored baseline: use the oldest ANCHOR_SPAN_S seconds of a camera's
# available history, and never a frame newer than MIN_AGE_S before the query.
ANCHOR_SPAN_S = 1200.0
MIN_AGE_S = 300.0


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
    ap.add_argument("--baseline", choices=("sliding", "anchored"), default="anchored",
                    help="history pool for the compare_history step; see module docstring")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating point; default = the fpr~0.10 point from results.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--traces", default=None)
    a = ap.parse_args()

    out_path = a.out or f"evaluation/suppression_{a.baseline}.json"
    traces_path = a.traces or f"evaluation/traces_{a.baseline}.json"

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
    print(f"operating point threshold = {thr:.4f}   baseline = {a.baseline}")

    # Fast lookup: (sequence, offset) -> model probability
    pmap: dict[tuple[str, float], float] = {}
    for s, o, p in zip(g, off, prob):
        pmap[(s, float(o))] = float(p)
    seq_offsets: dict[str, list[float]] = {}
    label_map: dict[tuple[str, float], int] = {}
    for s, o, lab in zip(g, off, y):
        seq_offsets.setdefault(s, []).append(float(o))
        label_map[(s, float(o))] = int(lab)
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

    # Instrumentation: how much of each history pool is post-ignition? This is
    # the number that explains the sliding baseline's failure, so it is measured
    # rather than asserted.
    pool_frames = {"total": 0, "post_ignition": 0}

    def history(seq: str, offset_s: int) -> list[float]:
        offs = seq_offsets.get(seq, [])
        if not offs:
            return []
        if a.baseline == "sliding":
            pool = [o for o in offs if o < offset_s - MIN_AGE_S]
        else:
            # Anchored: only the oldest span of this camera's history. A plume
            # that appears later cannot raise this baseline. Uses frame age
            # only - no labels - so it is legitimate at inference time.
            span_end = offs[0] + ANCHOR_SPAN_S
            pool = [o for o in offs if o <= span_end and o < offset_s - MIN_AGE_S]
        pool_frames["total"] += len(pool)
        pool_frames["post_ignition"] += sum(label_map.get((seq, o), 0) for o in pool)
        return [pmap.get((seq, o), 0.0) for o in pool]

    policy = VerificationPolicy(rescore=rescore, neighbours=neighbours_of, history=history)

    raw_alerts, verified_alerts = [], []
    traces = []
    blocked = 0
    suppress_rules: dict[str, dict[str, int]] = {}
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
        if outcome.action == "suppress":
            rule = next((t.detail.get("rule", "?") for t in reversed(outcome.trace)
                         if t.kind == "decision"), "guardrail")
            d = suppress_rules.setdefault(rule, {"true_smoke": 0, "clear": 0})
            d["true_smoke" if lab == 1 else "clear"] += 1
        if len(traces) < 400:
            traces.append({"sequence": s, "offset_s": float(o), "label": int(lab),
                           "detector_confidence": round(float(p), 4), **outcome.as_dict()})
        if outcome.action == "escalate":
            verified_alerts.append((s, float(o), float(p), int(lab)))

    def summarise(alerts, name):
        fa = [(s, o) for s, o, _, lab in alerts if lab == 0]
        tp = [(s, o) for s, o, _, lab in alerts if lab == 1]
        all_seqs = sorted(set(g))
        # Exposure counts only the cameras that actually contributed scored
        # pre-ignition frames, which is what evaluate.py does. Counting all 43
        # inflates the denominator and understates the false-alarm rate, and
        # made this script disagree with the headline number at the same
        # threshold (60.3 vs 86.4 per camera-day) for no real reason.
        exposed = [s for s in all_seqs if ((g == s) & (off < 0)).any()]
        pre_hours = len(exposed) * SEQUENCE_HOURS / 2.0
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
    after = summarise(verified_alerts, f"detector_plus_agent_{a.baseline}")
    reduction = (1 - after["false_alarms_per_camera_day"] /
                 max(before["false_alarms_per_camera_day"], 1e-9)) * 100
    recall_cost = (1 - after["recall_frames"] / max(before["recall_frames"], 1e-9)) * 100
    pool_pollution = (100.0 * pool_frames["post_ignition"] / max(pool_frames["total"], 1))

    print(f"\n{'metric':<34}{'detector':>12}{'+ agent':>12}")
    for k in ("alerts_total", "false_alarm_alerts", "false_alarms_per_camera_day",
              "true_positive_alerts", "sequences_detected", "recall_frames",
              "median_latency_s", "p90_latency_s"):
        print(f"{k:<34}{str(before[k]):>12}{str(after[k]):>12}")
    print(f"\nfalse-alarm reduction: {reduction:.1f}%   recall cost: {recall_cost:.1f}%")
    print(f"history pool was {pool_pollution:.1f}% post-ignition frames")
    print(f"guardrail blocks: {blocked} of {len(raw_alerts)} alerts "
          f"({100.0 * blocked / max(len(raw_alerts), 1):.1f}%)")
    if suppress_rules:
        print("\nsuppressions by rule (true smoke / clear):")
        for rule, d in sorted(suppress_rules.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"  {rule:<32} {d['true_smoke']:>5} / {d['clear']:>5}")

    json.dump({"baseline": a.baseline, "threshold": thr, "before": before, "after": after,
               "false_alarm_reduction_pct": round(float(reduction), 2),
               "recall_cost_pct": round(float(recall_cost), 2),
               "history_pool_post_ignition_pct": round(float(pool_pollution), 1),
               "guardrail_blocks": blocked,
               "guardrail_block_rate_pct": round(100.0 * blocked / max(len(raw_alerts), 1), 1),
               "alerts_considered": len(raw_alerts),
               "suppressions_by_rule": suppress_rules},
              open(out_path, "w"), indent=1)
    json.dump(traces, open(traces_path, "w"), indent=1)
    print(f"-> {out_path} and {traces_path} ({len(traces)} traces)")


if __name__ == "__main__":
    main()
