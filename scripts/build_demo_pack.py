"""Build the static demo pack the web UI replays.

Why precompute instead of serving a live pipeline
-------------------------------------------------
One frame costs an ECC alignment, a photometric fit, two texture passes and a
median reference build: about 1.5 s. A browser cannot wait 1.5 s per frame and a
headless recording must be deterministic, so the pipeline runs here, once, and
the UI replays exactly what it produced. Nothing is simulated.

Where the numbers come from
---------------------------
Scores and agent decisions are **not** recomputed. They are taken from the
committed artefacts - ``evaluation/heldout_features.jsonl`` and
``evaluation/model.joblib`` - normalised with the same per-camera median/MAD used
in ``evaluate.py`` and ``suppression_experiment.py``. The probability shown for a
frame in the UI is therefore bit-identical to the one behind the headline AUC,
and the agent trace is identical to ``evaluation/traces_anchored.json``.

The only thing recomputed here is the **imagery**: the residual mask and the
candidate boxes, which are not stored in the feature table and have to be
rendered from the cached frames.

Usage::

    uv run python scripts/build_demo_pack.py              # default 3 sequences
    uv run python scripts/build_demo_pack.py --sequences A B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentinel import background, data, detector  # noqa: E402
from sentinel.agent import VerificationPolicy, guardrails  # noqa: E402
from sentinel.features import FEATURE_NAMES  # noqa: E402

# Chosen from evaluation/traces_anchored.json so the three panels show three
# different behaviours rather than three copies of the easy case:
#   sm-n   - escalation carried by cross-camera corroboration
#   so-s   - the loop suppressing, including true smoke it should not have
#   sp-n   - a strong plume the detector gets on its own, no corroboration
DEFAULT_SEQUENCES = [
    "20190924_FIRE_sm-n-mobo-c",
    "20170722_FIRE_so-s-mobo-c",
    "20180719_Skyline_sp-n-mobo-c",
]

ANCHOR_SPAN_S = 1200.0
MIN_AGE_S = 300.0
OUT_WIDTH = 640


def load_features(path: Path):
    """Per-camera normalised feature matrix, exactly as the evaluation does."""
    X, y, g, off = [], [], [], []
    rows = []
    for line in open(path):
        r = json.loads(line)
        rows.append(r)
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
    return X, y, g, off, rows


def cached_frames(seq: str) -> list[data.Frame]:
    """Frames already on disk, without touching the network."""
    d = data.CACHE / seq
    out = []
    for p in sorted(d.glob("*.jpg")):
        o = data._parse_offset(p.name)
        if o is not None and p.stat().st_size > 0:
            out.append(data.Frame(sequence=seq, name=p.name, offset_s=o))
    out.sort(key=lambda f: f.offset_s)
    return out


def render(frame: np.ndarray, res: detector.DetectionResult) -> np.ndarray:
    """Annotated frame with the residual mask tinted over it."""
    img = detector._to_working(frame, detector.WORK_WIDTH).copy()
    if res.mask is not None and res.mask.shape[:2] == img.shape[:2]:
        tint = img.copy()
        tint[res.mask > 0] = (0, 90, 255)
        img = cv2.addWeighted(img, 0.72, tint, 0.28, 0)
    cv2.line(img, (0, res.horizon_y), (img.shape[1], res.horizon_y), (90, 180, 255), 1, cv2.LINE_AA)
    for i, c in enumerate(res.candidates[:3]):
        colour = (40, 90, 255) if i == 0 else (120, 200, 255)
        cv2.rectangle(img, (c.x, c.y), (c.x + c.w, c.y + c.h), colour, 2)
    h, w = img.shape[:2]
    return cv2.resize(img, (OUT_WIDTH, int(round(h * OUT_WIDTH / w))), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=DEFAULT_SEQUENCES)
    ap.add_argument("--heldout", default="evaluation/heldout_features.jsonl")
    ap.add_argument("--model", default="evaluation/model.joblib")
    ap.add_argument("--neighbours", default="evaluation/neighbours.json")
    ap.add_argument("--out", default="web/pack")
    a = ap.parse_args()

    X, y, g, off, rows = load_features(ROOT / a.heldout)
    model = joblib.load(ROOT / a.model)["model"]
    prob = model.predict_proba(X)[:, 1]
    nbrs = json.load(open(ROOT / a.neighbours))

    res_json = json.load(open(ROOT / "evaluation/results.json"))
    thr = float(min(res_json["sweep"], key=lambda r: abs(r["target_fpr"] - 0.10))["threshold"])

    # Lookups over the whole held-out split: the agent's second looks reach other
    # cameras, so they need the full map, not just the demo sequences.
    pmap = {(s, float(o)): float(p) for s, o, p in zip(g, off, prob)}
    label_map = {(s, float(o)): int(l) for s, o, l in zip(g, off, y)}
    seq_offsets: dict[str, list[float]] = {}
    for s, o in zip(g, off):
        seq_offsets.setdefault(s, []).append(float(o))
    for s in seq_offsets:
        seq_offsets[s].sort()

    def rescore(seq, offset_s, sensitivity):
        offs = seq_offsets.get(seq)
        if not offs:
            return {"confidence": 0.0}
        near = [o for o in offs if abs(o - offset_s) <= 120.0]
        if not near:
            k = min(offs, key=lambda o: abs(o - offset_s))
            return {"confidence": pmap.get((seq, k), 0.0)}
        return {"confidence": float(np.median([pmap.get((seq, o), 0.0) for o in near]))}

    def history(seq, offset_s):
        offs = seq_offsets.get(seq, [])
        if not offs:
            return []
        span_end = offs[0] + ANCHOR_SPAN_S
        pool = [o for o in offs if o <= span_end and o < offset_s - MIN_AGE_S]
        return [pmap.get((seq, o), 0.0) for o in pool]

    policy = VerificationPolicy(rescore=rescore, neighbours=lambda s: list(nbrs.get(s, [])),
                                history=history)

    out_root = ROOT / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    bcfg = background.BackgroundConfig()
    dcfg = detector.DetectorConfig()
    index = []

    for seq in a.sequences:
        t0 = time.time()
        frames = cached_frames(seq)
        if not frames:
            print(f"SKIP {seq}: nothing cached")
            continue
        seq_dir = out_root / seq
        seq_dir.mkdir(parents=True, exist_ok=True)

        imgs: dict[int, np.ndarray] = {}
        for f in frames:
            try:
                imgs[f.offset_s] = data.load(f)
            except Exception:
                pass
        offs = sorted(imgs)

        # Scored frames for this sequence, from the committed table.
        scored = {float(o): float(p) for s, o, p in zip(g, off, prob) if s == seq}

        out_frames = []
        prev_mask = None
        for o in offs:
            hist = [(t, imgs[t]) for t in offs if t < o]
            ref = background.build_reference(hist, o, bcfg)
            res = detector.detect(imgs[o], ref, dcfg, prev_mask)
            prev_mask = res.mask if res.mask is not None else prev_mask
            name = f"f{len(out_frames):03d}.jpg"
            cv2.imwrite(str(seq_dir / name), render(imgs[o], res),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 72])

            p = scored.get(float(o))
            rec = {
                "i": len(out_frames),
                "offset_s": int(o),
                "label": int(o >= 0),
                "image": name,
                "scored": p is not None,
                "prob": round(p, 4) if p is not None else None,
                "cv_score": round(res.score, 4),
                "has_reference": res.has_reference,
                "boxes": [[c.x, c.y, c.w, c.h] for c in res.candidates[:3]],
                "agent": None,
            }
            if p is not None and p >= thr:
                outcome = policy.run(seq, int(o), p)
                ok, why = guardrails(outcome)
                corrected = None
                if not ok:
                    outcome.action = "suppress" if outcome.action == "escalate" else "escalate"
                    outcome.reason = f"guardrail correction ({why})"
                    corrected = why
                d = outcome.as_dict()
                d["guardrail"] = corrected
                rec["agent"] = d
            out_frames.append(rec)

        pos = [f for f in out_frames if f["label"] == 1 and f["prob"] is not None]
        alerts = [f for f in out_frames if f["prob"] is not None and f["prob"] >= thr]
        esc = [f for f in alerts if f["agent"] and f["agent"]["action"] == "escalate"]
        meta = {
            "sequence": seq,
            "frames": out_frames,
            "threshold": round(thr, 4),
            "image_width": OUT_WIDTH,
            "stats": {
                "frames": len(out_frames),
                "scored": sum(1 for f in out_frames if f["scored"]),
                "positives": len(pos),
                "detector_alerts": len(alerts),
                "detector_false_alarms": sum(1 for f in alerts if f["label"] == 0),
                "agent_escalations": len(esc),
                "agent_suppressions": len(alerts) - len(esc),
                "agent_suppressed_true_smoke": sum(
                    1 for f in alerts if f["label"] == 1 and f["agent"]
                    and f["agent"]["action"] == "suppress"),
                "neighbours": list(nbrs.get(seq, []))[:2],
            },
        }
        json.dump(meta, open(seq_dir / "pack.json", "w"), indent=1)
        index.append({"sequence": seq, "frames": len(out_frames), **meta["stats"]})
        print(f"{seq}: {len(out_frames)} frames, {len(alerts)} alerts, "
              f"{len(esc)} escalations in {time.time() - t0:.0f}s")

    json.dump({"sequences": index, "threshold": round(thr, 4),
               "credit": data.CREDIT},
              open(out_root / "index.json", "w"), indent=1)
    print(f"-> {out_root}/index.json")


if __name__ == "__main__":
    main()
