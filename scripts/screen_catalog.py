"""Screen every FIgLib sequence for usable smoke signal.

Pure data triage: computed from raw frames only, never from detector output, so
it cannot flatter the detector. Writes evaluation/manifest.json.
"""
import sys, json, time, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import data, detector
import cv2, numpy as np

def screen(seq):
    fr = data.list_frames(seq)
    if len(fr) < 45: return None
    by = {f.offset_s: f for f in fr}; offs = sorted(by)
    a = cv2.resize(data.load(by[offs[0]]), (640, 480))
    b = cv2.resize(data.load(by[offs[-1]]), (640, 480))
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b)]
    dmean = float(gb.mean() - ga.mean())
    darkest = float(min(ga.mean(), gb.mean()))
    gbn = gb - dmean
    hz_full = detector.estimate_horizon(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    hz = int(hz_full * 480 / a.shape[0]) if a.shape[0] != 480 else hz_full
    top, bot = max(0, hz - 60), min(480, hz + 40)
    d = np.abs(ga - gbn); d[np.maximum(ga, gb) > 225] = 0
    band = d[top:bot, :]
    if band.size == 0: return None
    ch, cw = max(band.shape[0] // 4, 1), max(band.shape[1] // 16, 1)
    grid = np.array([band[r*ch:(r+1)*ch, c*cw:(c+1)*cw].mean() for r in range(4) for c in range(16)])
    conc = float(grid.max() / (np.median(grid) + 1e-6))
    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    i = int(np.argmax(grid)); r_, c_ = i // 16, i % 16
    ys, xs = slice(top + r_*ch, top + (r_+1)*ch), slice(c_*cw, (c_+1)*cw)
    satd = float((hsv_a[ys, xs] - hsv_b[ys, xs]).mean())
    reasons = []
    if abs(dmean) >= 25: reasons.append("exposure_or_dawn")
    if darkest <= 45: reasons.append("night")
    if satd <= 3: reasons.append("no_desaturation")
    if conc <= 4: reasons.append("diffuse")
    return dict(seq=seq, n=len(fr), conc=round(conc, 2), hottest=round(float(grid.max()), 2),
                satdrop=round(satd, 2), dmean=round(dmean, 2), darkest=round(darkest, 1),
                usable=not reasons, reasons=reasons)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="evaluation/manifest.json")
    args = ap.parse_args()
    seqs = data.list_sequences()
    if args.limit: seqs = seqs[:args.limit]
    rows, t0 = [], time.time()
    for i, s in enumerate(seqs, 1):
        try:
            r = screen(s)
            if r:
                rows.append(r)
                if r["usable"]: print(f"[{i}/{len(seqs)}] USABLE {s} conc={r['conc']} sat+={r['satdrop']}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(seqs)}] ERR {s} {type(e).__name__}", flush=True)
    use = sorted([r for r in rows if r["usable"]], key=lambda r: -(r["conc"] * r["satdrop"]))
    rng = random.Random(1337); rng.shuffle(use)
    split = int(len(use) * 0.4)
    manifest = dict(
        source="HPWREN FIgLib", credit=data.CREDIT,
        screened=len(rows), usable=len(use),
        design_set=[r["seq"] for r in use[:split]],
        held_out=[r["seq"] for r in use[split:]],
        hard_subset=[r["seq"] for r in rows if not r["usable"]][:40],
        rows=rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(args.out, "w"), indent=1)
    print(f"\nscreened {len(rows)} in {time.time()-t0:.0f}s | usable {len(use)} "
          f"| design {split} | held-out {len(use)-split} -> {args.out}")
