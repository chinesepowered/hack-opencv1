"""Extract per-frame classical features for a split. Writes JSONL."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import data, background, features

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="design_set")
ap.add_argument("--out", required=True)
ap.add_argument("--max-seqs", type=int, default=0)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

man = json.load(open("evaluation/manifest.json"))
seqs = man[a.split]
if a.max_seqs: seqs = seqs[:a.max_seqs]
bcfg = background.BackgroundConfig()
done_seqs = set()
if a.resume:
    try:
        for line in open(a.out):
            done_seqs.add(json.loads(line)["seq"])
    except FileNotFoundError:
        pass
    print(f"resume: {len(done_seqs)} sequences already extracted", flush=True)
out = open(a.out, "a" if a.resume else "w")
t0 = time.time(); nrows = 0
for si, seq in enumerate(seqs, 1):
    if seq in done_seqs:
        continue
    try:
        fr = data.list_frames(seq)
        by = {f.offset_s: f for f in fr}
        offs = sorted(by)
        imgs = {}
        for o in offs:
            try: imgs[o] = data.load(by[o])
            except Exception: pass
        offs = [o for o in offs if o in imgs]
        n_seq = 0
        for o in offs:
            hist = [(t, imgs[t]) for t in offs if t < o]
            ref = background.build_reference(hist, o, bcfg)
            f = features.extract(imgs[o], ref, history=hist, now_s=o)
            if f is None: continue
            f["seq"] = seq; f["offset_s"] = o; f["label"] = int(o >= 0)
            out.write(json.dumps(f) + "\n"); n_seq += 1; nrows += 1
        out.flush()
        print(f"[{si}/{len(seqs)}] {seq} rows={n_seq} total={nrows} {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{si}/{len(seqs)}] {seq} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
out.close()
print(f"DONE rows={nrows} in {time.time()-t0:.0f}s -> {a.out}")
