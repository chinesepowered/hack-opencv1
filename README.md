# Smoke Sentinel

Early wildfire-smoke detection on fixed camera networks, built with **OpenCV 5** and measured
honestly on held-out cameras.

**All rights reserved.** This repository is published for judging in the OpenCV AI Competition
2026. No open-source licence is granted; see *Licence* below.

Camera imagery: **HPWREN FIgLib**, credit reference <https://www.hpwren.ucsd.edu/> as their terms
require. No original FIgLib frame is redistributed here; the committed figures and demo pack are
derived, annotated renderings, which their terms permit with that credit. See
[`docs/DATA.md`](docs/DATA.md).

**Full technical report: [`docs/report.md`](docs/report.md).**

---

## The problem

A wildfire is cheapest to fight in its first twenty minutes. Fixed fire-watch cameras already look
at the right ridgelines around the clock, but somebody has to be watching the right one of hundreds
of feeds at the right moment. Smoke in those first minutes is not an object you can draw a box
around. It is a faint, low-contrast, slowly spreading change in texture, and it looks a great deal
like fog, dust, haze and the cloud shadow moving across the same hillside.

This project treats that as what it is: a **temporal change-detection problem**, not an object
detection problem, and reports what classical computer vision can and cannot do about it.

![Pipeline](docs/architecture.png)

## What it does

1. **Builds a clean reference** for each camera from its own earlier frames, anchored to the oldest
   available history so a developing plume cannot contaminate its own baseline (see *Bug 1*).
2. **Measures the change** between the current frame and that reference: signed luminance residual,
   texture-energy drop, desaturation, connected-component geometry, vertical band coverage, growth
   rate and drift coherence.
3. **Scores the frame** with a depth-3 decision tree fitted on design cameras only, never on the
   held-out set.
4. **Escalates to a human** with the evidence. The system never dispatches anything.

## Results

Held out: **43 cameras never seen during design**, 2,261 frames, model fitted on 35 design
sequences only.

| Metric | Value |
| --- | --- |
| **Held-out AUC** | **0.846** |
| Held-out average precision | 0.906 |
| Naive baseline (absolute luminance residual) | 0.440 |
| Design grouped-CV AUC (depth-3 tree) | 0.837 |
| Per-sequence AUC | median 0.937, 26 of 30 above 0.70 |

Sensitivity sweep, with detection latency measured in real seconds from the moment a human
annotator marked the plume visible:

| Target FPR | Recall | Sequences detected | Median latency | False alarms / camera-day |
| --- | --- | --- | --- | --- |
| 0.05 | 0.489 | 36 / 43 | 1110 s | 20.4 |
| 0.10 | 0.735 | **43 / 43** | 420 s | 86.4 |
| 0.20 | 0.796 | 43 / 43 | 240 s | 163.2 |

**The honest headline: the ranking is good, the operating point is not deployable.** At the
sensitivity where every sequence is caught, the system also cries wolf about 86 times per camera
per day. A real fire-watch desk needs well under one. That gap is the finding, not a footnote, and
it is what the agent layer was built to close.

## The agent layer is a measured negative result

![Verification loop](docs/agent-workflow.png)

A verification loop sits behind the detector: an attention floor, a comparison against the camera's
own history, a temporal re-examination of real adjacent frames, a check against a neighbouring
camera that saw the same event, and deterministic guardrails the model cannot argue past.

It had a bar set before it was measured: materially fewer false alarms for **under 5% of recall**.
On the held-out split, at the same threshold:

| History pool | Pool post-ignition | False alarms / camera-day | Recall | FA reduction | Recall cost |
| --- | --- | --- | --- | --- | --- |
| Detector only | - | 86.4 | 0.735 | - | - |
| + agent, sliding | 46.0% | 79.2 | 0.528 | 8.3% | **28.2%** |
| + agent, anchored | 22.4% | 80.4 | 0.666 | 7.0% | **9.3%** |

Finding the contaminated history pool and fixing it cut the recall cost from 28.2% to 9.3%, **and it
still misses the bar**. Seven percent fewer false alarms for 9.3% less recall is not a trade a
fire-watch desk should take, so the loop is **disabled** in the reported operating point and written
up as a negative result in [report section 7](docs/report.md#7-the-agent-layer-a-measured-negative-result).

The diagnosis generalises: every verification step reads the *same measurement* the detector already
read, so a frame the detector gets wrong is one the loop gets wrong too. A second look needs a second
*kind* of evidence.

## Three bugs, and they are all the same bug

Found by measurement disagreeing with expectation, not by inspection.

**Bug 1 - the reference contaminated itself.** The sliding baseline window moved forward with the
query frame, so by +2400 s the "clean" reference was **83% post-ignition frames**. The detector was
subtracting the plume from itself, worst exactly when the fire was largest. This inverted the
strongest feature (signed luminance separated the wrong way, Cohen's d = -1.05) and reversed the
growth term. Anchoring the reference to the oldest history fixed it. Progression as it was fixed:
AUC 0.50 -> 0.74 -> 0.82 -> 0.85.

**Bug 2 - per-scene offsets swamped the signal.** Features scored 0.67-0.73 AUC *within* a
sequence but ~0.53 pooled, because an absolute band-coverage number on a hazy valley camera means
something different from the same number on a clear ridge. Per-camera baseline normalisation
(median/MAD of each camera's own earliest frames) fixed it, which is what a deployed per-camera
calibration would do anyway.

**Bug 3 - the agent's history pool contaminated itself too.** Same class as Bug 1, in a second
place: the pool was 66% post-ignition at +2400 s, so the growing plume raised its own baseline and
was then suppressed for not beating it. That one rule suppressed **326 true-smoke frames against 6
clear ones**. Anchoring the pool cut it to 108 against 5.

Any time a system compares "now" against "recently", on data where the event of interest persists,
the reference is a candidate for contamination.

## Demo interface

```bash
uv run python -m http.server 3041 --directory web
```

A static 1280x800 page that **replays a real run** on three held-out cameras: the annotated frame
with its residual mask, the detector score against the operating point, the operator card a human
would receive, and the full verification trace.

Every probability and every trace line is read from the committed `evaluation/` artefacts, so what
is on screen is bit-identical to the numbers above. Only the imagery is re-rendered.
`scripts/build_demo_pack.py` builds the pack.

One of the three cameras is `20170722_FIRE_so-s-mobo-c`, where the loop suppresses 11 frames of
**real smoke**. The card says so on those frames rather than quietly hiding the suppression.

## OpenCV 5 implementation

| Stage | OpenCV 5 API | File |
| --- | --- | --- |
| Horizon estimation | `Sobel`, `magnitude`, `GaussianBlur` | `src/sentinel/detector.py` |
| Reference building | `resize`, median stacking, photometric match | `src/sentinel/background.py` |
| Drift alignment | `findTransformECC` | `src/sentinel/background.py` |
| Residual + masking | `cvtColor`, `morphologyEx`, `getStructuringElement` | `src/sentinel/detector.py` |
| Texture energy | `Sobel`, `boxFilter` | `src/sentinel/detector.py` |
| Region geometry | `connectedComponentsWithStats` | `src/sentinel/detector.py` |
| Edge density | `Canny` | `src/sentinel/features.py` |
| Overlays | `rectangle`, `putText`, `line`, `addWeighted` | `src/sentinel/detector.py` |

OpenCV 5.0.0 installs from PyPI as `opencv-python-headless==5.0.0.93`; no source build is needed.

## Reproducing the numbers

```bash
uv sync
uv run python scripts/screen_catalog.py           # screen the catalogue -> evaluation/manifest.json
uv run python scripts/extract_features.py --split design_set --out evaluation/design_features.jsonl
uv run python scripts/extract_features.py --split held_out   --out evaluation/heldout_features.jsonl
uv run python scripts/fit_classifier.py           # fit on design sequences only
uv run python scripts/evaluate.py                 # held-out metrics -> evaluation/results.json
uv run python scripts/suppression_experiment.py --baseline sliding
uv run python scripts/suppression_experiment.py --baseline anchored
uv run python scripts/make_diagrams.py            # the figures above
uv run python scripts/build_demo_pack.py          # the demo UI's replay pack
```

To check every number in this README on your own machine in about twenty seconds, with no imagery
and no network:

```bash
uv run python scripts/selftest.py
```

It recomputes the held-out metrics, the operating point and both agent-layer runs from the committed
artefacts, and confirms the demo UI is displaying exactly the probabilities the evaluation produced.
25 checks; exit code 0 means everything reported here reproduced.

Only the two `extract_features.py` steps need imagery; they download into a git-ignored cache and
take about forty minutes per split. Everything else runs in seconds from the **committed** per-frame
feature tables in `evaluation/`, so every number here can be recomputed without re-downloading a
single image.

## AWS: designed, not deployed

**This project was never deployed to AWS, and nothing in this repository should be read as
claiming otherwise.** The competition's $150 compute-grant round closed before we could enter it,
and we chose not to put a personal credit card behind a student competition entry. The deployment
is therefore **designed and written as infrastructure-as-code, and never applied**: see
[`docs/deploy.md`](docs/deploy.md) and `infra/`. Every measurement in this README and in the
technical report was produced locally. There are no cloud benchmarks, no Graviton numbers and no
live endpoint, because we did not run any.

## Limitations

- The operating point is not deployable at ~86 false alarms per camera-day. See the report.
- Only **78 of 445** FIgLib sequences (~18%) carry usable classical-CV signal under our screening;
  the rest are too diffuse, lack desaturation, or are night/dawn exposures.
- Latency is measured against a human annotator's "plume visible" mark, which is itself a judgement.
- FIgLib is curated around real ignitions, so a real deployment sees far more clear-sky camera-hours
  than this data contains. The false-alarm rate above is a *lower* bound.
- Daytime only. Night ignition detection is a different problem and is out of scope.

## Licence

**All rights reserved.** The competition permits a judge-accessible repository that is not open
source, and this one is published on that basis. No permission is granted to use, copy, modify or
distribute this code. FIgLib imagery retains its own terms; see [`docs/DATA.md`](docs/DATA.md).

Built with Claude Code as a coding assistant.
