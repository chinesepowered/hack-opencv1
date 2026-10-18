# Smoke Sentinel

Early wildfire-smoke detection on fixed camera networks, built with **OpenCV 5** and measured
honestly on held-out cameras.

**All rights reserved.** This repository is published for judging in the OpenCV AI Competition
2026. No open-source licence is granted; see *Licence* below.

Camera imagery: **HPWREN FIgLib**, credit reference <https://www.hpwren.ucsd.edu/> as their terms
require. No FIgLib imagery is redistributed here; see [`docs/DATA.md`](docs/DATA.md).

---

## The problem

A wildfire is cheapest to fight in its first twenty minutes. Fixed fire-watch cameras already look
at the right ridgelines around the clock, but somebody has to be watching the right one of hundreds
of feeds at the right moment. Smoke in those first minutes is not an object you can draw a box
around. It is a faint, low-contrast, slowly spreading change in texture, and it looks a great deal
like fog, dust, haze and the cloud shadow moving across the same hillside.

This project treats that as what it is: a **temporal change-detection problem**, not an object
detection problem, and reports what classical computer vision can and cannot do about it.

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

## Two bugs worth reading about

Both were found by measurement disagreeing with expectation, not by inspection.

**Bug 1 — the reference contaminated itself.** The sliding baseline window moved forward with the
query frame, so by +2400 s the "clean" reference was **83% post-ignition frames**. The detector was
subtracting the plume from itself, worst exactly when the fire was largest. This inverted the
strongest feature (signed luminance separated the wrong way, Cohen's d = −1.05) and reversed the
growth term. Anchoring the reference to the oldest history fixed it. Progression as it was fixed:
AUC 0.50 → 0.74 → 0.82 → 0.85.

**Bug 2 — per-scene offsets swamped the signal.** Features scored 0.67–0.73 AUC *within* a
sequence but ~0.53 pooled, because an absolute band-coverage number on a hazy valley camera means
something different from the same number on a clear ridge. Per-camera baseline normalisation
(median/MAD of each camera's own earliest frames) fixed it, which is what a deployed per-camera
calibration would do anyway.

## OpenCV 5 implementation

| Stage | OpenCV 5 API | File |
| --- | --- | --- |
| Horizon estimation | `Sobel`, `magnitude`, `GaussianBlur` | `src/sentinel/detector.py` |
| Reference building | `resize`, median stacking, photometric match | `src/sentinel/background.py` |
| Drift alignment | `findTransformECC` | `src/sentinel/background.py` |
| Residual + masking | `cvtColor`, `morphologyEx`, `getStructuringElement` | `src/sentinel/detector.py` |
| Texture energy | `Sobel`, `boxFilter` | `src/sentinel/detector.py` |
| Region geometry | `connectedComponentsWithStats` | `src/sentinel/detector.py` |
| Overlays | `rectangle`, `putText`, `line` | `src/sentinel/detector.py` |

OpenCV 5.0.0 installs from PyPI as `opencv-python-headless==5.0.0.93`; no source build is needed.

## Reproducing the numbers

```bash
uv sync
uv run python scripts/screen_catalog.py      # screen the FIgLib catalogue -> evaluation/manifest.json
uv run python scripts/extract_features.py    # cache frames, extract features (downloads imagery)
uv run python scripts/fit_classifier.py      # fit on design sequences only
uv run python scripts/evaluate.py            # held-out metrics -> evaluation/results.json
uv run python scripts/suppression_experiment.py   # agent-layer experiment -> evaluation/suppression.json
```

Committed outputs in `evaluation/` are the real artefacts of those runs, including the per-frame
feature tables, so the headline numbers can be recomputed without re-downloading imagery.

## AWS: designed, not deployed

**This project was never deployed to AWS, and nothing in this repository should be read as
claiming otherwise.** The competition's $150 compute-grant round closed before we could enter it,
and we chose not to put a personal credit card behind a student competition entry. The deployment
is therefore **designed and written as infrastructure-as-code, and never applied**: see
[`docs/deploy.md`](docs/deploy.md). Every measurement in this README and in the technical report
was produced locally, on the machine described in the report. There are no cloud benchmarks, no
Graviton numbers and no live endpoint, because we did not run any.

## Limitations

- The operating point is not deployable at ~86 false alarms per camera-day. See the report.
- Only **78 of 445** FIgLib sequences (~18%) carry usable classical-CV signal under our screening;
  the rest are too diffuse, lack desaturation, or are night/dawn exposures.
- Latency is measured against a human annotator's "plume visible" mark, which is itself a judgement.
- Daytime only. Night ignition detection is a different problem and is out of scope.

## Licence

**All rights reserved.** The competition permits a judge-accessible repository that is not open
source, and this one is published on that basis. No permission is granted to use, copy, modify or
distribute this code. FIgLib imagery retains its own terms; see [`docs/DATA.md`](docs/DATA.md).

Built with Claude Code as a coding assistant.
