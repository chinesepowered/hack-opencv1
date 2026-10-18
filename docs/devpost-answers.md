# Devpost submission answers

Prepared answers for **OpenCV AI Competition 2026, powered by AWS** (`opencv26`), deadline
**2026-10-27 06:45 UTC**. Copy into the Devpost form; nothing here is submitted automatically.

---

## Form fields

### Special Award Consideration (dropdown, optional, multi-select)

**Leave empty. Select neither award.**

**Best Use of COOL Award: not eligible, do not select.** We never used the Cloud-Optimized OpenCV
Library. There is no AWS account, no Graviton instance, and no COOL workload anywhere in this
project. Eligibility requires that "COOL must execute the claimed core workload on AWS Graviton",
which did not happen, so selecting this would be a false claim.

**Agentic Vision Award: eligible on paper, deliberately not selected.** The verification loop does
meet the stated bar, in that OpenCV output changes the next tool call and the final action, and the
repository has the workflow diagram, the traces and the human-control story the award asks for. We
are not selecting it because **the loop did not work**: measured on held-out cameras it removed 7.0%
of false alarms at a cost of 9.3% of recall, missed the bar we set beforehand, and is disabled in the
reported system. Submitting a loop we measured as a net negative for an award about agentic
effectiveness is not a claim we are willing to make.

> If the submitter disagrees, this is their call to make, not ours. The evidence either way is in
> `docs/report.md` section 7, and the number to weigh is the 9.3% recall cost.

### Repository URL (required)

```
https://github.com/chinesepowered/hack-opencv1
```

Public, judge-accessible, **all rights reserved** (the rules permit a repository that is not open
source). `uv.lock` pins every dependency exactly; OpenCV is pinned in `pyproject.toml` as
`opencv-python-headless==5.0.0.93`.

### Working web endpoint (optional)

**Leave blank.** There is no deployed endpoint, and we are not going to put a URL in this field that
does not exist. The requirement offers "a working web endpoint **or** an arranged live screen-share
demonstration", and we are taking the second option. The testing instructions below say so and offer
the screen-share.

### Testing instructions (required, textarea)

Paste verbatim:

---

**There is no hosted endpoint. We offer a live screen-share instead, and the interface runs locally
in about two minutes.**

**1. Run the interface (no imagery download, no API key, no AWS account):**

```
git clone https://github.com/chinesepowered/hack-opencv1
cd hack-opencv1
uv sync
uv run python -m http.server 3041 --directory web
```

Open http://localhost:3041 at a 1280x800 window. Pick any of the three held-out cameras along the
top, press Play, or use "Jump to ignition". You will see the annotated frame with its residual mask,
the fitted probability against the operating point, the operator card a dispatcher would receive,
and the full verification trace.

This replays a real run. Every probability and every trace line on screen is read from the committed
evaluation artefacts, so what you see is bit-identical to the numbers in the report. Check number 4
below proves that automatically.

**Please look at the second camera (`so-s 2017-07-22`) and the footer.** It is where the system
suppresses eleven frames of real smoke, and says so on screen. That is the honest failure case.

**2. Verify every reported number reproduces on your machine (about 20 seconds, no network):**

```
uv run python scripts/selftest.py
```

25 checks. It recomputes the held-out AUC, average precision, recall, operating point and
false-alarm rate from the committed per-frame feature tables plus the committed model, re-reads both
agent-layer experiments, and confirms the interface is displaying exactly the probabilities the
evaluation produced. Exit code 0 means every number in the README and the report was reproduced.

**3. Re-run the evaluation and the agent experiment yourself (seconds, no imagery needed):**

```
uv run python scripts/fit_classifier.py
uv run python scripts/evaluate.py
uv run python scripts/suppression_experiment.py --baseline sliding
uv run python scripts/suppression_experiment.py --baseline anchored
```

**4. Rebuild from raw imagery (optional, about 40 minutes per split).** `scripts/screen_catalog.py`
then `scripts/extract_features.py` download FIgLib frames into a git-ignored cache. Everything else
runs from the committed tables, so this is only needed to verify the feature extraction itself.

**5. AWS.** `infra/` contains the SAM stack, Dockerfile and Lambda handler. It has **never been
deployed**: the $150 compute-grant round closed before we could enter it and we did not attach a
personal card. `docs/deploy.md` explains the design and states plainly that there are no cloud
benchmarks, no Graviton numbers and no live endpoint, because we never ran any. Please do not expect
a stack to come up.

**Live screen-share:** happy to arrange one at any time convenient to the judges, running the
pipeline end to end on cached sequences and on a fresh download.

---

## Project description (Devpost "About the project")

### Inspiration

A wildfire is cheapest to fight in its first twenty minutes, and fire-watch cameras are already
pointed at the right ridgelines around the clock. The scarce resource is not the camera. It is a
human looking at the right one of hundreds of feeds at the right moment.

### What it does

Smoke Sentinel scores every frame from a fixed camera for early smoke and escalates to a human
dispatcher with the evidence. It never dispatches anything itself.

The key decision is that early smoke is **not an object you can draw a box around**. It is a faint,
low-contrast, slowly spreading change in texture that looks like fog, dust, haze or a cloud shadow.
So this is temporal change detection, not object detection: build a reference from the camera's own
earlier frames, and measure what is different now.

### How we built it

Classical OpenCV 5 throughout: an anchored median reference, `findTransformECC` drift alignment, a
per-channel photometric fit, signed luminance residual, `Sobel` and `boxFilter` texture energy, HSV
desaturation, `morphologyEx` masking and `connectedComponentsWithStats` geometry. Twelve
measurements feed a depth-3 decision tree fitted on 35 design cameras. No learned representation
appears anywhere, and the model never scores pixels.

**Results on 43 cameras never seen during design:** AUC **0.846**, average precision **0.906**,
against **0.440** for the naive luminance-residual baseline. Per-sequence median AUC 0.937. At the
operating point where all 43 sequences are caught, median detection latency is 420 s from the
annotator's "plume visible" mark.

### Challenges we ran into

Three bugs, and they turned out to be the same bug three times.

The reference **contaminated itself**: a sliding baseline window crept forward with the query frame
until it was 83% post-ignition, so the detector was subtracting the plume from itself, worst exactly
when the fire was largest. Anchoring it took AUC from 0.50 to 0.85.

Per-scene offsets **swamped the signal**: features scored 0.67 to 0.73 AUC within a sequence but
0.53 pooled, because an absolute band-coverage number means different things on a hazy valley camera
and a clear ridge. Per-camera median/MAD calibration fixed it.

Then the agent's own history pool turned out to be contaminated the same way, which is the next
section.

### Accomplishments we are proud of

Honestly, the negative result. We built an agentic verification layer specifically to fix the
false-alarm rate, measured it properly, found it cost 9.3% of recall for 7.0% fewer false alarms,
and **reported that instead of quietly dropping it**. The diagnosis generalises: every verification
step was reading the same measurement the detector had already read, so a frame the detector got
wrong was one the loop got wrong too. A second look needs a second *kind* of evidence.

We are also proud that the headline is a limitation. At the operating point where every fire is
caught, this system raises about 86 false alarms per camera-day, and a real fire-watch desk needs
well under one. That gap is stated in the first paragraph of the README, not buried.

### What we learned

Any time a system compares "now" against "recently", on data where the event of interest persists,
the reference is a candidate for contamination. We found that bug class three times in one project,
and the third time only because we instrumented the thing we assumed was fine.

### What's next

Rebuild the camera-neighbour map over the design split so the verification policy can be tuned
without touching the reported numbers, then give the loop a genuinely independent second signal: a
vision-language model reading the candidate crop, a wind-direction prior, or a satellite hotspot
feed. Per-camera threshold calibration from each camera's own quiet hours after that.

### Built with

`opencv` `python` `scikit-learn` `numpy` `strands-agents` `aws-sam` `aws-lambda` `uv` `playwright`

---

## Deliverables checklist

| Requirement | Status |
| --- | --- |
| Technical report (problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use) | `docs/report.md` |
| Judge-accessible repository | GitHub, public, all rights reserved |
| Pinned dependencies | `uv.lock`, OpenCV pinned exactly |
| Build / deploy / test instructions | README, `docs/deploy.md`, `scripts/selftest.py` |
| Architecture diagram (OpenCV 5 + AWS + agent components) | `docs/architecture.png` |
| Agent workflow diagram | `docs/agent-workflow.png` |
| Working endpoint **or** arranged screen-share | Screen-share offered; no endpoint exists |
| Video, 5 minutes maximum, shows team / app / architecture / results | ~3 min 40 s |
| Evaluation evidence including failure cases | `evaluation/` committed; failures in report sections 5, 7, 11 |
| Zip archive | Built by `scripts/make_archive.sh`, excludes `.venv`, `data/`, caches, secrets |

## Things to never claim for this entry

* **No COOL.** It was not used at all.
* **No AWS deployment**, no endpoint URL, no Graviton or cost numbers. The stack was never applied.
* **No agentic win.** The loop is a measured negative result and is disabled.
* Do not describe the demo interface as a live pipeline. It is a replay of a real run, and it says
  so on screen.
