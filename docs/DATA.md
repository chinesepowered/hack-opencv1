# Data: HPWREN FIgLib

## Source

**HPWREN Fire Ignition images Library (FIgLib)** — <https://www.hpwren.ucsd.edu/FIgLib/>
Data index: <https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html>

Fixed field-of-view cameras at HPWREN sites in southern California. Each sequence covers one
ignition: 81 frames spaced 60 seconds apart, from 40 minutes before the plume became visible to
40 minutes after.

## Terms of use, quoted

> "While we are making the data publicly available, its use requires a credit reference to
> https://www.hpwren.ucsd.edu/ in derivative work."

> "This data is provided as-is, no guarantees for anything is included."

**How we comply.** The credit reference appears in this file, in the project `README.md`, in the
technical report, in the web UI footer, and on screen in the demo video. Frames are cached locally
under `data/cache/` which is git-ignored, so **this repository redistributes no FIgLib imagery**.
Figures in the report are derived visualisations (detector overlays, plots), each carrying the
credit.

## Why this dataset, and what we rejected

| Candidate | Licence | Decision |
| --- | --- | --- |
| **HPWREN FIgLib** | Public, requires credit reference | **Chosen.** Permissive, explicit, and the labels are free (see below) |
| AI For Mankind annotated smoke sets | CC BY-NC-SA 4.0 | **Rejected.** The NonCommercial clause is a poor fit for a competition entry with commercial intent, and the ShareAlike term would encumber derived work. We do not need their bounding boxes |
| Roboflow mirror of the above | CC BY-NC-SA 4.0 | Rejected for the same reason |

## Labels come from the filenames

FIgLib names each frame:

```
<origin_unix_timestamp>_<offset_seconds_from_visible_plume_appearance>.jpg
```

Non-negative offsets are URL-encoded with `%2B` for the leading `+`:

```
1500758913_-02400.jpg     40 minutes before the plume was visible   -> label: no smoke
1500761313_%2B00000.jpg   the moment of visible plume appearance     -> label: smoke
1500763713_%2B02400.jpg   40 minutes after                           -> label: smoke
```

This gives two things at once:

1. **Binary ground truth** for precision/recall, with no annotation work and no annotation licence.
2. **Real detection latency in seconds.** Because offsets are true seconds from the human-visible
   plume, "first alert at +180 s" is a measured quantity, not an estimate. A detection at a negative
   offset means the system saw the plume *before* a human annotator marked it visible.

> Parsing note: a regex of the form `_(-?\d+)\.jpg$` silently matches only the negative half of every
> sequence, because the positive offsets carry `%2B`. That would quietly delete every positive label.
> `src/sentinel/data.py` handles the encoded form explicitly.

## Sequence selection, and why it is not cherry-picking

FIgLib holds 455 sequences and they are **not** equally usable. Many are "fires" only in the sense
that a plume existed somewhere; the camera may show heavy valley haze before ignition and only a
slightly hazier scene after, with no discrete plume at any offset. A detector tuned on such a
sequence is being tuned against a label with almost no corresponding image signal.

We therefore screen sequences before use, with a cheap, documented, reproducible metric:

1. Take the earliest (most negative) and latest (most positive) frame.
2. Remove the global brightness difference, so a general exposure shift does not count as change.
3. Measure absolute difference per cell on a 12x16 grid.
4. **Concentration** = hottest cell / median cell. High means change is *localised* (a plume);
   low means change is spread across the whole frame (haze, exposure, weather).

Measured across a random sample of 14 sequences, concentration ranged from **2.1** (diffuse, no
usable plume) to **9.7** (sharply localised). `20170722_FIRE_bm-n-mobo-c`, chosen arbitrarily at the
start of this project, scores near the bottom and shows no visible plume in any frame.

This screen selects for *image quality*, not for detector success: it is computed from raw frames
only, never from detector output, and it is applied before any threshold is tuned. The sequence
list, each sequence's concentration score, and the train/held-out split are committed in
`evaluation/manifest.json` so the selection is auditable. Low-concentration sequences are **not
discarded** — they are reported separately as the hard subset, because "our detector finds nothing
in diffuse regional haze" is an honest and useful limitation to publish.

## Split integrity: which sequences were seen during development

A held-out set is only held out if nobody looked at it. During development the following sequences
were viewed as triptychs, or used for detector debugging and threshold inspection, and are therefore
**forced into the design set** regardless of how the random split fell:

| Sequence | How it was used |
| --- | --- |
| `20251102_ScissorsFire_vo-e-mobo-c` | Primary design sequence; three rounds of detector debugging, residual/mask diagnostics |
| `20240729_NixonFire_hp-n-mobo-c` | Viewed in selection triptych; used in feature calibration |
| `20200812_LakeFire_dwpgm-n-mobo-c` | Viewed in selection triptych; used in feature calibration |
| `20250617_MonteFire_cp-w-mobo-c` | Viewed in selection triptych |
| `20170927_FIRE_smer-tcs9-mobo-c` | Viewed in selection triptych |
| `20200521_FIRE_om-n-mobo-c` | Viewed in selection triptych |
| `20180723_FIRE_tp-e-mobo-c`, `20240909_BridgeFire_stgo-e-mobo-c` | Viewed while repairing the screening metric |

`evaluation/manifest.json` records these under `design_contaminated_by_inspection`. **No held-out
metric is ever quoted from them.** The split after this correction is 35 design / 43 held-out.

This matters because the initial random split placed ScissorsFire — the sequence the detector was
literally tuned against — into the held-out set. Any "held-out" score computed from it would have
been leakage, and would have flattered the result.

## Multi-camera corroboration is only available for a third of sequences

FIgLib sequence names encode `YYYYMMDD_Event_camera`, so two sequences sharing a date are the same
fire seen from different cameras — a genuine cross-confirmation opportunity rather than a simulated
one. Measured on the held-out split:

- 43 sequences across **36 distinct dates**
- only **7 dates** carry two or more cameras
- so **14 of 43 sequences (33%)** have at least one same-date neighbour

This is a real limit on the agent's verification loop and is reported as such. Neighbour
corroboration cannot be the primary suppression mechanism when it is unavailable for two thirds of
candidates; the camera-history baseline and temporal-persistence checks necessarily carry most of
the load. Any claim that "multi-camera confirmation suppresses false alarms" must be read against
this 33% coverage, and the suppression experiment reports per-mechanism counts so the contribution
of each is visible separately.

The index is committed as `evaluation/neighbours.json`.

## Measurement bugs found by looking at the numbers

Two defects in this project were caught only because a result looked wrong, not because a test
failed. Both are recorded here because the pattern is the point: a plausible-looking metric is not
evidence that the pipeline is correct.

1. **Reference contamination.** The background reference used a sliding window that crept forward
   with the query frame, so at +2400 s it was 83% post-ignition — the plume was subtracted from
   itself. Symptom: detection got *worse* as the fire grew, and signed luminance inverted
   (Cohen's d = −1.05 the wrong way). Fixed by anchoring the reference to the oldest history.
2. **Inverted guardrail handling.** The suppression experiment treated every guardrail violation by
   forcing the alert to `suppress`. One guardrail is "cannot suppress a very high confidence
   candidate", so the handler enforced the opposite of that rule and discarded 145 candidates at
   confidence ≈ 0.98. Symptom: an implausible count of exactly 400 blocks, equal to the trace cap,
   and a recall collapse from 0.735 to 0.275. Fixed by inverting the rejected action rather than
   defaulting to suppression.

## Caveats that belong in the evaluation

- **Label noise at the boundary.** Offset zero is when a *human annotator* judged the plume visible;
  it is not the physical ignition time. Detections slightly before zero are plausible, not impossible.
- **Geographic and seasonal bias.** Southern California terrain, mostly daytime, mostly summer.
  Results do not transfer unexamined to boreal forest, tropical, or night-time conditions.
- **Camera mix.** Colour `mobo-c` and monochrome/near-IR `iqeye` imagers behave differently; the
  evaluation reports per-camera-type results rather than pooling them silently.
- **Not every sequence is a wildfire.** Some directories are prescribed burns (`BBM`) or named
  incidents. The evaluation manifest records exactly which sequences were used.
- **Sequence length varies; spacing does not.** Measured across sampled sequences: frame counts run
  64–81 and the post-ignition tail can stop early (e.g. `20170711_FIRE_bm-s-mobo-c` ends at +1380 s,
  with 40 pre and 24 post frames), while the step between consecutive frames is uniformly 60 s.
  Evaluation code must therefore iterate the frame list and read each offset, never assume a fixed
  offset grid such as -2400…+2400; indexing by a hardcoded offset raises `IndexError` on short
  sequences.
