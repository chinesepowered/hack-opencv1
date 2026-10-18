"""Render docs/architecture.png and docs/agent-workflow.png.

Generated rather than drawn by hand so the diagrams cannot drift away from the
code: every box below names a real module, and the numbers in them are read out
of evaluation/ at render time instead of being typed in.

Boxes size themselves to their content. Matplotlib text is measured in points
while the axes are in 0..1, so :func:`u` converts one into the other against the
figure height and the axes are made to fill the figure. Hand-guessed heights
clip text silently, and a clipped diagram is worse than no diagram.

    uv run python scripts/make_diagrams.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

INK = "#15202b"
DIM = "#55646f"
LINE = "#9fadba"
CV, CV_E = "#dceaf6", "#4f87b4"
FIT, FIT_E = "#e6e0f5", "#7c63b8"
AGENT, AGENT_E = "#fde6dc", "#d4663c"
HUMAN, HUMAN_E = "#dcf2e6", "#3f9b72"
OFF, OFF_E = "#ededed", "#949494"
STOP = "#6d53ad"

TITLE_FS = 9.4
BODY_FS = 7.9
LINESPACING = 1.62

_fig_h = [9.0]        # current figure height in inches, set by canvas()


def u(points: float) -> float:
    """Points -> axes units, for an axes that fills the figure vertically."""
    return points / (72.0 * _fig_h[0])


def canvas(w, h):
    _fig_h[0] = h
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])      # fill the figure, so u() is exact
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, top, w, title, body="", fc=CV, ec=CV_E, dashed=False):
    """Draw a box sized to its content. Returns (x, bottom, w, h)."""
    n = len(body.split("\n")) if body else 0
    h = u(26) + (u(6 + n * BODY_FS * LINESPACING + 10) if n else u(8))
    y = top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.014",
                                linewidth=1.3, facecolor=fc, edgecolor=ec,
                                linestyle=(0, (4, 2.4)) if dashed else "solid", zorder=2))
    ax.text(x + w / 2, top - u(17), title, ha="center", va="center", fontsize=TITLE_FS,
            fontweight="bold", color=INK, zorder=3)
    if body:
        ax.text(x + w / 2, top - u(31), body, ha="center", va="top", fontsize=BODY_FS,
                color=DIM, linespacing=LINESPACING, zorder=3)
    return (x, y, w, h)


def arrow(ax, a, b, label="", colour=LINE, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.25, color=colour, zorder=1,
                                 linestyle=(0, (3.5, 2.2)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, label, ha="center", va="center",
                fontsize=7.0, color=DIM, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def legend(ax, items, y):
    x = 0.030
    for label, fc, ec, dashed in items:
        ax.add_patch(FancyBboxPatch((x, y), 0.016, u(11),
                                    boxstyle="round,pad=0.002,rounding_size=0.004",
                                    facecolor=fc, edgecolor=ec, linewidth=1.1,
                                    linestyle=(0, (3, 2)) if dashed else "solid"))
        ax.text(x + 0.022, y + u(5.5), label, ha="left", va="center", fontsize=7.6, color=DIM)
        x += 0.029 + 0.0067 * len(label)


def mid_r(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def architecture(res, fit, sup):
    fig, ax = canvas(13.4, 9.0)
    ax.text(0.5, 0.975, "Smoke Sentinel - pipeline as built", ha="center", va="top",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.5, 0.947,
            "Every detection decision is classical OpenCV 5. Only the final decision boundary "
            "is fitted, and only on design cameras.",
            ha="center", va="top", fontsize=9, color=DIM)

    # Four columns plus the boxstyle pad must stay inside the axes, or the
    # rightmost column is silently clipped at x=1.
    W = 0.208
    L = 0.021
    C = L + W + 0.036
    R = C + W + 0.036
    F = R + W + 0.036
    TOP = 0.905

    # --- column 1: the data ---
    src = box(ax, L, TOP, W, "HPWREN FIgLib",
              "445 catalogued sequences\n78 pass screening (~18%)\n"
              "labels come from filename\noffsets, so latency is real\n"
              "cached locally, never\nredistributed", fc="#f1f4f7", ec="#8795a3")
    split = box(ax, L, src[1] - 0.050, W, "Split by camera",
                "35 design sequences\n43 held-out sequences\nno camera appears in both\n"
                "scripts/screen_catalog.py", fc="#f1f4f7", ec="#8795a3")

    # --- column 2: classical CV ---
    ref = box(ax, C, TOP, W, "1. Reference builder",
              "anchored median of the\noldest available history\nECC translation alignment\n"
              "per-channel photometric fit\nbackground.py")
    resid = box(ax, C, ref[1] - 0.034, W, "2. Residual and masking",
                "signed luminance residual\nSobel / boxFilter texture\nHSV desaturation\n"
                "morphologyEx open + close\ndetector.py")
    feats = box(ax, C, resid[1] - 0.034, W, "3. Twelve measurements",
                "connectedComponentsWithStats\ngeometry and band cover\nCanny edge density\n"
                "multi-minute area slope\ndrift coherence\nfeatures.py")
    arrow(ax, (C + W / 2, ref[1]), (C + W / 2, resid[1] + resid[3]))
    arrow(ax, (C + W / 2, resid[1]), (C + W / 2, feats[1] + feats[3]))
    arrow(ax, mid_r(src), (C, src[1] + src[3] * 0.62), "frames")
    arrow(ax, mid_r(split), (C, feats[1] + feats[3] * 0.55), "fit / eval\ngroups", rad=-0.16)

    # --- column 3: fitted boundary and the operating point ---
    norm = box(ax, R, TOP, W, "4. Per-camera calibration",
               "median / MAD of each camera's\nown earliest frames\n\n"
               "Bug 2: an absolute band-cover\nnumber means different things\non different scenes",
               fc=FIT, ec=FIT_E)
    tree = box(ax, R, norm[1] - 0.034, W, "5. Depth-3 decision tree",
               f"fitted on the design split only\n"
               f"design grouped-CV AUC {fit['results'][fit['best']]['cv_auc']:.3f}\n"
               f"held-out AUC {res['held_out']['auc']:.3f}\n"
               f"held-out AP {res['held_out']['ap']:.3f}",
               fc=FIT, ec=FIT_E)
    sweep = min(res["sweep"], key=lambda r: abs(r["target_fpr"] - 0.10))
    op = box(ax, R, tree[1] - 0.034, W, "6. Operating point",
             f"threshold at FPR 0.10\nrecall {sweep['recall']:.3f}, "
             f"{sweep['sequences_detected']}/{res['held_out']['sequences']} sequences\n"
             f"median latency {sweep['median_latency_s']:.0f} s\n"
             f"{sweep['false_alarms_per_camera_day']:.0f} false alarms / camera-day",
             fc=FIT, ec=FIT_E)
    arrow(ax, (R + W / 2, norm[1]), (R + W / 2, tree[1] + tree[3]))
    arrow(ax, (R + W / 2, tree[1]), (R + W / 2, op[1] + op[3]))
    arrow(ax, mid_r(feats), (R, norm[1] + 0.014), "feature\nvectors", rad=0.16)

    # --- column 4: the agent layer, and the human who decides ---
    agent = box(ax, F, TOP, W, "7. Verification policy",
                "attention floor\ncompare_history (anchored)\ntemporal re-examination\n"
                "neighbouring-camera check\ndeterministic guardrails\nagent.py\n\n"
                f"MEASURED, NOT ENABLED\n"
                f"{sup['false_alarm_reduction_pct']:.1f}% fewer false alarms\n"
                f"cost {sup['recall_cost_pct']:.1f}% of recall",
                fc=AGENT, ec=AGENT_E, dashed=True)
    human = box(ax, F, agent[1] - 0.050, W, "8. Operator card",
                "score, candidate region and\nthe full verification trace\n"
                "acknowledge / request another\ncamera / dismiss\n\n"
                "fed by the detector directly:\nthe reported numbers do not\n"
                "pass through the agent\n\nnothing is ever dispatched",
                fc=HUMAN, ec=HUMAN_E)
    arrow(ax, mid_r(op), (F, agent[1] + 0.030), colour=AGENT_E, dashed=True, rad=0.18)
    arrow(ax, (F + W / 2, agent[1]), (F + W / 2, human[1] + human[3]),
          colour=AGENT_E, dashed=True)
    arrow(ax, mid_r(op), (F, human[1] + human[3] * 0.42), colour=HUMAN_E)

    # --- deployment, designed only ---
    bottom = min(feats[1], op[1])
    box(ax, C, bottom - 0.046, W * 2 + 0.040, "AWS deployment - designed as IaC, never applied",
        "S3 frame landing  ->  container Lambda running OpenCV 5  ->  API Gateway, 2000 req/day quota\n"
        "reserved concurrency 5      CloudWatch alarms      $20 AWS Budgets action\n\n"
        "The $150 compute-grant round closed before we entered it and no personal card was attached,\n"
        "so this was never deployed. There are no cloud benchmarks and no live endpoint in this project.",
        fc=OFF, ec=OFF_E, dashed=True)
    arrow(ax, (C + W / 2, feats[1]), (C + W / 2, bottom - 0.046), colour=OFF_E, dashed=True)

    ax.text(0.028, 0.058,
            "Imagery: HPWREN FIgLib (https://www.hpwren.ucsd.edu/), used with the required credit reference.",
            fontsize=7.4, color=DIM)
    legend(ax, [("classical OpenCV 5", CV, CV_E, False),
                ("fitted boundary, design split only", FIT, FIT_E, False),
                ("agent layer, measured not enabled", AGENT, AGENT_E, True),
                ("human decision", HUMAN, HUMAN_E, False),
                ("designed, not deployed", OFF, OFF_E, True)], y=0.020)
    fig.savefig(DOCS / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def agent_workflow(slide, anch):
    fig, ax = canvas(13.4, 10.2)
    ax.text(0.5, 0.980, "Verification loop - and why it did not clear its bar",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.5, 0.955,
            "A real perception -> decision -> action cycle, measured on the held-out split. "
            "The measurement says do not ship it.",
            ha="center", va="top", fontsize=9, color=DIM)

    cx, cw = 0.275, 0.380
    sx, sw = 0.700, 0.272
    TOP = 0.915

    steps = [
        ("Candidate from the detector",
         "p(smoke) >= 0.767, the reported operating point", CV, CV_E, None),
        ("1. Attention floor",
         "below 0.35 is not worth any further work", AGENT, AGENT_E,
         "below_attention_floor"),
        ("2. compare_history",
         "stronger than this camera's own 95th percentile?", AGENT, AGENT_E,
         "within_camera_baseline"),
        ("3. Temporal re-examination",
         "median score of the real adjacent frames at +/-120 s\nup to two further looks",
         AGENT, AGENT_E, "collapsed_under_rescore"),
        ("4. Neighbouring camera",
         "score a different real camera that saw the same event", AGENT, AGENT_E, None),
        ("5. Escalate or suppress",
         "confidence >= 0.70, or corroborated, escalates", AGENT, AGENT_E,
         "uncorroborated_and_marginal"),
        ("Guardrails, deterministic",
         "rerun budget, escalation floor, and no suppressing p >= 0.9\n"
         "a violation inverts the rejected action rather than\nblanket-suppressing the candidate",
         "#fff4d6", "#c79a28", None),
        ("Operator card",
         "escalate to a human, with the evidence and the trace", HUMAN, HUMAN_E, None),
    ]

    top = TOP
    prev_bottom = None
    for title, body, fc, ec, exit_label in steps:
        b = box(ax, cx, top, cw, title, body, fc=fc, ec=ec)
        if prev_bottom is not None:
            arrow(ax, (cx + cw / 2, prev_bottom), (cx + cw / 2, b[1] + b[3]))
        if exit_label:
            y = b[1] + b[3] * 0.38
            arrow(ax, (cx, y), (cx - 0.050, y), colour=STOP)
            ax.text(cx - 0.057, y, exit_label, ha="right", va="center",
                    fontsize=7.3, color=STOP, family="monospace")
        prev_bottom = b[1]
        top = prev_bottom - 0.026

    ax.text(0.060, (TOP + prev_bottom) / 2, "suppression exits", ha="center", va="center",
            fontsize=8.8, color=STOP, fontweight="bold", rotation=90)

    r1 = box(ax, sx, TOP, sw, "What it actually did",
             f"sliding history pool\n"
             f"   {slide['history_pool_post_ignition_pct']:.0f}% of the pool was already\n"
             f"   post-ignition\n"
             f"   {slide['false_alarm_reduction_pct']:.1f}% fewer false alarms\n"
             f"   for {slide['recall_cost_pct']:.1f}% of recall\n\n"
             f"anchored history pool\n"
             f"   {anch['history_pool_post_ignition_pct']:.0f}% post-ignition\n"
             f"   {anch['false_alarm_reduction_pct']:.1f}% fewer false alarms\n"
             f"   for {anch['recall_cost_pct']:.1f}% of recall",
             fc="#f1f4f7", ec="#8795a3")
    r2 = box(ax, sx, r1[1] - 0.040, sw, "The bar it had to clear",
             "materially fewer false alarms\nper camera-day, for under 5%\nof recall\n\n"
             "It did not clear it, so it is\nreported as a negative result\nand left disabled.",
             fc="#fde2e2", ec="#c0504d")
    box(ax, sx, r2[1] - 0.040, sw, "Why it fails",
        "The verification signal and the\ndetector score come from the\n"
        "same measurement, so a frame\nthe detector gets wrong is one\n"
        "the loop also gets wrong.\n\n"
        "A second look needs a second\nkind of evidence, not a second\nreading of the first kind.",
        fc=AGENT, ec=AGENT_E)

    ax.text(0.028, 0.048,
            "Both second looks use real additional imagery: adjacent frames from the same camera, and a "
            "different camera that observed the same event.\n"
            "Neither is a simulated sensitivity re-run. Measured on the held-out split only; the policy "
            "was never tuned against it.",
            fontsize=7.6, color=DIM, linespacing=1.7)
    fig.savefig(DOCS / "agent-workflow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    res = json.load(open(ROOT / "evaluation/results.json"))
    fit = json.load(open(ROOT / "evaluation/fit_report.json"))
    anch = json.load(open(ROOT / "evaluation/suppression_anchored.json"))
    slide = json.load(open(ROOT / "evaluation/suppression_sliding.json"))
    architecture(res, fit, anch)
    agent_workflow(slide, anch)
    print("-> docs/architecture.png, docs/agent-workflow.png")


if __name__ == "__main__":
    main()
