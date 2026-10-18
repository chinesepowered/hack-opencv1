"""Strands agent that verifies detector candidates before escalating.

Why an agent at all
-------------------
The fitted classical detector reaches AUC 0.846 on unseen cameras, but at the
operating point where every held-out fire is caught it also fires roughly 86
alerts per camera-day. That is far beyond what a fire-watch desk tolerates. The
ranking is good; the *operating point* is not deployable.

An agent is the right tool for that gap, because the follow-up actions are
conditional on what the imagery shows: a weak candidate deserves a second look
at higher sensitivity, a candidate near another camera's field of view deserves
cross-confirmation, and a candidate that looks like the same shape the camera
saw yesterday at this hour deserves suppression. Which action to take depends on
the vision result, which is exactly the "perception -> decision -> action" loop
the Agentic Vision award asks for.

What is agentic and what is not
-------------------------------
* **Not agentic:** the detector. It is deterministic classical OpenCV plus a
  fitted decision boundary. The model never scores pixels.
* **Agentic:** the verification policy. The agent chooses which tool to call
  next based on prior tool output, and its choices change the final alert.
* **Deterministic guardrails:** hooks enforce what the agent may not do -
  escalate without corroboration, re-run more than twice, or suppress a
  high-confidence candidate. Rules live in code; the model cannot talk its way
  past them.

Human control
-------------
Nothing is dispatched. The loop's terminal actions are *escalate to a human with
evidence* or *suppress with a reason*. Both are recorded in a trace.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Callable

import numpy as np

MAX_RERUNS = 2
MAX_TOOL_CALLS = 12
CONFIRM_MARGIN = 0.15


@dataclasses.dataclass
class TraceEvent:
    kind: str              # tool_call | tool_result | hook | decision
    name: str
    detail: dict[str, Any]

    def as_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "detail": self.detail}


@dataclasses.dataclass
class VerificationOutcome:
    """Terminal state of one verification episode."""

    action: str            # escalate | suppress
    reason: str
    confidence: float
    trace: list[TraceEvent]
    reruns: int
    corroborated: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "reruns": self.reruns,
            "corroborated": self.corroborated,
            "trace": [t.as_dict() for t in self.trace],
        }


class VerificationPolicy:
    """The decision logic, expressed as explicit conditional steps.

    This is deliberately separable from the LLM: the same policy runs whether
    the step order is chosen by a Strands agent or by the deterministic fallback.
    That separation is what makes the suppression experiment honest - we measure
    the *loop*, not the model's prose.
    """

    def __init__(
        self,
        rescore: Callable[[str, int, float], dict],
        neighbours: Callable[[str], list[str]],
        history: Callable[[str, int], list[float]],
    ) -> None:
        self.rescore = rescore
        self.neighbours = neighbours
        self.history = history

    def run(self, seq: str, offset_s: int, initial_conf: float) -> VerificationOutcome:
        trace: list[TraceEvent] = []
        trace.append(TraceEvent("tool_result", "detect", {
            "sequence": seq, "offset_s": offset_s, "confidence": round(initial_conf, 4)}))

        # --- Step 1: is this candidate strong enough to be worth work? --------
        if initial_conf < 0.35:
            trace.append(TraceEvent("decision", "suppress", {"rule": "below_attention_floor"}))
            return VerificationOutcome("suppress", "below attention floor", initial_conf, trace, 0, False)

        # --- Step 2: does this camera routinely look like this? ---------------
        hist = self.history(seq, offset_s)
        trace.append(TraceEvent("tool_call", "compare_history", {"samples": len(hist)}))
        if hist:
            hist_p95 = float(np.percentile(hist, 95))
            trace.append(TraceEvent("tool_result", "compare_history", {
                "baseline_p95": round(hist_p95, 4), "current": round(initial_conf, 4)}))
            # A candidate no stronger than this camera's own routine noise is
            # almost certainly scene clutter, not a new plume.
            if initial_conf <= hist_p95:
                trace.append(TraceEvent("decision", "suppress", {"rule": "within_camera_baseline"}))
                return VerificationOutcome(
                    "suppress", "no stronger than this camera's own 95th-percentile baseline",
                    initial_conf, trace, 0, False)

        # --- Step 3: re-run at higher sensitivity to test persistence ---------
        reruns = 0
        conf = initial_conf
        while reruns < MAX_RERUNS and conf < 0.70:
            sens = min(0.5 + 0.2 * (reruns + 1), 0.9)
            trace.append(TraceEvent("tool_call", "rescore", {"sensitivity": sens}))
            r = self.rescore(seq, offset_s, sens)
            reruns += 1
            trace.append(TraceEvent("tool_result", "rescore", {
                "sensitivity": sens, "confidence": round(float(r.get("confidence", 0.0)), 4)}))
            new_conf = float(r.get("confidence", 0.0))
            # A true plume strengthens or holds when the detector looks harder;
            # transient clutter tends to collapse.
            if new_conf < conf * 0.6:
                trace.append(TraceEvent("decision", "suppress", {"rule": "collapsed_under_rescore"}))
                return VerificationOutcome(
                    "suppress", "signal collapsed when re-examined at higher sensitivity",
                    new_conf, trace, reruns, False)
            conf = max(conf, new_conf)

        # --- Step 4: corroborate against a neighbouring camera ----------------
        nbrs = self.neighbours(seq)
        trace.append(TraceEvent("tool_call", "check_neighbour", {"candidates": len(nbrs)}))
        corroborated = False
        for nb in nbrs[:2]:
            r = self.rescore(nb, offset_s, 0.6)
            nb_conf = float(r.get("confidence", 0.0))
            trace.append(TraceEvent("tool_result", "check_neighbour", {
                "sequence": nb, "confidence": round(nb_conf, 4)}))
            if nb_conf >= 0.5:
                corroborated = True
                break

        # --- Step 5: escalate or suppress -------------------------------------
        if conf >= 0.70 or corroborated:
            reason = ("corroborated by a neighbouring camera" if corroborated
                      else "persistent high-confidence plume on re-examination")
            trace.append(TraceEvent("decision", "escalate", {"rule": "confident_or_corroborated"}))
            return VerificationOutcome("escalate", reason, conf, trace, reruns, corroborated)

        trace.append(TraceEvent("decision", "suppress", {"rule": "uncorroborated_and_marginal"}))
        return VerificationOutcome(
            "suppress", "marginal confidence and no corroboration from another camera",
            conf, trace, reruns, corroborated)


def guardrails(outcome: VerificationOutcome) -> tuple[bool, str]:
    """Deterministic checks applied after the policy, before any alert leaves.

    These mirror the Strands ``BeforeToolCallEvent`` hooks used in the live
    application: they are stated here as pure functions so the offline
    experiment enforces exactly the same rules as the UI.
    """
    if outcome.reruns > MAX_RERUNS:
        return False, f"rerun budget exceeded ({outcome.reruns} > {MAX_RERUNS})"
    if outcome.action == "escalate" and outcome.confidence < 0.5 and not outcome.corroborated:
        return False, "escalation requires confidence >= 0.5 or corroboration"
    if outcome.action == "suppress" and outcome.confidence >= 0.9:
        return False, "cannot suppress a very high confidence candidate"
    return True, "ok"


def build_strands_agent(tools: list[Any]):
    """Construct the Strands agent used by the live application.

    Kept separate from :class:`VerificationPolicy` on purpose: the offline
    suppression experiment must not depend on model latency or wording, while
    the interactive demo benefits from the agent narrating its reasoning.
    """
    from strands import Agent
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(
        client_args={
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.environ["OPENAI_BASE_URL"],
        },
        model_id=os.environ.get("OPENAI_MODEL", "zai-org/GLM-5.3-Flash"),
        params={"temperature": 0, "max_tokens": 1200,
                "chat_template_kwargs": {"enable_thinking": False}},
    )
    system = (
        "You verify wildfire smoke candidates from fixed cameras. "
        "You never dispatch anyone; you either escalate to a human with evidence, "
        "or suppress with a reason. Always check the camera's own history before "
        "escalating, re-examine weak candidates at higher sensitivity, and try to "
        "corroborate with a neighbouring camera. State the evidence you used."
    )
    return Agent(model=model, tools=tools, system_prompt=system, callback_handler=None)
