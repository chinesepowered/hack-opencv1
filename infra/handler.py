"""Lambda entry point for the scoring pipeline.

NEVER RUN IN AWS. This handler has not been deployed or invoked in Lambda. It
is written out because a deployment design that stops at the YAML is not a
design, and because the shape of the handler is what shows whether the
pipeline actually fits the runtime. It reuses ``sentinel`` unchanged, so the
code that would score a frame in Lambda is the same code that produced every
number in docs/report.md.

Two things about fixed-camera monitoring drive the shape below.

**The reference is state.** Scoring one frame needs tens of minutes of that
camera's earlier frames. A stateless function cannot rebuild that per request
without re-reading the whole window from S3 every time, so the reference is
built once per camera and cached in the artefact bucket, with the Lambda
execution context's warm memory as a second-level cache. The anchored strategy
of ``background.BackgroundConfig`` means the reference does not need rebuilding
on every frame, which is what makes this affordable at all.

**A missing reference is an answer.** A camera that has just come online has no
clean history. The honest response is ``has_reference: false`` and no score,
not a zero that a downstream consumer would read as "no smoke".
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3
import cv2
import joblib
import numpy as np

from sentinel import background, detector, features

ARTEFACTS = os.environ.get("ARTEFACT_BUCKET", "")
FRAMES = os.environ.get("FRAME_BUCKET", "")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/model.joblib")
THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.7668"))
VERIFICATION_ENABLED = os.environ.get("VERIFICATION_ENABLED", "false").lower() == "true"

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")

# Warm-start caches. A container serves many frames from the same few cameras,
# so this is the difference between one S3 GET per request and twenty.
_model = None
_references: dict[str, tuple[float, np.ndarray]] = {}
REFERENCE_TTL_S = 3600.0


def _load_model():
    global _model
    if _model is None:
        body = s3.get_object(Bucket=ARTEFACTS, Key=MODEL_KEY)["Body"].read()
        tmp = "/tmp/model.joblib"
        with open(tmp, "wb") as f:
            f.write(body)
        _model = joblib.load(tmp)
    return _model


def _decode(blob: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def _history(camera: str, now_s: int, span_s: int = 1800) -> list[tuple[int, np.ndarray]]:
    """Oldest frames in this camera's retained window, for the anchored reference."""
    prefix = f"{camera}/"
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=FRAMES, Prefix=prefix)
    keys: list[tuple[int, str]] = []
    for page in pages:
        for obj in page.get("Contents", []):
            stem = obj["Key"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            try:
                keys.append((int(stem), obj["Key"]))
            except ValueError:
                continue
    if not keys:
        return []
    keys.sort()
    t0 = keys[0][0]
    out = []
    for t, key in keys:
        if t - t0 > span_s or now_s - t < background.BackgroundConfig().min_age_s:
            continue
        img = _decode(s3.get_object(Bucket=FRAMES, Key=key)["Body"].read())
        if img is not None:
            out.append((t, img))
    return out


def _reference(camera: str, now_s: int) -> np.ndarray | None:
    cached = _references.get(camera)
    if cached and time.time() - cached[0] < REFERENCE_TTL_S:
        return cached[1]
    ref = background.build_reference(_history(camera, now_s), now_s, background.BackgroundConfig())
    if ref is not None:
        _references[camera] = (time.time(), ref)
    return ref


def _normalise(vec: list[float], stats: dict | None) -> np.ndarray:
    """Per-camera median/MAD calibration (report section 4.4).

    Without a camera's own baseline statistics the features are not comparable
    to the ones the model was fitted on, so this returns the raw vector and the
    caller marks the result as uncalibrated rather than pretending otherwise.
    """
    x = np.asarray(vec, float)
    if not stats:
        return x
    med = np.asarray(stats["median"], float)
    scale = np.asarray(stats["mad"], float)
    scale[scale < 1e-6] = 1.0
    return (x - med) / scale


def lambda_handler(event: dict, context: Any) -> dict:
    """Score one frame, from an S3 event or a direct API POST."""
    if "Records" in event:
        rec = event["Records"][0]["s3"]
        bucket, key = rec["bucket"]["name"], rec["object"]["key"]
        camera = key.split("/")[0]
        blob = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        now_s = int(stem) if stem.lstrip("-").isdigit() else int(time.time())
    else:
        body = json.loads(event.get("body") or "{}")
        camera = body.get("camera", "unknown")
        now_s = int(body.get("timestamp", time.time()))
        import base64
        blob = base64.b64decode(body["image_base64"])

    frame = _decode(blob)
    if frame is None:
        return {"statusCode": 400, "body": json.dumps({"error": "undecodable image"})}

    ref = _reference(camera, now_s)
    result = detector.detect(frame, ref)
    if ref is None:
        # Not a failure: a camera without clean history genuinely cannot be
        # scored, and saying so is more useful than returning 0.0.
        return {"statusCode": 200, "body": json.dumps({
            "camera": camera, "has_reference": False, "scored": False,
            "reason": "insufficient clean history to build a reference"})}

    feat = features.extract(frame, ref, history=None, now_s=now_s)
    payload: dict[str, Any] = {
        "camera": camera,
        "timestamp": now_s,
        "has_reference": True,
        "classical_score": round(result.score, 4),
        "horizon_y": result.horizon_y,
        "candidates": [c.as_dict() for c in result.candidates[:3]],
    }

    if feat is None:
        payload.update({"scored": False, "reason": "no region deviates enough to measure"})
    else:
        stats = _camera_stats(camera)
        x = _normalise(features.vector(feat), stats)
        prob = float(_load_model()["model"].predict_proba(x.reshape(1, -1))[0, 1])
        payload.update({
            "scored": True,
            "calibrated": bool(stats),
            "probability": round(prob, 4),
            "threshold": THRESHOLD,
            "alert": bool(prob >= THRESHOLD),
            "bbox": feat.get("bbox"),
            # The verification loop is measured and disabled by default: it cost
            # 9.3% of recall for 7.0% fewer false alarms (report section 7.4).
            "verification_enabled": VERIFICATION_ENABLED,
            "action": "escalate_to_human" if prob >= THRESHOLD else "none",
        })
        cloudwatch.put_metric_data(
            Namespace="SmokeSentinel",
            MetricData=[{"MetricName": "Probability", "Value": prob,
                         "Dimensions": [{"Name": "Camera", "Value": camera}]}])

    return {"statusCode": 200, "headers": {"content-type": "application/json"},
            "body": json.dumps(payload)}


def _camera_stats(camera: str) -> dict | None:
    """Per-camera calibration statistics, written by an offline job."""
    try:
        body = s3.get_object(Bucket=ARTEFACTS, Key=f"calibration/{camera}.json")["Body"].read()
        return json.loads(body)
    except Exception:
        return None
