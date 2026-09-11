"""
Feature extraction for the score-based genuine_event / sensor_fault gate.

This is the *front-line* classifier: it consumes Agent 2 & Agent 3's output
scores directly (spatial_deviation_score, ml_anomaly_score,
physically_coherent) rather than raw sensor windows. It decides the
high-level question -- "is this reading a real event, or a fault?" -- before
anything gets to the more detailed raw-signal root-cause taxonomy in
`root_cause_classifier.py` (which subclassifies confirmed faults into
SENSOR_SPIKE / FROZEN_VALUE / COMMS_ERROR / DRIFT).

Same function used for training (data/generate_score_synthetic.py) and
inference (screening_gate.py), so there is no train/serve skew.
"""

from __future__ import annotations
import numpy as np

SCORE_FEATURE_NAMES = [
    "spatial_deviation_score",
    "ml_anomaly_score",
    "physically_coherent",
    "corroborated_event_strength",   # spatial_deviation_score * physically_coherent
    "incoherent_anomaly_strength",   # ml_anomaly_score * (1 - physically_coherent)
]


def extract_score_features(
    spatial_deviation_score: float,
    ml_anomaly_score: float,
    physically_coherent: bool,
) -> dict:
    """
    Builds interaction features on top of the three raw score inputs.

    corroborated_event_strength is high only when a strong spatial signal
    is ALSO physically coherent -- i.e. evidence that points toward a real,
    physically-extended event (Case A in the design brief).

    incoherent_anomaly_strength is high only when a strong ML anomaly score
    is paired with physical incoherence -- i.e. evidence that points toward
    a sensor fault (Case B in the design brief). A high ml_anomaly_score on
    its own is NOT enough to call fault; physical incoherence is what
    distinguishes "the model noticed something unusual" from "the model
    noticed something impossible."
    """
    sds = float(np.clip(spatial_deviation_score, 0.0, 1.0))
    mas = float(np.clip(ml_anomaly_score, 0.0, 1.0))
    coherent = 1.0 if physically_coherent else 0.0

    return {
        "spatial_deviation_score": sds,
        "ml_anomaly_score": mas,
        "physically_coherent": coherent,
        "corroborated_event_strength": sds * coherent,
        "incoherent_anomaly_strength": mas * (1.0 - coherent),
    }


def score_feature_vector(features: dict) -> np.ndarray:
    """Order-locked vector for the model, matching SCORE_FEATURE_NAMES."""
    return np.array([features[name] for name in SCORE_FEATURE_NAMES], dtype=float)
