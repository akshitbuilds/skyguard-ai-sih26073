"""
Generates a labeled synthetic dataset of
(spatial_deviation_score, ml_anomaly_score, physically_coherent) -> label
for the score-based genuine_event / sensor_fault gate classifier.

This stands in for real historical Agent 2 / Agent 3 output + analyst-labeled
ground truth until that data exists. Swap-out path: once Agent 2 and Agent 3
push real fields and there's a labeled history of confirmed events/faults,
replace `build_score_dataset()` with a loader over that data -- keep using
`agent4.score_features.extract_score_features` so the feature vector stays
identical between training and inference.

Design of the synthetic distributions (deliberately mirrors the two
reference cases in the brief):
  - genuine_event: usually physically coherent, and spatial_deviation_score
    tends to be high (the event is corroborated across nearby sensors).
    ml_anomaly_score is high because it was flagged at all.
  - sensor_fault: ml_anomaly_score is high (that's *why* it got flagged),
    but physically_coherent is usually False and spatial_deviation_score
    tends to be low (an isolated, uncorroborated reading).

Both classes overlap in the middle deliberately -- real data is not
cleanly separable, and the safety gate (not the classifier) is what has
to be trusted for the ambiguous, high-stakes cases.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent4.score_features import extract_score_features, SCORE_FEATURE_NAMES

RNG = np.random.default_rng(42)


def _sample_genuine_event():
    ml_anomaly_score = float(np.clip(RNG.beta(6, 2), 0, 1))          # skewed high
    spatial_deviation_score = float(np.clip(RNG.beta(5, 2), 0, 1))   # skewed high (corroborated)
    physically_coherent = RNG.random() < 0.88                        # usually coherent
    return spatial_deviation_score, ml_anomaly_score, physically_coherent


def _sample_sensor_fault():
    ml_anomaly_score = float(np.clip(RNG.beta(6, 2), 0, 1))          # also skewed high -- it got flagged
    spatial_deviation_score = float(np.clip(RNG.beta(2, 5), 0, 1))   # skewed low (isolated, uncorroborated)
    physically_coherent = RNG.random() < 0.12                        # usually incoherent
    return spatial_deviation_score, ml_anomaly_score, physically_coherent


GENERATORS = {
    "genuine_event": _sample_genuine_event,
    "sensor_fault": _sample_sensor_fault,
}


def build_score_dataset(n_per_class: int = 1000) -> pd.DataFrame:
    rows = []
    for label, gen in GENERATORS.items():
        for _ in range(n_per_class):
            sds, mas, coherent = gen()
            feats = extract_score_features(sds, mas, coherent)
            feats["label"] = label
            rows.append(feats)
    df = pd.DataFrame(rows)[SCORE_FEATURE_NAMES + ["label"]]
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = build_score_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_score_training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["label"].value_counts())
