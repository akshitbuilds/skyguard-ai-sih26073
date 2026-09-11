"""
Generates a labeled synthetic dataset of (feature_vector -> root_cause_label)
so the classifier has something principled to train on before real historical
IMD-labeled incidents are available.

Each sample simulates a short window of a weather variable (default: temperature,
degC) around a flagged point, injects one specific failure mode, and extracts
features with the SAME function used at inference time (agent4.features).

Swap-out path: once real labeled incident data exists, replace `build_dataset()`
with a loader that reads historical flagged windows + analyst-assigned labels,
keeping the same feature extraction so the model stays compatible.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent4.features import extract_features, FEATURE_NAMES
from agent4.schemas import RootCauseLabel

RNG = np.random.default_rng(42)
EXPECTED_RANGE = (-5.0, 48.0)   # plausible ambient temp range, deg C, for synthetic sensor
WINDOW = 8


def _baseline_series(n, base=25.0, amp=3.0, noise=0.3):
    t = np.arange(n)
    seasonal = amp * np.sin(2 * np.pi * t / 24.0)
    return base + seasonal + RNG.normal(0, noise, n)


def _sample_normal_window():
    series = _baseline_series(WINDOW + WINDOW)
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, [False] * WINDOW, 1

def _sample_spike():
    series = _baseline_series(WINDOW + WINDOW)
    spike_mag = RNG.choice([-1, 1]) * RNG.uniform(8, 20)
    series[-1] += spike_mag
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, [False] * WINDOW, 1

def _sample_frozen():
    series = _baseline_series(WINDOW + WINDOW)
    freeze_len = RNG.integers(4, WINDOW + 1)
    stuck_val = series[-freeze_len - 1]
    series[-freeze_len:] = stuck_val + RNG.normal(0, 1e-4, freeze_len)
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, [False] * WINDOW, 1

def _sample_comms_error():
    series = _baseline_series(WINDOW + WINDOW)
    n_missing = RNG.integers(2, 5)
    missing_mask = [False] * WINDOW
    idxs = RNG.choice(range(WINDOW - n_missing, WINDOW), size=n_missing, replace=False)
    for i in idxs:
        missing_mask[i] = True
        series[-WINDOW + i] = 0.0 if RNG.random() < 0.5 else -999.0
    gap = int(RNG.integers(2, 6))
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, missing_mask, gap

def _sample_drift():
    series = _baseline_series(WINDOW + WINDOW)
    drift_rate = RNG.choice([-1, 1]) * RNG.uniform(0.4, 1.2)
    drift = np.linspace(0, drift_rate * WINDOW, WINDOW)
    series[-WINDOW:] += drift
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, [False] * WINDOW, 1

def _sample_genuine_extreme():
    series = _baseline_series(WINDOW + WINDOW, base=RNG.choice([25.0, 25.0, 40.0]))
    # a real extreme event: gradual physically-consistent ramp to an extreme, then holds
    direction = RNG.choice([-1, 1])
    ramp = np.linspace(0, direction * RNG.uniform(10, 16), WINDOW)
    series[-WINDOW:] += ramp
    series[-WINDOW:] += RNG.normal(0, 0.4, WINDOW)  # still noisy/real, not flatlined
    window = series[-WINDOW:].tolist()
    prior = series[:WINDOW].tolist()
    return window, prior, [False] * WINDOW, 1


GENERATORS = {
    RootCauseLabel.SENSOR_SPIKE: _sample_spike,
    RootCauseLabel.FROZEN_VALUE: _sample_frozen,
    RootCauseLabel.COMMS_ERROR: _sample_comms_error,
    RootCauseLabel.DRIFT: _sample_drift,
    RootCauseLabel.GENUINE_EXTREME: _sample_genuine_extreme,
}


def build_dataset(n_per_class: int = 600) -> pd.DataFrame:
    rows = []
    for label, gen in GENERATORS.items():
        for _ in range(n_per_class):
            window, prior, missing_mask, gap = gen()
            feats = extract_features(
                window=window,
                expected_range=EXPECTED_RANGE,
                missing_mask=missing_mask,
                gap_steps=gap,
                prior_window=prior,
            )
            feats["label"] = label.value
            rows.append(feats)
    df = pd.DataFrame(rows)[FEATURE_NAMES + ["label"]]
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["label"].value_counts())
