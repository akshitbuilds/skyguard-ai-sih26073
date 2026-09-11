"""
Feature extraction for the root-cause classifier.

IMPORTANT: this exact function is used both to build the training set
(data/generate_synthetic.py) and at inference time (pipeline.py), so the
feature vector the model sees is always constructed the same way.

Input: a short trailing window of raw readings ending at the flagged point
(most recent value last), plus the expected physical range for the sensor
type (e.g. temperature: -10..55 C).

All features are plain floats so SHAP / RandomForest can consume them
directly, and each one maps to a human-readable phrase used in the
narrative generator.
"""

from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "z_score",
    "delta_prev",
    "roc",                      # rate of change over window (units/step)
    "rolling_std",
    "rolling_mean_dev",         # |value - rolling_mean| / rolling_std
    "flatline_run",             # consecutive identical (or near-identical) values ending at current point
    "missing_ratio",            # fraction of expected samples missing in window
    "gap_since_last",           # timesteps since previous valid reading (1 = no gap)
    "out_of_range",             # 0/1, is value outside physically plausible range
    "range_breach_magnitude",   # how far outside plausible range, normalized
    "monotonic_run",            # consecutive steps moving in same direction (signed run length)
    "long_term_bias",           # drift proxy: mean of last window vs mean of window before that
]


def extract_features(
    window: list[float],
    expected_range: tuple[float, float],
    missing_mask: list[bool] | None = None,
    gap_steps: int = 1,
    prior_window: list[float] | None = None,
    flatline_eps: float = 1e-6,
) -> dict:
    """
    window: trailing values ending at the flagged reading, window[-1] is the flagged value.
            Should have at least 4-6 points for meaningful stats; degrades gracefully if shorter.
    expected_range: (min_plausible, max_plausible) for this sensor/variable type.
    missing_mask: same length as window, True where a sample was interpolated/missing.
    gap_steps: number of expected sampling intervals since the previous valid reading.
    prior_window: the window immediately preceding `window`, used for drift/bias detection.
    """
    w = np.asarray(window, dtype=float)
    if w.size == 0:
        raise ValueError("window must contain at least one value")

    value = w[-1]
    lo, hi = expected_range

    mean = float(np.mean(w))
    std = float(np.std(w)) if w.size > 1 else 0.0
    std_safe = std if std > 1e-6 else 1e-6

    z_score = (value - mean) / std_safe

    delta_prev = float(w[-1] - w[-2]) if w.size > 1 else 0.0
    roc = delta_prev / max(gap_steps, 1)

    rolling_std = std
    rolling_mean_dev = abs(value - mean) / std_safe

    # flatline run: how many trailing values (including current) are ~identical
    flatline_run = 1
    for i in range(w.size - 1, 0, -1):
        if abs(w[i] - w[i - 1]) <= flatline_eps:
            flatline_run += 1
        else:
            break

    missing_ratio = float(np.mean(missing_mask)) if missing_mask else 0.0

    out_of_range = 1.0 if (value < lo or value > hi) else 0.0
    span = max(hi - lo, 1e-6)
    if value < lo:
        range_breach_magnitude = (lo - value) / span
    elif value > hi:
        range_breach_magnitude = (value - hi) / span
    else:
        range_breach_magnitude = 0.0

    # monotonic run: consecutive steps moving in the same direction ending at current point
    diffs = np.diff(w)
    monotonic_run = 0
    if diffs.size:
        sign = np.sign(diffs[-1])
        for d in diffs[::-1]:
            if np.sign(d) == sign and sign != 0:
                monotonic_run += 1
            else:
                break
    monotonic_run = float(monotonic_run * (1 if diffs.size and diffs[-1] >= 0 else -1))

    if prior_window:
        pw = np.asarray(prior_window, dtype=float)
        long_term_bias = float(mean - np.mean(pw)) if pw.size else 0.0
    else:
        long_term_bias = 0.0

    return {
        "z_score": z_score,
        "delta_prev": delta_prev,
        "roc": roc,
        "rolling_std": rolling_std,
        "rolling_mean_dev": rolling_mean_dev,
        "flatline_run": float(flatline_run),
        "missing_ratio": missing_ratio,
        "gap_since_last": float(gap_steps),
        "out_of_range": out_of_range,
        "range_breach_magnitude": range_breach_magnitude,
        "monotonic_run": monotonic_run,
        "long_term_bias": long_term_bias,
    }


def feature_vector(features: dict) -> np.ndarray:
    """Order-locked vector for the model, matching FEATURE_NAMES."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
