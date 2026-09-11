"""
Real Agent-3 adapter for SkyGuard.

Uses the committed LSTM Autoencoder when TensorFlow/model artifacts
are available.

For deterministic screening scenarios near the beginning of the
historical dataset, a short-history warm-up is created by padding
the earliest valid observation backward at hourly intervals. This
allows the real LSTM model to execute instead of silently switching
to the deterministic fallback.

The fallback remains available only when the LSTM/model itself
cannot be loaded.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

AGENT3_ROOT = os.path.join(
    ROOT,
    "agent3_ml_detection",
    "agent3_ml_detection_repo",
)

if AGENT3_ROOT not in sys.path:
    sys.path.insert(0, AGENT3_ROOT)


_detector = None
_load_error = None


def _load_detector():
    global _detector, _load_error

    if _detector is not None or _load_error is not None:
        return _detector

    try:
        from src.inference import Agent3AnomalyDetector

        _detector = Agent3AnomalyDetector()

    except Exception as exc:
        _load_error = exc

    return _detector


def engine_status() -> dict:
    """
    Truthful Agent-3 runtime status.
    """

    det = _load_detector()

    return {
        "available": det is not None,
        "engine": (
            "LSTM_AUTOENCODER"
            if det is not None
            else "DETERMINISTIC_FALLBACK"
        ),
        "error": (
            None
            if det is not None
            else str(_load_error)
        ),
    }


def _fallback(df: pd.DataFrame) -> dict:
    """
    Deterministic proxy used only if the real LSTM cannot be loaded.

    It is explicitly labelled as a fallback and never pretends
    to be the LSTM Autoencoder.
    """

    phys = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
    ]

    recent = df[phys].apply(
        pd.to_numeric,
        errors="coerce",
    )

    recent = recent.ffill().bfill()

    if len(recent) < 4:
        return {
            "ml_anomaly_score": 0.0,
            "reconstruction_error": 0.0,
            "is_anomaly": False,
            "engine": "DETERMINISTIC_FALLBACK",
        }

    hist = recent.iloc[:-1]
    cur = recent.iloc[-1]
    prev = hist.iloc[-1]

    deltas = {
        "temperature_c": abs(
            float(
                cur["temperature_c"]
                - prev["temperature_c"]
            )
        ),
        "pressure_hpa": abs(
            float(
                cur["pressure_hpa"]
                - prev["pressure_hpa"]
            )
        ),
        "humidity_pct": abs(
            float(
                cur["humidity_pct"]
                - prev["humidity_pct"]
            )
        ),
    }

    normalized = max(
        deltas["temperature_c"] / 8.0,
        deltas["pressure_hpa"] / 8.0,
        deltas["humidity_pct"] / 20.0,
    )

    score = float(
        np.clip(
            0.05 + 0.9 * min(normalized, 1.0),
            0.0,
            0.99,
        )
    )

    err = float(
        min(normalized * normalized, 1.0)
    )

    return {
        "ml_anomaly_score": round(score, 4),
        "reconstruction_error": round(err, 6),
        "is_anomaly": bool(score >= 0.65),
        "engine": "DETERMINISTIC_FALLBACK",
    }


def _prepare_warm_history(
    history: List[dict],
    reading: dict,
) -> List[dict]:
    """
    Build a strict, gap-free 11-point hourly sequence ending exactly
    1 hour before `reading`. Real historical points are used where
    available; any missing hour is filled by holding the last known
    physical state forward (same philosophy as preprocessing's
    ffill/bfill), so the window is always exactly hourly-contiguous
    regardless of gaps in the underlying data.
    """

    physical = [
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
    ]

    current_ts = pd.to_datetime(
        reading.get("timestamp"),
        errors="coerce",
    )

    if pd.isna(current_ts):
        return []

    valid_history = []

    for item in history:

        ts = pd.to_datetime(
            item.get("timestamp"),
            errors="coerce",
        )

        if pd.isna(ts) or ts >= current_ts:
            continue

        if not all(
            item.get(field) is not None
            and pd.notna(item.get(field))
            for field in physical
        ):
            continue

        valid_history.append(dict(item))

    if not valid_history:
        return []

    valid_history.sort(
        key=lambda x: pd.to_datetime(x["timestamp"])
    )

    by_ts = {
        pd.to_datetime(item["timestamp"]): item
        for item in valid_history
    }

    earliest = valid_history[0]

    needed = 11

    target_index = pd.date_range(
        end=current_ts - pd.Timedelta(hours=1),
        periods=needed,
        freq="h",
    )

    out_rows = []
    last_known = None

    for ts in target_index:

        if ts in by_ts:
            last_known = by_ts[ts]

        source = last_known if last_known is not None else earliest

        synthetic = dict(source)
        synthetic["timestamp"] = ts.isoformat()

        out_rows.append(synthetic)

    return out_rows

def run_agent3(
    reading: dict,
    history: List[dict],
) -> dict:
    """
    Run Agent 3 using the real LSTM Autoencoder.

    Always attempts the real LSTM when the model is available.
    """

    det = _load_detector()

    # ---------------------------------------------------------
    # Model unavailable -> explicit fallback
    # ---------------------------------------------------------

    if det is None:

        return _fallback(
            pd.DataFrame(
                list(history) + [reading]
            )
        )

    # ---------------------------------------------------------
    # Build a valid warm history
    # ---------------------------------------------------------

    warm_history = _prepare_warm_history(
        history,
        reading,
    )

    rows = (
        warm_history[-11:]
        + [dict(reading)]
    )

    df = pd.DataFrame(rows)

    required = [
        "station_id",
        "timestamp",
        "temperature_c",
        "pressure_hpa",
        "humidity_pct",
    ]

    if any(
        column not in df.columns
        for column in required
    ):
        raise RuntimeError(
            "Agent 3 input schema is incomplete: "
            f"missing one of {required}"
        )

    df = df[required].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df[
        [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct",
        ]
    ] = df[
        [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Final safety check.
    if len(df) < 12:
        raise RuntimeError(
            "Agent 3 could not construct a 12-point "
            "warm-up window."
        )

    if df["timestamp"].isna().any():
        raise RuntimeError(
            "Agent 3 received invalid timestamps."
        )

    # ---------------------------------------------------------
    # REAL LSTM INFERENCE
    # ---------------------------------------------------------

    try:

        out = det.predict(df)

        if out.empty:

            raise RuntimeError(
                "LSTM-AE returned no valid sliding window "
                "after warm-up preparation."
            )

        latest = out.iloc[-1]

        return {
            "ml_anomaly_score": round(
                float(
                    latest[
                        "ml_anomaly_score"
                    ]
                ),
                4,
            ),
            "reconstruction_error": round(
                float(
                    latest[
                        "reconstruction_error"
                    ]
                ),
                6,
            ),
            "is_anomaly": bool(
                latest["is_anomaly"]
            ),
            "engine": "LSTM_AUTOENCODER",
            "window_start": str(
                latest["seq_start_ts"]
            ),
            "window_end": str(
                latest["seq_end_ts"]
            ),
        }

    except Exception as exc:

        # IMPORTANT:
        # If the model loaded successfully but inference failed,
        # do NOT silently pretend that the fallback is the LSTM.

        raise RuntimeError(
            "Agent 3 LSTM-AE inference failed: "
            f"{exc}"
        ) from exc