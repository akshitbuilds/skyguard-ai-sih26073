#!/usr/bin/env python3
"""
inference_example.py -- runnable example of the exact call Agent 6 makes.

Run from the repo root:
    python3 examples/inference_example.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import Agent3AnomalyDetector


def main():
    # 1. Load the detector ONCE (loads model + scaler + threshold; no retraining)
    detector = Agent3AnomalyDetector()
    print(f"Loaded Agent 3 detector: SEQ_LEN={detector.seq_len}h, "
          f"threshold={detector.threshold:.5f}")

    # 2. Prepare at least SEQ_LEN consecutive hourly rows for ONE station.
    #    Columns required: station_id, timestamp, temperature_c, pressure_hpa,
    #    humidity_pct. (latitude/longitude accepted but not used as features.)
    example_data = pd.DataFrame({
        "station_id": ["AWS_BHAVNAGAR"] * 15,
        "timestamp": pd.date_range("2021-07-01T00:00:00", periods=15, freq="h").astype(str),
        "temperature_c": [29.1, 29.3, 29.0, 28.8, 28.5, 28.2, 28.0, 27.9,
                           28.1, 28.6, 29.2, 30.1, 31.0, 31.5, 31.8],
        "pressure_hpa": [1006.2, 1006.1, 1006.0, 1005.9, 1005.8, 1005.7, 1005.9,
                          1006.0, 1006.2, 1006.4, 1006.5, 1006.3, 1006.1, 1005.9, 1005.8],
        "humidity_pct": [68, 69, 70, 71, 72, 73, 74, 73, 71, 68, 65, 61, 58, 55, 53],
    })

    # 3. Run inference. Returns one row per valid sliding window.
    result = detector.predict(example_data)

    # 4. Every row has the two REQUIRED fields for downstream agents:
    #    reconstruction_error, ml_anomaly_score -- plus is_anomaly (boolean-ish
    #    int, derived from reconstruction_error > detector.threshold) and
    #    traceability fields (station_id, seq_start_ts, seq_end_ts).
    print(f"\n{len(result)} window(s) scored:")
    print(result.to_string(index=False))

    print("\nRequired output fields present:", 
          {"reconstruction_error", "ml_anomaly_score"}.issubset(result.columns))


if __name__ == "__main__":
    main()
