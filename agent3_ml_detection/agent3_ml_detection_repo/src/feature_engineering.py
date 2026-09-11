"""
feature_engineering.py -- cyclic temporal feature construction.

Design decisions (empirically justified, see reports/evaluation_report.md
"Features" section for the eta^2 investigation that produced these):
  - hour_sin/hour_cos: KEPT (diurnal cycle has real signal for all 3 physical
    variables, strongest for temperature).
  - doy_sin/doy_cos: KEPT (seasonal cycle -- strongest signal of any temporal
    grouping tested, especially for pressure).
  - month: DROPPED -- redundant with day_of_year at coarser granularity.
  - day_of_week: DROPPED -- weather has no weekly cycle (eta^2 <= 0.002).
"""
import numpy as np
import pandas as pd

PHYSICAL_FEATURES = ["temperature_c", "pressure_hpa", "humidity_pct"]
TEMPORAL_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_LIST = [f"{c}_scaled" for c in PHYSICAL_FEATURES] + TEMPORAL_FEATURES


def add_temporal_features(df, timestamp_col="timestamp"):
    """Add hour_sin/cos and doy_sin/cos cyclic encodings. Returns a copy."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    hour = ts.dt.hour
    doy = ts.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # divide by 366 (not 365) to safely handle leap years without discontinuity
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return df
