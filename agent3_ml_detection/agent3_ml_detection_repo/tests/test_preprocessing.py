"""Tests for schema validation, temporal feature generation, Tauktae
exclusion, and per-station scaling."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import add_temporal_features, FEATURE_LIST, PHYSICAL_FEATURES
from src.preprocessing import (
    split_tauktae_window, fit_station_scalers, apply_station_scalers, TAUKTAE_STATION,
)
from src.data_loader import _validate_schema


def _make_hourly_df(station_id, start="2021-01-01", n=48):
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "station_id": station_id, "timestamp": ts.astype(str),
        "temperature_c": np.random.normal(28, 2, n),
        "pressure_hpa": np.random.normal(1008, 3, n),
        "humidity_pct": np.random.normal(60, 5, n),
        "latitude": 21.0, "longitude": 71.0,
    })


def test_schema_validation_passes_on_valid_df():
    df = _make_hourly_df("AWS_TEST")
    _validate_schema(df)  # should not raise


def test_schema_validation_fails_on_missing_field():
    df = _make_hourly_df("AWS_TEST").drop(columns=["humidity_pct"])
    with pytest.raises(ValueError):
        _validate_schema(df)


def test_temporal_features_bounded_and_present():
    df = _make_hourly_df("AWS_TEST")
    out = add_temporal_features(df)
    for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]:
        assert col in out.columns
        assert out[col].between(-1.0, 1.0).all()


def test_temporal_features_hour_cycle_correct():
    df = _make_hourly_df("AWS_TEST", n=24)
    out = add_temporal_features(df)
    # hour_sin at hour=0 should be 0, hour_cos at hour=0 should be 1
    assert abs(out["hour_sin"].iloc[0]) < 1e-9
    assert abs(out["hour_cos"].iloc[0] - 1.0) < 1e-9


def test_tauktae_window_excluded_and_preserved():
    df = _make_hourly_df(TAUKTAE_STATION, start="2021-05-14", n=168)  # covers the window
    eligible, holdout = split_tauktae_window(df)
    assert len(holdout) == 72  # 3 days * 24h
    assert len(eligible) + len(holdout) == len(df)
    # no overlap
    assert set(eligible["timestamp"]).isdisjoint(set(holdout["timestamp"]))


def test_tauktae_exclusion_only_affects_aws_diu():
    df = _make_hourly_df("AWS_OTHER", start="2021-05-14", n=168)
    eligible, holdout = split_tauktae_window(df)
    assert len(holdout) == 0
    assert len(eligible) == len(df)


def test_scaler_fit_transform_only_never_refits_on_unseen_station():
    train_df = add_temporal_features(_make_hourly_df("AWS_A", n=100))
    scalers = fit_station_scalers(train_df)
    other_df = add_temporal_features(_make_hourly_df("AWS_B", n=10))
    with pytest.raises(ValueError):
        apply_station_scalers(other_df, scalers)


def test_scaler_output_standardized_on_train():
    train_df = add_temporal_features(_make_hourly_df("AWS_A", n=500))
    scalers = fit_station_scalers(train_df)
    scaled = apply_station_scalers(train_df, scalers)
    for col in PHYSICAL_FEATURES:
        assert abs(scaled[f"{col}_scaled"].mean()) < 1e-6
        assert abs(scaled[f"{col}_scaled"].std(ddof=0) - 1.0) < 1e-6
