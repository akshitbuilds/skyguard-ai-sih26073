"""Tests for gap-aware sliding-window sequence generation."""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import add_temporal_features, FEATURE_LIST, PHYSICAL_FEATURES
from src.preprocessing import fit_station_scalers, apply_station_scalers
from src.sequence_builder import build_sequences, identify_contiguous_runs


def _make_scaled_df(station_id, start="2021-01-01", n=48, drop_hours=None):
    ts = pd.date_range(start, periods=n, freq="h")
    df = pd.DataFrame({
        "station_id": station_id, "timestamp": ts.astype(str),
        "temperature_c": np.random.normal(28, 2, n),
        "pressure_hpa": np.random.normal(1008, 3, n),
        "humidity_pct": np.random.normal(60, 5, n),
    })
    if drop_hours:
        df = df.drop(df.index[drop_hours]).reset_index(drop=True)
    df = add_temporal_features(df)
    scalers = fit_station_scalers(df)
    return apply_station_scalers(df, scalers)


def test_sequence_shape_matches_seq_len_and_feature_count():
    df = _make_scaled_df("AWS_TEST", n=48)
    seq_len = 12
    X, station_id, start_ts, end_ts = build_sequences(df, seq_len)
    assert X.shape[1] == seq_len
    assert X.shape[2] == len(FEATURE_LIST)
    # 48 hours -> 48 - 12 + 1 = 37 windows at stride 1
    assert X.shape[0] == 48 - seq_len + 1
    assert len(station_id) == len(start_ts) == len(end_ts) == X.shape[0]


def test_sequence_count_respects_stride():
    df = _make_scaled_df("AWS_TEST", n=48)
    seq_len, stride = 12, 3
    X, *_ = build_sequences(df, seq_len, stride=stride)
    expected = len(range(0, 48 - seq_len + 1, stride))
    assert X.shape[0] == expected


def test_no_window_shorter_than_seq_len_is_produced():
    df = _make_scaled_df("AWS_TEST", n=5)  # shorter than seq_len
    X, *_ = build_sequences(df, seq_len=12)
    assert X.shape[0] == 0


def test_gap_detection_splits_into_multiple_runs():
    # drop hours 20-25 to create a 6-hour gap in an otherwise-hourly series
    df = _make_scaled_df("AWS_TEST", n=48, drop_hours=list(range(20, 26)))
    runs = identify_contiguous_runs(df)
    assert runs["_run_id"].nunique() == 2


def test_no_sequence_bridges_a_time_gap():
    df = _make_scaled_df("AWS_TEST", n=48, drop_hours=list(range(20, 26)))
    seq_len = 12
    X, station_id, start_ts, end_ts = build_sequences(df, seq_len)
    for s, e in zip(start_ts, end_ts):
        s_ts, e_ts = pd.Timestamp(str(s)), pd.Timestamp(str(e))
        # a valid (non-gap-bridging) window's span must equal exactly seq_len-1 hours
        assert (e_ts - s_ts) == pd.Timedelta(hours=seq_len - 1)


def test_sequences_built_independently_per_station():
    df_a = _make_scaled_df("AWS_A", n=24)
    df_b = _make_scaled_df("AWS_B", n=24)
    combined = pd.concat([df_a, df_b], ignore_index=True)
    X, station_id, *_ = build_sequences(combined, seq_len=12)
    assert set(station_id) == {"AWS_A", "AWS_B"}
    # no window should ever contain rows from both stations (guaranteed by
    # grouping on station_id before windowing -- this just double-checks
    # the count matches doing each station separately)
    Xa, *_ = build_sequences(df_a, seq_len=12)
    Xb, *_ = build_sequences(df_b, seq_len=12)
    assert X.shape[0] == Xa.shape[0] + Xb.shape[0]
