"""
sequence_builder.py -- gap-aware sliding-window sequence generation.

Sequences are built independently within each time-contiguous run per
station, so a window can never silently bridge a removed time gap (e.g. the
AWS_DIU Tauktae exclusion, which splits that station's training timeline
into two separate contiguous blocks -- verified explicitly in
reports/evaluation_report.md).
"""
import numpy as np
import pandas as pd

from .feature_engineering import FEATURE_LIST


def identify_contiguous_runs(df, station_col="station_id", timestamp_col="timestamp"):
    """Assigns a run_id per row: increments whenever the gap to the previous
    row (same station) is not exactly 1 hour."""
    df = df.sort_values([station_col, timestamp_col]).reset_index(drop=True)
    ts = pd.to_datetime(df[timestamp_col])
    run_id = np.zeros(len(df), dtype=int)
    current_run = 0
    for i in range(1, len(df)):
        same_station = df[station_col].iloc[i] == df[station_col].iloc[i - 1]
        one_hour_gap = (ts.iloc[i] - ts.iloc[i - 1]) == pd.Timedelta(hours=1)
        if not (same_station and one_hour_gap):
            current_run += 1
        run_id[i] = current_run
    df = df.copy()
    df["_run_id"] = run_id
    return df


def build_sequences(df, seq_len, stride=1, feature_list=None,
                     station_col="station_id", timestamp_col="timestamp"):
    """Slide a window of length seq_len (step=stride) within each
    (station_id, run_id) contiguous block.

    Returns:
      X: np.ndarray, shape (n_sequences, seq_len, n_features), float32
      station_id: np.ndarray of station ids, one per sequence
      start_ts: np.ndarray of each sequence's first timestamp
      end_ts: np.ndarray of each sequence's last timestamp (the evaluated
              point, by convention -- see inference.py)
    """
    feature_list = feature_list or FEATURE_LIST
    df = identify_contiguous_runs(df, station_col, timestamp_col)
    X_list, station_list, start_list, end_list = [], [], [], []
    for (sid, run_id), g in df.groupby([station_col, "_run_id"]):
        g = g.sort_values(timestamp_col).reset_index(drop=True)
        n = len(g)
        if n < seq_len:
            continue
        feats = g[feature_list].values
        ts = g[timestamp_col].values
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            X_list.append(feats[start:end])
            station_list.append(sid)
            start_list.append(ts[start])
            end_list.append(ts[end - 1])
    if not X_list:
        return (np.empty((0, seq_len, len(feature_list)), dtype=np.float32),
                np.array([]), np.array([]), np.array([]))
    X = np.stack(X_list).astype(np.float32)
    return X, np.array(station_list), np.array(start_list), np.array(end_list)
