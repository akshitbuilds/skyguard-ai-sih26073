"""
preprocessing.py -- Tauktae exclusion, per-station StandardScaler fit/transform,
and missing-value handling for evaluation-only faulty data.

Why per-station scaling instead of a station_id feature:
latitude/longitude are constant per station (verified: 1 unique value each),
i.e. they are a direct proxy for station identity. Including station
identity as a model input would let the LSTM shortcut "which station is
this" instead of learning genuine temporal/seasonal behavior. Station
baseline differences are real, though (mean temp varies 26.7-28.4 C across
stations) -- handled instead via a SEPARATE StandardScaler fit per station,
using only that station's TRAINING rows.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_engineering import PHYSICAL_FEATURES

TAUKTAE_STATION = "AWS_DIU"
TAUKTAE_START = "2021-05-16T00:00:00"
TAUKTAE_END = "2021-05-18T23:00:00"


def split_tauktae_window(df, timestamp_col="timestamp", station_col="station_id"):
    """Returns (training_eligible_df, tauktae_holdout_df). Does not mutate df
    on disk -- caller decides what (if anything) to write out."""
    ts = pd.to_datetime(df[timestamp_col])
    mask = (df[station_col] == TAUKTAE_STATION) & (ts >= TAUKTAE_START) & (ts <= TAUKTAE_END)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def chronological_split(df, train_frac=0.70, val_frac=0.15,
                         timestamp_col="timestamp", station_col="station_id"):
    """Per-station, non-shuffled, chronological train/val/test split."""
    train_parts, val_parts, test_parts = [], [], []
    df = df.copy()
    df["_ts"] = pd.to_datetime(df[timestamp_col])
    for sid, g in df.sort_values("_ts").groupby(station_col):
        g = g.sort_values("_ts").reset_index(drop=True)
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val:])
    drop = lambda d: pd.concat(d).sort_values(["station_id", "_ts"]).drop(columns="_ts").reset_index(drop=True)
    return drop(train_parts), drop(val_parts), drop(test_parts)


def fit_station_scalers(train_df, station_col="station_id"):
    """Fit one StandardScaler per station on TRAIN rows only. Returns dict."""
    scalers = {}
    for sid in train_df[station_col].unique():
        mask = train_df[station_col] == sid
        scaler = StandardScaler()
        scaler.fit(train_df.loc[mask, PHYSICAL_FEATURES])
        scalers[sid] = scaler
    return scalers


def apply_station_scalers(df, scalers, station_col="station_id"):
    """Transform-only using already-fitted scalers. Raises if a station in df
    was never seen during fitting (unseen-station transform is a bug, not
    something to silently paper over)."""
    df = df.copy()
    unseen = set(df[station_col].unique()) - set(scalers.keys())
    if unseen:
        raise ValueError(f"No fitted scaler for station(s): {unseen}")
    scaled_cols = {f"{c}_scaled": np.zeros(len(df)) for c in PHYSICAL_FEATURES}
    for sid in df[station_col].unique():
        mask = df[station_col] == sid
        transformed = scalers[sid].transform(df.loc[mask, PHYSICAL_FEATURES])
        for i, c in enumerate(PHYSICAL_FEATURES):
            scaled_cols[f"{c}_scaled"][mask.values] = transformed[:, i]
    for k, v in scaled_cols.items():
        df[k] = v
    return df


def impute_missing_physical_values(df, station_col="station_id"):
    """Forward-fill then back-fill missing physical readings, per station.

    KNOWN LIMITATION: this makes a 'dropout' fault window resemble a
    'frozen' window to the model (both become locally constant), which
    measurably reduces recall for dropout specifically -- see
    reports/evaluation_report.md 'Failure Analysis' for the quantified
    effect (dropout recall 3.5% vs e.g. spike recall 66.4%).
    """
    df = df.copy()
    df[PHYSICAL_FEATURES] = df.groupby(station_col)[PHYSICAL_FEATURES].transform(
        lambda s: s.ffill().bfill()
    )
    return df
