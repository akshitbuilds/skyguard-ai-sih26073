"""
inference.py -- Agent 6 integration point.

Loads the trained model + scalers + threshold + config ONCE (no retraining,
no re-fitting of anything) and exposes a simple predict() function that
takes a per-station dataframe of consecutive hourly readings (station_id,
timestamp, temperature_c, pressure_hpa, humidity_pct) and returns, for every
valid sliding window:

  station_id, seq_start_ts, seq_end_ts, reconstruction_error,
  ml_anomaly_score, is_anomaly

See examples/inference_example.py for a runnable end-to-end example.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .feature_engineering import add_temporal_features, FEATURE_LIST
from .preprocessing import apply_station_scalers, impute_missing_physical_values
from .sequence_builder import build_sequences
from .threshold import compute_reconstruction_error, MLAnomalyScorer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class Agent3AnomalyDetector:
    def __init__(self, models_dir=None):
        models_dir = models_dir or MODELS_DIR
        with open(os.path.join(models_dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)
        self.model = tf.keras.models.load_model(
            os.path.join(models_dir, "lstm_autoencoder.keras")
        )
        self.scalers = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.scorer = MLAnomalyScorer.from_file(
            os.path.join(models_dir, "train_error_reference_sorted.npy")
        )
        self.seq_len = self.metadata["sequence_length"]
        self.stride = self.metadata["stride"]
        self.threshold = self.metadata["threshold"]
        self.feature_list = self.metadata["features"]

    def predict(self, df):
        """df: DataFrame with columns station_id, timestamp, temperature_c,
        pressure_hpa, humidity_pct (at minimum -- latitude/longitude are
        accepted but ignored, matching the schema). Rows should be hourly
        and consecutive per station; missing physical values are
        forward/back-filled per station (documented limitation -- see
        README 'Known limitations').

        Returns a DataFrame with one row per valid sliding window:
        station_id, seq_start_ts, seq_end_ts, reconstruction_error,
        ml_anomaly_score, is_anomaly.
        """
        df = impute_missing_physical_values(df)
        df = add_temporal_features(df)
        df = apply_station_scalers(df, self.scalers)

        X, station_id, start_ts, end_ts = build_sequences(
            df, seq_len=self.seq_len, stride=self.stride, feature_list=self.feature_list
        )
        if X.shape[0] == 0:
            return pd.DataFrame(columns=["station_id", "seq_start_ts", "seq_end_ts",
                                          "reconstruction_error", "ml_anomaly_score", "is_anomaly"])

        reconstruction_error, _ = compute_reconstruction_error(self.model, X)
        ml_anomaly_score = self.scorer.score(reconstruction_error)
        is_anomaly = (reconstruction_error > self.threshold).astype(int)

        return pd.DataFrame({
            "station_id": station_id,
            "seq_start_ts": start_ts,
            "seq_end_ts": end_ts,
            "reconstruction_error": reconstruction_error,
            "ml_anomaly_score": ml_anomaly_score,
            "is_anomaly": is_anomaly,
        })
