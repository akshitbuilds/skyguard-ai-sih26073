"""Tests for the packaged inference path: model loads without retraining,
and predict() returns the required output fields with sane values."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ARTIFACTS_PRESENT = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ["lstm_autoencoder.keras", "scaler.joblib",
              "model_metadata.json", "train_error_reference_sorted.npy"]
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Trained model artifacts not present in models/ -- run train.py + evaluate.py first"
)


@pytest.fixture(scope="module")
def detector():
    from src.inference import Agent3AnomalyDetector
    return Agent3AnomalyDetector()


@pytest.fixture
def sample_station_df(detector):
    n = detector.seq_len + 10
    ts = pd.date_range("2021-06-01", periods=n, freq="h")
    return pd.DataFrame({
        "station_id": "AWS_BHAVNAGAR", "timestamp": ts.astype(str),
        "temperature_c": np.random.normal(28, 2, n),
        "pressure_hpa": np.random.normal(1008, 3, n),
        "humidity_pct": np.random.normal(60, 5, n),
        "latitude": 21.7, "longitude": 72.1,
    })


def test_model_loads_without_retraining(detector):
    assert detector.model is not None
    assert detector.seq_len > 0
    assert isinstance(detector.threshold, float)


def test_predict_returns_required_output_fields(detector, sample_station_df):
    result = detector.predict(sample_station_df)
    for col in ["station_id", "seq_start_ts", "seq_end_ts",
                "reconstruction_error", "ml_anomaly_score", "is_anomaly"]:
        assert col in result.columns


def test_reconstruction_error_is_nonnegative(detector, sample_station_df):
    result = detector.predict(sample_station_df)
    assert (result["reconstruction_error"] >= 0).all()


def test_ml_anomaly_score_bounded_zero_to_one(detector, sample_station_df):
    result = detector.predict(sample_station_df)
    assert (result["ml_anomaly_score"] >= 0).all()
    assert (result["ml_anomaly_score"] <= 1).all()


def test_is_anomaly_consistent_with_threshold(detector, sample_station_df):
    result = detector.predict(sample_station_df)
    expected = (result["reconstruction_error"] > detector.threshold).astype(int)
    assert (result["is_anomaly"] == expected).all()


def test_predict_handles_missing_values_without_crashing(detector, sample_station_df):
    df = sample_station_df.copy()
    df.loc[3:5, "temperature_c"] = np.nan
    result = detector.predict(df)  # should not raise
    assert not result.empty


def test_predict_on_too_short_series_returns_empty(detector):
    short_df = pd.DataFrame({
        "station_id": ["AWS_BHAVNAGAR"] * 3,
        "timestamp": pd.date_range("2021-06-01", periods=3, freq="h").astype(str),
        "temperature_c": [28.0, 28.1, 28.2],
        "pressure_hpa": [1008.0, 1008.1, 1008.2],
        "humidity_pct": [60.0, 60.1, 60.2],
    })
    result = detector.predict(short_df)
    assert len(result) == 0
