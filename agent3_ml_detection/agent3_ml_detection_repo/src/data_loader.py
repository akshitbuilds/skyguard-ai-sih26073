"""
data_loader.py -- load AWS sensor CSVs conforming to schema.json.

Agent 3 does NOT own the source data. By default it expects clean_dataset.csv,
faulty_dataset.csv, ground_truth_labels.csv and schema.json to be reachable
via the DATA_DIR path below (see README "Dataset" section for why these are
not duplicated into this repo). Override with the AGENT3_DATA_DIR
environment variable if Agent 1's shared data directory lives elsewhere.
"""
import json
import os
import pandas as pd

DATA_DIR = os.environ.get("AGENT3_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))

REQUIRED_FIELDS = ["station_id", "timestamp", "temperature_c", "pressure_hpa",
                    "humidity_pct", "latitude", "longitude"]


def load_schema(data_dir=None):
    data_dir = data_dir or DATA_DIR
    with open(os.path.join(data_dir, "schema.json")) as f:
        return json.load(f)


def load_clean_dataset(data_dir=None):
    """Load clean_dataset.csv -- the ONLY file that may be used for training."""
    data_dir = data_dir or DATA_DIR
    df = pd.read_csv(os.path.join(data_dir, "clean_dataset.csv"))
    _validate_schema(df)
    return df


def load_faulty_dataset(data_dir=None):
    """Load faulty_dataset.csv -- EVALUATION ONLY. Never used for fitting."""
    data_dir = data_dir or DATA_DIR
    df = pd.read_csv(os.path.join(data_dir, "faulty_dataset.csv"))
    _validate_schema(df)
    return df


def load_ground_truth_labels(data_dir=None):
    """Load ground_truth_labels.csv -- EVALUATION ONLY. Never used for fitting.
    Columns: station_id, timestamp, fault_type. Any (station_id, timestamp)
    NOT present here is implicitly 'normal' -- there is no explicit normal row."""
    data_dir = data_dir or DATA_DIR
    df = pd.read_csv(os.path.join(data_dir, "ground_truth_labels.csv"))
    return df


def _validate_schema(df):
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required schema fields: {missing}")
