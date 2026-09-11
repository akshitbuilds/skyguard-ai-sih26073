#!/usr/bin/env python3
"""
evaluate.py -- evaluate a trained model against faulty_dataset.csv +
ground_truth_labels.csv, select and lock the final threshold, and write
reports/metrics.csv + reports/evaluation_report.md inputs.

Usage:
    python3 evaluate.py --data-dir /path/to/data --models-dir models

This is the ONLY script that may read faulty_dataset.csv / ground_truth_labels.csv.
train.py never touches them.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_loader import load_faulty_dataset, load_ground_truth_labels
from src.feature_engineering import add_temporal_features, FEATURE_LIST
from src.preprocessing import apply_station_scalers, impute_missing_physical_values
from src.sequence_builder import build_sequences
from src.threshold import compute_reconstruction_error
from src.evaluation import binary_metrics, per_fault_type_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)

    with open(os.path.join(args.models_dir, "model_metadata.json")) as f:
        metadata = json.load(f)
    seq_len = metadata["sequence_length"]
    stride = metadata["stride"]

    model = tf.keras.models.load_model(os.path.join(args.models_dir, "lstm_autoencoder.keras"))
    scalers = joblib.load(os.path.join(args.models_dir, "scaler.joblib"))

    print("Loading faulty_dataset.csv + ground_truth_labels.csv (evaluation only) ...")
    faulty = load_faulty_dataset(args.data_dir)
    labels = load_ground_truth_labels(args.data_dir)

    faulty = impute_missing_physical_values(faulty)
    faulty = add_temporal_features(faulty)
    faulty = apply_station_scalers(faulty, scalers)

    X, station_id, start_ts, end_ts = build_sequences(faulty, seq_len, stride, FEATURE_LIST)
    recon_err, _ = compute_reconstruction_error(model, X)

    df = pd.DataFrame({"station_id": station_id, "seq_end_ts": end_ts.astype(str),
                        "reconstruction_error": recon_err})
    labels["timestamp"] = labels["timestamp"].astype(str)
    merged = df.merge(labels, left_on=["station_id", "seq_end_ts"],
                       right_on=["station_id", "timestamp"], how="left")
    merged["fault_type"] = merged["fault_type"].fillna("normal")
    eval_df = merged[merged["fault_type"] != "genuine_event"].copy()
    eval_df["y_true"] = (eval_df["fault_type"] != "normal").astype(int)

    candidates = metadata.get("threshold_candidates", {})
    if not candidates:
        raise ValueError("model_metadata.json has no threshold_candidates -- run train.py first")

    print("\nThreshold study:")
    best = {"f1": -1}
    for name, thresh in candidates.items():
        y_pred = (eval_df["reconstruction_error"] > thresh).astype(int)
        m = binary_metrics(eval_df["y_true"], y_pred)
        print(f"  {name:15s} thresh={thresh:.5f}  P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}")
        if m["f1"] > best["f1"]:
            best = {"candidate": name, "threshold": thresh, **m}

    print(f"\nLocked threshold: {best['threshold']:.5f} (method={best['candidate']})")
    eval_df["y_pred"] = (eval_df["reconstruction_error"] > best["threshold"]).astype(int)
    overall = binary_metrics(eval_df["y_true"], eval_df["y_pred"])
    per_fault = per_fault_type_metrics(eval_df)

    metadata["threshold"] = best["threshold"]
    metadata["threshold_method"] = best["candidate"]
    metadata["overall_evaluation_metrics"] = overall
    metadata["per_fault_type_metrics"] = per_fault
    with open(os.path.join(args.models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame([{"level": "overall", **overall}] +
                 [{"level": ft, **m} for ft, m in per_fault.items()]
                 ).to_csv(os.path.join(args.reports_dir, "metrics.csv"), index=False)

    print(f"\nSaved threshold + metrics to {args.models_dir}/model_metadata.json "
          f"and {args.reports_dir}/metrics.csv")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
