#!/usr/bin/env python3
"""
train.py -- reproduce the full training pipeline from raw data to a saved
model. Uses ONLY clean_dataset.csv. Never touches faulty_dataset.csv or
ground_truth_labels.csv.

Usage:
    python3 train.py --data-dir /path/to/data --seq-len 12 --epochs 80

See README.md "Reproduction" for exact commands.
"""
import argparse
import json
import os
import random

import joblib
import numpy as np
import tensorflow as tf

from src.data_loader import load_clean_dataset
from src.feature_engineering import add_temporal_features, FEATURE_LIST
from src.preprocessing import (
    split_tauktae_window, chronological_split,
    fit_station_scalers, apply_station_scalers,
)
from src.sequence_builder import build_sequences
from src.lstm_autoencoder import build_model
from src.threshold import compute_reconstruction_error, threshold_candidates, MLAnomalyScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="Directory containing clean_dataset.csv etc.")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] Loading clean_dataset.csv ...")
    df = load_clean_dataset(args.data_dir)

    print("[2/6] Excluding AWS_DIU Tauktae window (2021-05-16..18) from training pool ...")
    training_eligible, tauktae_holdout = split_tauktae_window(df)
    print(f"      excluded {len(tauktae_holdout)} rows, {len(training_eligible)} remain eligible")

    print("[3/6] Chronological per-station train/val/test split (70/15/15) ...")
    train_df, val_df, test_df = chronological_split(training_eligible)
    print(f"      train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("[4/6] Temporal features + per-station scaling (fit on TRAIN only) ...")
    train_df = add_temporal_features(train_df)
    val_df = add_temporal_features(val_df)
    scalers = fit_station_scalers(train_df)
    train_df = apply_station_scalers(train_df, scalers)
    val_df = apply_station_scalers(val_df, scalers)
    joblib.dump(scalers, os.path.join(args.out_dir, "scaler.joblib"))

    print(f"[5/6] Building SEQ_LEN={args.seq_len} sequences ...")
    X_train, _, _, _ = build_sequences(train_df, args.seq_len, args.stride, FEATURE_LIST)
    X_val, _, _, _ = build_sequences(val_df, args.seq_len, args.stride, FEATURE_LIST)
    print(f"      train sequences={X_train.shape}  val sequences={X_val.shape}")

    print("[6/6] Training LSTM Autoencoder ...")
    model = build_model(args.seq_len, len(FEATURE_LIST), args.learning_rate)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
    )
    model.fit(
        X_train, X_train, validation_data=(X_val, X_val),
        epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
        callbacks=[early_stop], verbose=2,
    )
    model.save(os.path.join(args.out_dir, "lstm_autoencoder.keras"))

    # Threshold candidates + ml_anomaly_score reference, derived from VAL only
    train_err, _ = compute_reconstruction_error(model, X_train)
    val_err, _ = compute_reconstruction_error(model, X_val)
    candidates = threshold_candidates(val_err)
    MLAnomalyScorer(np.sort(train_err)).save(
        os.path.join(args.out_dir, "train_error_reference_sorted.npy")
    )

    metadata = {
        "model_type": "LSTM_Autoencoder",
        "sequence_length": args.seq_len,
        "stride": args.stride,
        "features": FEATURE_LIST,
        "threshold_candidates": candidates,
        "threshold": None,  # NOTE: final threshold must be selected via evaluate.py
                              # against labeled faulty data, not set here -- see README
        "threshold_method": None,
        "training_dataset": "clean_dataset.csv",
        "excluded_training_event": {
            "station_id": "AWS_DIU", "start": "2021-05-16", "end": "2021-05-18",
            "event": "Tauktae", "excluded_row_count": int(len(tauktae_holdout)),
        },
    }
    with open(os.path.join(args.out_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Model + scaler + threshold candidates saved to {args.out_dir}/")
    print("Run evaluate.py next to select and lock the final threshold against labeled faults.")


if __name__ == "__main__":
    main()
