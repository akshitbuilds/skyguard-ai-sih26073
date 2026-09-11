# Agent 3 — ML-Based Anomaly Detection (LSTM Autoencoder)

## Purpose

Agent 3 trains an unsupervised LSTM Autoencoder on normal AWS (Automatic
Weather Station) sensor behavior and uses reconstruction error to flag
sequences that look unusual — sensor faults (spikes, frozen readings,
dropouts, drift) or genuine extreme weather events. It outputs
`reconstruction_error` and a bounded, deterministic `ml_anomaly_score` for
every scored window. It does **not** classify *why* something is
anomalous (fault type vs. real event) — that root-cause reasoning belongs
to Agent 4, downstream.

## Dataset

Agent 3 does not own or duplicate the shared data. It expects, at a path
given by the `AGENT3_DATA_DIR` environment variable (default: `../data`
relative to this repo, i.e. a sibling `data/` directory — point this at
Agent 1's shared data directory in your deployment):

```
schema.json
clean_dataset.csv
faulty_dataset.csv
ground_truth_labels.csv
```

**Why these are not committed to this repository:** they are shared,
canonical inputs owned by Agent 1's data pipeline (per `schema.json`'s own
description: *"the exact record shape every downstream agent should
expect"*). Duplicating multi-megabyte CSVs into every downstream agent's
repo would let copies drift out of sync with the canonical source. Only
this repo's own **derived artifacts** (trained model, scaler, thresholds,
metadata — see `models/`) are committed, since those are Agent 3's actual
work product and Agent 6 needs them without re-running training.

If your deployment prefers datasets physically inside this repo instead,
point `AGENT3_DATA_DIR` at a local `data/` folder — `src/data_loader.py`
already supports this via the environment variable; no code changes needed.
The source data files themselves are never modified by any script in this
repository (verified by checksum in the original development log).

## Training rules (confirmed)

- **Only `clean_dataset.csv`** is used to fit the scaler and train the
  model (`train.py`).
- `faulty_dataset.csv` and `ground_truth_labels.csv` are read **only** by
  `evaluate.py`, never by `train.py`.

## Tauktae exclusion

`AWS_DIU`, **2021-05-16 → 2021-05-18** (72 hourly rows) is excluded from
the training pool. This window covers Cyclone Tauktae — a real, documented
weather event, not a synthetic fault. Training on it would teach the model
to treat a genuine extreme event as normal, defeating the point of anomaly
detection. The rows are **not deleted** from `clean_dataset.csv` — they are
held out and specifically evaluated in `reports/evaluation_report.md` §10
as an "unseen event" test: does the model correctly find a real,
never-seen extreme event unusual?

## Model

`Input(seq_len=12, 7 features) → LSTM(64) → LSTM(32, latent) →
RepeatVector(12) → LSTM(32) → LSTM(64) → TimeDistributed(Dense(7))`. Adam
optimizer, MSE loss. SEQ_LEN=12h was chosen after testing 12h/24h/48h both
on validation MSE and on actual detection F1 against labeled faults — see
`reports/experiment_log.csv` and `reports/evaluation_report.md` §5.

## Features

7-dimensional per timestep: `temperature_c_scaled, pressure_hpa_scaled,
humidity_pct_scaled, hour_sin, hour_cos, doy_sin, doy_cos`. Physical
variables are scaled with a **separate StandardScaler per station** (fit on
that station's training rows only) rather than including station identity
as a feature, to avoid identity-shortcut leakage — latitude/longitude are
constant per station and were excluded for the same reason. Full
justification (variance-explained analysis) in
`reports/evaluation_report.md` §4.

## Threshold

Five candidates (`p95, p97.5, p99, mean+2σ, mean+3σ`) were derived from the
**validation split only** (100% normal data — never from labeled faults).
Each candidate's precision/recall/F1/FPR against labeled faults was studied
once; the best-F1 candidate was locked and never re-tuned afterward.
**Locked threshold: 0.0540** (`mean + 2σ`).

## Evaluation

Window-level: **Precision 0.140, Recall 0.065, F1 0.089** (FPR 5.7%).
Per-fault-type recall: spike 0.664, frozen 0.054, dropout 0.035, drift
0.036. **A simple rolling z-score baseline matches or beats the LSTM at a
matched false-positive rate** (F1 0.123 vs 0.089) — reported honestly, not
hidden; see `reports/evaluation_report.md` §13-14 for the full failure
analysis and why. These are synthetic-fault results and are **not
equivalent to real-world deployment validation**.

## Tauktae (unseen-event) results

Mean reconstruction error on the 61 held-out Tauktae windows was **0.1023,
5.8× the normal validation mean**; mean `ml_anomaly_score` was **0.932**,
and **31.1%** of windows were flagged anomalous (vs a 5.7% baseline false
positive rate). The model correctly identifies this genuinely unseen
extreme weather event as unusual. Full detail in
`reports/evaluation_report.md` §10 and `plots/tauktae_analysis.png`.

## Integration — instructions for Agent 6

```python
from src.inference import Agent3AnomalyDetector

detector = Agent3AnomalyDetector()          # loads model+scaler+threshold ONCE, no retraining
result = detector.predict(station_df)       # station_df: station_id, timestamp, temperature_c,
                                             # pressure_hpa, humidity_pct — ≥12 consecutive hourly rows
```

`result` is a DataFrame, one row per valid 12h sliding window, with columns:

| Column | Meaning |
|---|---|
| `station_id` | which station |
| `seq_start_ts` / `seq_end_ts` | window span (use `seq_end_ts` as "the evaluated hour") |
| `reconstruction_error` | raw MSE, primary signal |
| `ml_anomaly_score` | bounded [0,1], ECDF-based, deterministic — see `src/threshold.py` docstring |
| `is_anomaly` | 1 if `reconstruction_error > threshold` (0.0540) else 0 |

Runnable example: `examples/inference_example.py`. Root-cause / fault-type
classification is **not** performed here — that is Agent 4's responsibility,
consuming `reconstruction_error`/`ml_anomaly_score` as input signals.

## Repository structure

```
agent3_ml_detection/
├── README.md                 <- this file
├── requirements.txt
├── .gitignore
├── train.py                  <- trains on clean_dataset.csv only
├── evaluate.py                <- evaluates against faulty_dataset.csv + labels, locks threshold
├── src/
│   ├── data_loader.py         <- schema-validated CSV loading
│   ├── preprocessing.py       <- Tauktae exclusion, per-station scaling, dropout imputation
│   ├── feature_engineering.py <- cyclic temporal features
│   ├── sequence_builder.py    <- gap-aware sliding-window sequences
│   ├── lstm_autoencoder.py    <- model architecture
│   ├── threshold.py           <- reconstruction error + ml_anomaly_score
│   ├── inference.py           <- Agent 6 entry point (Agent3AnomalyDetector)
│   └── evaluation.py          <- metric helpers
├── models/                    <- COMMITTED: everything needed for inference, no retraining
│   ├── lstm_autoencoder.keras
│   ├── scaler.joblib           (per-station StandardScalers, dict)
│   ├── train_error_reference_sorted.npy  (ml_anomaly_score reference distribution)
│   └── model_metadata.json
├── reports/
│   ├── evaluation_report.md   <- full findings, plots, failure analysis
│   ├── metrics.csv
│   └── experiment_log.csv     <- all tested configurations incl. baseline
├── plots/                     <- 7 required plots, see evaluation_report.md for context
├── tests/                     <- 21 tests, all passing (see Reproduction below)
└── examples/
    └── inference_example.py   <- runnable Agent 6-style usage example
```

## Reproduction

```bash
pip install -r requirements.txt

# Train (reads ONLY clean_dataset.csv from $AGENT3_DATA_DIR or --data-dir)
python3 train.py --data-dir /path/to/data --seq-len 12 --epochs 80

# Evaluate + lock threshold (reads faulty_dataset.csv + ground_truth_labels.csv)
python3 evaluate.py --data-dir /path/to/data

# Run tests
python3 -m pytest tests/ -v

# Try inference exactly as Agent 6 would
python3 examples/inference_example.py
```

The committed `models/` artifacts already reflect a completed run of the
above (SEQ_LEN=12, threshold=0.0540) — Agent 6 does **not** need to run
`train.py` or `evaluate.py` to use this repo; only `examples/inference_example.py`
/ `src/inference.py`.

## Known limitations (see `reports/evaluation_report.md` §14-15 for full detail)

- Drift and dropout detection are weak (recall 3-4%) for documented,
  specific technical reasons — not silently hidden.
- A simple rolling z-score baseline is competitive with or better than this
  LSTM on this dataset/fault mix.
- Forward-fill imputation of dropout NaNs makes dropout resemble "frozen"
  to the model.
- Results are on synthetic injected faults; real-world validation would
  require live deployment data.
