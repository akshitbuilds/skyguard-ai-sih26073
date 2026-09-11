"""
Correction-accuracy evaluation for SkyGuard AI - Agent 5.

Runs the REAL correction.py logic (estimate_corrected_value) against the real
agent1_ingestion dataset, and measures how close the corrected/imputed value
lands to the actual pre-fault ground truth.

Data source (agent1_ingestion/):
  - clean_dataset.csv         the TRUE values, never corrupted
  - faulty_dataset.csv        same station+timestamp grid, with faults injected
  - ground_truth_labels.csv   (station_id, timestamp, fault_type) for every
                               injected fault - drift/frozen/spike/dropout/genuine_event

Method: for every (station, timestamp) labeled with a real sensor fault
(excluding genuine_event, which isn't a fault), we:
  1. Build that station's history from the FAULTY stream (only points strictly
     before the target timestamp, and only non-missing points) - this mirrors
     what a live system would actually have seen.
  2. Build a "current snapshot" of every OTHER station's faulty-stream reading
     at that same timestamp, and mark which of them are themselves flagged
     faulty at that instant (per ground_truth_labels) so they're excluded as
     spatial neighbors - the same trustworthy-neighbor logic used by the integrated Agent 5 adapter.
  3. Call estimate_corrected_value() - the exact function the dashboard uses.
  4. Compare the result to the real value in clean_dataset.csv at that
     station+timestamp.

Run:
    python correction_accuracy.py
    python correction_accuracy.py --data-dir /path/to/agent1_ingestion --sample 1000
"""

import argparse
import csv
import math
import statistics
import time
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

from correction import estimate_corrected_value, FIELDS

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "agent1_ingestion"

# Tolerance bands used for the "% within tolerance" headline accuracy number.
# These are intentionally generous-but-meaningful thresholds for AWS-grade
# sensors, not arbitrary - tighten/loosen per your team's calibration spec.
TOLERANCE = {"temperature_c": 2.0, "pressure_hpa": 3.0, "humidity_pct": 8.0}

HISTORY_WINDOW = 40  # max prior valid points kept per station for the temporal fit


def _parse_row(row):
    out = {
        "station_id": row["station_id"],
        "timestamp": datetime.fromisoformat(row["timestamp"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }
    for f in FIELDS:
        v = row.get(f, "")
        out[f] = float(v) if v not in ("", None) else None
    return out


def load_csv(path):
    with open(path, newline="") as fh:
        return [_parse_row(r) for r in csv.DictReader(fh)]


def load_labels(path):
    with open(path, newline="") as fh:
        return [
            {"station_id": r["station_id"], "timestamp": datetime.fromisoformat(r["timestamp"]),
             "fault_type": r["fault_type"]}
            for r in csv.DictReader(fh)
        ]


def build_indices(clean_rows, faulty_rows, label_rows):
    meta = {}
    for r in clean_rows:
        meta.setdefault(r["station_id"], {"latitude": r["latitude"], "longitude": r["longitude"]})

    clean_by_key = {(r["station_id"], r["timestamp"]): r for r in clean_rows}
    faulty_by_key = {(r["station_id"], r["timestamp"]): r for r in faulty_rows}
    fault_type_by_key = {(r["station_id"], r["timestamp"]): r["fault_type"] for r in label_rows}

    # per-station sorted list of VALID (non-missing) faulty-stream points, for
    # fast "give me everything before timestamp X" lookups via bisect.
    by_station_valid = {}
    for sid in meta:
        pts = sorted(
            (r for r in faulty_rows if r["station_id"] == sid and all(r[f] is not None for f in FIELDS)),
            key=lambda r: r["timestamp"],
        )
        by_station_valid[sid] = {"rows": pts, "timestamps": [r["timestamp"] for r in pts]}

    return meta, clean_by_key, faulty_by_key, fault_type_by_key, by_station_valid


def prior_history(by_station_valid, station_id, ts, window=HISTORY_WINDOW):
    entry = by_station_valid[station_id]
    idx = bisect_left(entry["timestamps"], ts)  # first index with timestamp >= ts
    start = max(0, idx - window)
    return entry["rows"][start:idx]


def evaluate(data_dir, sample=None, seed=42):
    clean_rows = load_csv(data_dir / "clean_dataset.csv")
    faulty_rows = load_csv(data_dir / "faulty_dataset.csv")
    label_rows = load_labels(data_dir / "ground_truth_labels.csv")

    meta, clean_by_key, faulty_by_key, fault_type_by_key, by_station_valid = build_indices(
        clean_rows, faulty_rows, label_rows
    )

    targets = [r for r in label_rows if r["fault_type"] != "genuine_event"]
    if sample:
        import random
        random.seed(seed)
        targets = random.sample(targets, min(sample, len(targets)))

    results = []
    skipped = 0
    t0 = time.time()

    for lr in targets:
        sid, ts, ftype = lr["station_id"], lr["timestamp"], lr["fault_type"]

        prior = prior_history(by_station_valid, sid, ts)
        if len(prior) < 3:
            skipped += 1
            continue

        hist_list = [
            {"timestamp": r["timestamp"], **{f: r[f] for f in FIELDS}} for r in prior
        ]
        # anchor entry so _temporal_estimate can compute the correct forward offset;
        # its own field values are never read (they fall inside the excluded tail).
        hist_list.append({"timestamp": ts, **{f: 0.0 for f in FIELDS}})
        histories = {sid: hist_list}

        current_snapshot = {}
        healthy_ids = []
        for other_sid in meta:
            if other_sid == sid:
                continue
            row = faulty_by_key.get((other_sid, ts))
            if row is None or any(row[f] is None for f in FIELDS):
                continue  # neighbor itself missing/dropped out right now
            current_snapshot[other_sid] = {f: row[f] for f in FIELDS}
            neighbor_fault = fault_type_by_key.get((other_sid, ts))
            if neighbor_fault in (None, "genuine_event"):
                healthy_ids.append(other_sid)

        corrected = estimate_corrected_value(sid, histories, current_snapshot, meta, healthy_ids)
        if corrected is None:
            skipped += 1
            continue

        true_row = clean_by_key.get((sid, ts))
        if true_row is None:
            skipped += 1
            continue

        row_result = {"station_id": sid, "timestamp": ts, "fault_type": ftype}
        for f in FIELDS:
            row_result[f"{f}_true"] = true_row[f]
            row_result[f"{f}_corrected"] = corrected[f]
            row_result[f"{f}_abs_err"] = abs(true_row[f] - corrected[f])
        results.append(row_result)

    elapsed = time.time() - t0
    return results, skipped, len(targets), elapsed


def summarize(results, total_targets, skipped):
    if not results:
        print("No results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"SkyGuard AI — Agent 5 Correction Accuracy Report")
    print(f"{'='*70}")
    print(f"Total labeled sensor faults (excl. genuine_event): {total_targets}")
    print(f"Corrected (evaluable):                              {len(results)}")
    print(f"Skipped (insufficient history/neighbors):           {skipped}")
    print(f"Coverage:                                           {len(results)/total_targets:.1%}\n")

    overall_within = []
    for field in FIELDS:
        errs = [r[f"{field}_abs_err"] for r in results]
        mae = statistics.mean(errs)
        rmse = math.sqrt(statistics.mean(e**2 for e in errs))
        within = sum(1 for e in errs if e <= TOLERANCE[field]) / len(errs)
        overall_within.append(within)
        print(f"  {field:16} MAE = {mae:6.2f}   RMSE = {rmse:6.2f}   "
              f"within ±{TOLERANCE[field]}: {within:.1%}")

    print(f"\n  Overall accuracy (avg. % within tolerance across all 3 fields): "
          f"{statistics.mean(overall_within):.1%}")

    print(f"\n  --- Breakdown by fault type ---")
    by_type = {}
    for r in results:
        by_type.setdefault(r["fault_type"], []).append(r)
    for ftype, rows in sorted(by_type.items()):
        print(f"\n  {ftype}  (n={len(rows)})")
        for field in FIELDS:
            errs = [r[f"{field}_abs_err"] for r in rows]
            mae = statistics.mean(errs)
            within = sum(1 for e in errs if e <= TOLERANCE[field]) / len(errs)
            print(f"    {field:16} MAE = {mae:6.2f}   within tolerance: {within:.1%}")
    print(f"\n{'='*70}\n")


def save_results_csv(results, out_path):
    if not results:
        return
    fieldnames = ["station_id", "timestamp", "fault_type"]
    for f in FIELDS:
        fieldnames += [f"{f}_true", f"{f}_corrected", f"{f}_abs_err"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["timestamp"] = row["timestamp"].isoformat()
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SkyGuard correction accuracy against real ground truth.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="Path to agent1_ingestion/ (default: ../agent1_ingestion relative to this file)")
    parser.add_argument("--sample", type=int, default=None,
                         help="Evaluate a random sample of N faults instead of all (faster iteration)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "correction_accuracy_results.csv",
                         help="Where to save the per-row results CSV")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    results, skipped, total, elapsed = evaluate(args.data_dir, sample=args.sample)
    print(f"Evaluated {len(results)} faults in {elapsed:.1f}s")

    summarize(results, total, skipped)
    save_results_csv(results, args.out)
    print(f"Per-row results saved to: {args.out}")