"""
Streams a dataset CSV as if it were live sensor data - one JSON line
per reading, interleaved across stations in timestamp order, with a
small delay between readings to simulate real-time arrival.

Usage:
  python3 replay.py                        # replay faulty_dataset.csv, fast (no delay)
  python3 replay.py --speed 0.05           # 0.05 sec delay between readings
  python3 replay.py --file clean_dataset.csv --speed 0.1

Agent 2 (or anyone downstream) should just read stdout line by line,
json.loads() each line, and process it as one incoming sensor reading.
"""

import argparse
import csv
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="faulty_dataset.csv")
    parser.add_argument("--speed", type=float, default=0.0,
                         help="seconds to sleep between readings (0 = as fast as possible)")
    args = parser.parse_args()

    with open(args.file) as f:
        rows = list(csv.DictReader(f))

    # interleave stations in timestamp order, like a real multi-station feed
    rows.sort(key=lambda r: r["timestamp"])

    for row in rows:
        # keep numeric fields numeric, blank strings mean a dropout - pass through as null
        record = {
            "station_id": row["station_id"],
            "timestamp": row["timestamp"],
            "temperature_c": float(row["temperature_c"]) if row["temperature_c"] != "" else None,
            "pressure_hpa": float(row["pressure_hpa"]) if row["pressure_hpa"] != "" else None,
            "humidity_pct": float(row["humidity_pct"]) if row["humidity_pct"] != "" else None,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }
        try:
            print(json.dumps(record))
        except BrokenPipeError:
            # downstream reader (e.g. `| head`) stopped listening -- not an error
            return
        if args.speed > 0:
            time.sleep(args.speed)


if __name__ == "__main__":
    main()
