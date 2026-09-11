"""
Reads clean_dataset.csv, injects faults, writes two files:

  faulty_dataset.csv     - same schema as clean_dataset.csv, but with
                            faults applied. This is what Agent 2 sees.
  ground_truth_labels.csv - station_id, timestamp, fault_type. This is
                            the answer key. Keep it separate so nobody
                            accidentally trains on it.

Fault types (synthetic, injected by this script):
  spike     - one reading jumps far outside its normal range
  frozen    - value repeats unchanged for several consecutive readings
  dropout   - ALL fields go missing for one reading (simulates a full
              communication failure, not a single sensor glitch)
  drift     - value gradually drifts away from truth over a window

Real event labels (not injected -- these are genuine documented weather
events sitting inside the historical data we pulled):
  genuine_event - a real, severe weather event. These rows are PROTECTED
                   from random fault injection and instead labeled as
                   genuine_event, so we have ground truth to prove the
                   system doesn't mistake a real disaster signal for a
                   sensor fault -- the exact failure mode this project's
                   whole design is built around avoiding.
"""

import csv
import random

random.seed(7)

FAULT_PROB = 0.02          # chance a given reading starts a fault
FROZEN_LEN = (4, 10)        # how many readings a frozen fault lasts
DRIFT_LEN = (10, 30)         # how many readings a drift fault lasts
DRIFT_MAX_OFFSET = 6.0      # max degrees/hPa/percent drifted away by

FIELDS = ["temperature_c", "pressure_hpa", "humidity_pct"]

# Real documented extreme-weather events inside our data window.
# Cyclone Tauktae made landfall near Una/Diu, Gujarat, around 17-18 May
# 2021 -- one of our exact 5 stations. We protect this window from random
# fault injection and label it separately as a genuine event.
PROTECTED_EVENTS = [
    {
        "station_id": "AWS_DIU",
        "start": "2021-05-16T00:00:00",
        "end": "2021-05-18T23:59:59",
        "label": "genuine_event",
    },
]


def matching_protected_event(station_id, timestamp):
    for event in PROTECTED_EVENTS:
        if (station_id == event["station_id"]
                and event["start"] <= timestamp <= event["end"]):
            return event
    return None


def inject_faults(rows):
    """rows: list of dicts with station_id, timestamp, temperature_c,
    pressure_hpa, humidity_pct, latitude, longitude"""
    faulty_rows = [dict(r) for r in rows]
    labels = []

    # First pass: label the protected real-event window. Not touched by
    # random injection at all -- these rows keep their real values.
    for row in faulty_rows:
        event = matching_protected_event(row["station_id"], row["timestamp"])
        if event is not None:
            labels.append((row["station_id"], row["timestamp"], event["label"]))

    i = 0
    n = len(faulty_rows)
    while i < n:
        station = faulty_rows[i]["station_id"]
        timestamp = faulty_rows[i]["timestamp"]

        # Never inject a synthetic fault on top of a protected real event
        if matching_protected_event(station, timestamp) is not None:
            i += 1
            continue

        if random.random() < FAULT_PROB:
            fault_type = random.choice(["spike", "frozen", "dropout", "drift"])

            if fault_type == "spike":
                field = random.choice(FIELDS)
                orig = float(faulty_rows[i][field])
                sign = random.choice([1, -1])
                faulty_rows[i][field] = round(orig + sign * random.uniform(15, 30), 2)
                labels.append((station, timestamp, "spike"))
                i += 1

            elif fault_type == "frozen":
                field = random.choice(FIELDS)
                length = random.randint(*FROZEN_LEN)
                frozen_value = faulty_rows[i][field]
                for j in range(i, min(i + length, n)):
                    if faulty_rows[j]["station_id"] != station:
                        break
                    if matching_protected_event(faulty_rows[j]["station_id"], faulty_rows[j]["timestamp"]) is not None:
                        break
                    faulty_rows[j][field] = frozen_value
                    labels.append((station, faulty_rows[j]["timestamp"], "frozen"))
                i += length

            elif fault_type == "dropout":
                # A comm failure knocks out the whole reading, not one field
                for field in FIELDS:
                    faulty_rows[i][field] = ""
                labels.append((station, timestamp, "dropout"))
                i += 1

            elif fault_type == "drift":
                field = random.choice(FIELDS)
                length = random.randint(*DRIFT_LEN)
                offset_step = DRIFT_MAX_OFFSET / length
                for j in range(i, min(i + length, n)):
                    if faulty_rows[j]["station_id"] != station:
                        break
                    if matching_protected_event(faulty_rows[j]["station_id"], faulty_rows[j]["timestamp"]) is not None:
                        break
                    steps_in = j - i
                    orig = float(faulty_rows[j][field])
                    faulty_rows[j][field] = round(orig + offset_step * steps_in, 2)
                    labels.append((station, faulty_rows[j]["timestamp"], "drift"))
                i += length
        else:
            i += 1

    return faulty_rows, labels


def main():
    with open("clean_dataset.csv") as f:
        rows = list(csv.DictReader(f))

    faulty_rows, labels = inject_faults(rows)

    fieldnames = ["station_id", "timestamp", "temperature_c", "pressure_hpa",
                  "humidity_pct", "latitude", "longitude"]
    with open("faulty_dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(faulty_rows)

    with open("ground_truth_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "timestamp", "fault_type"])
        writer.writerows(labels)

    n_genuine = sum(1 for l in labels if l[2] == "genuine_event")
    n_synthetic = len(labels) - n_genuine
    print(f"Wrote faulty_dataset.csv ({len(faulty_rows)} rows)")
    print(f"Wrote ground_truth_labels.csv ({len(labels)} labeled readings: "
          f"{n_synthetic} synthetic faults, {n_genuine} real genuine-event rows)")


if __name__ == "__main__":
    main()
