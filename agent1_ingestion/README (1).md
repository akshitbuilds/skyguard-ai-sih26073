# Agent 1 - Data Ingestion

## Files
- `schema.json` - the record format. Read this first.
- `generate_dataset.py` -> `clean_dataset.csv` (5 Gujarat coastal stations, hourly, May 2021, ~3,720 readings)
- `fault_injector.py` -> `faulty_dataset.csv` + `ground_truth_labels.csv`
- `replay.py` -> streams a CSV as live JSON lines to stdout
- `run_all.sh` -> runs all three in order
- `requirements.txt` -> Python packages needed to run `generate_dataset.py`

## Data source
`generate_dataset.py` pulls **real historical weather data** from the
Open-Meteo Archive API (archive-api.open-meteo.com) for 5 Gujarat
coastal stations (Diu, Veraval, Mahuva, Porbandar, Bhavnagar),
Jan 1 - Dec 31, 2021 (full year, so the model sees real seasonal
variation), ~43,800 readings.

**Validation case:** Cyclone Tauktae made landfall near Una/Diu,
Gujarat around 16-18 May 2021 - real, documented, severe weather,
sitting inside this data at station AWS_DIU. `fault_injector.py`
protects this window from random fault injection and labels it
`genuine_event` in ground_truth_labels.csv instead. This is our proof
that the system doesn't mistake a real disaster signal for a sensor
fault - the core failure mode this whole project is designed around.

## Before running - install dependencies
```bash
pip install -r requirements.txt
```

## How to run it, in order
```bash
python3 generate_dataset.py     # makes clean_dataset.csv
python3 fault_injector.py       # makes faulty_dataset.csv + ground_truth_labels.csv
python3 replay.py               # streams faulty_dataset.csv as JSON lines
```
Or all at once:
```bash
./run_all.sh
```

## For Agent 2 (anomaly detector)
Read `faulty_dataset.csv` (or pipe from `replay.py`) - that's your input,
it has no labels. Train/evaluate against `ground_truth_labels.csv`
separately. Don't let the detector see the labels file.

```bash
python3 replay.py --speed 0.05 | your_detector.py
```

## Fault types in ground_truth_labels.csv
- `spike` - one reading jumps 15-30 units off
- `frozen` - value repeats unchanged for 4-10 readings
- `dropout` - ALL fields (temp/pressure/humidity) blank together,
  simulating a full communication failure, not a single sensor glitch
- `drift` - value gradually drifts up to 6 units over 10-30 readings
- `genuine_event` - NOT a fault. A real documented weather event
  (Cyclone Tauktae, AWS_DIU, 16-18 May 2021). Never train on this
  window - it must stay unseen so it's a genuine test of whether the
  system correctly flags it as unusual without misclassifying it as
  a sensor fault.

~12.5% of readings carry a synthetic fault; 72 additional readings
are the real genuine_event window (tune `FAULT_PROB` etc. in
`fault_injector.py` if you want faults rarer/more common).

## Tuning
- Different stations / date range: edit `STATIONS` and `start_date`/`end_date` in `generate_dataset.py`
- Different fault mix: edit `FAULT_PROB`, `FROZEN_LEN`, `DRIFT_LEN`, `DRIFT_MAX_OFFSET` in `fault_injector.py`

## Note
Delete `tempCodeRunnerFile.py` if you see it in this folder - it's a
VS Code "Code Runner" auto-generated leftover, not a real project file.
