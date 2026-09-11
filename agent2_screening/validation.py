"""
Phase 7 — Historical validation
==================================
A backtesting harness to run Agent2 against labeled historical episodes
(thunderstorms, heavy rainfall, floods, cyclones, heat waves, localized
storms, sensor failures, comms failures) and measure:

    False Positive Rate   (real weather flagged as SUSPICIOUS/sensor fault)
    False Negative Rate   (real sensor fault labeled CLEAN)
    Detection Rate        (true anomalies correctly flagged REVIEW/SUSPICIOUS)
    Processing Time       (per-reading latency)

This module does NOT ship with real historical AWS data (Agent2 doesn't
have access to it) — it defines the harness + the expected input format,
plus a synthetic dataset generator so the pipeline is testable end-to-end
today. Point `load_episodes_from_json` at real labeled data when available.

Expected label taxonomy per episode reading:
    "true_event"   -> a genuine weather event (should NOT be flagged as a fault;
                       CLEAN or REVIEW/SUSPICIOUS-as-POSSIBLE_REAL_EVENT is fine,
                       what's WRONG is reasoning.verdict == LIKELY_SENSOR_FAULT)
    "sensor_fault" -> a genuine sensor/comms failure (SHOULD be flagged; CLEAN is a miss)
    "normal"       -> unremarkable reading (should be CLEAN)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime, timedelta

from .schema import StationReading, Coordinates
from .pipeline import Agent2, AnomalyReport


@dataclass
class LabeledCase:
    label: str                     # "true_event" | "sensor_fault" | "normal"
    event_type: str                # e.g. "flood", "cyclone", "heat_wave", "comm_failure"
    current: StationReading
    neighbors: List[StationReading]
    history: List[StationReading] = field(default_factory=list)


@dataclass
class ValidationMetrics:
    n_cases: int
    false_positives: int           # true_event mislabeled as LIKELY_SENSOR_FAULT
    false_negatives: int           # sensor_fault mislabeled as CLEAN
    true_positives: int            # sensor_fault correctly flagged (REVIEW/SUSPICIOUS)
    true_negatives: int            # normal correctly labeled CLEAN
    detection_rate: float          # true_positives / (true_positives + false_negatives)
    false_positive_rate: float     # false_positives / n(true_event)
    avg_processing_time_ms: float
    by_event_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Cases evaluated       : {self.n_cases}",
            f"Detection rate        : {self.detection_rate:.1%}",
            f"False positive rate   : {self.false_positive_rate:.1%}  "
            f"(genuine events wrongly called sensor faults)",
            f"False negatives       : {self.false_negatives}  (missed real sensor faults)",
            f"Avg processing time   : {self.avg_processing_time_ms:.3f} ms/reading",
            "",
            "By event type:",
        ]
        for ev, counts in self.by_event_type.items():
            lines.append(f"  {ev:16s} n={counts['n']:3d}  correctly_handled={counts['correct']:3d}")
        return "\n".join(lines)


def run_validation(agent: Agent2, cases: List[LabeledCase]) -> ValidationMetrics:
    fp = fn = tp = tn = 0
    total_time = 0.0
    by_event: Dict[str, Dict[str, int]] = {}

    for case in cases:
        report: AnomalyReport = agent.evaluate(case.current, case.neighbors, case.history)
        total_time += report.processing_time_ms

        by_event.setdefault(case.event_type, {"n": 0, "correct": 0})
        by_event[case.event_type]["n"] += 1

        flagged = report.final_label in ("REVIEW", "SUSPICIOUS")
        called_fault = report.reasoning.verdict == "LIKELY_SENSOR_FAULT"

        if case.label == "true_event":
            if called_fault:
                fp += 1
            else:
                by_event[case.event_type]["correct"] += 1
        elif case.label == "sensor_fault":
            if flagged:
                tp += 1
                by_event[case.event_type]["correct"] += 1
            else:
                fn += 1
        elif case.label == "normal":
            if not flagged:
                tn += 1
                by_event[case.event_type]["correct"] += 1

    n_true_events = sum(1 for c in cases if c.label == "true_event")
    n_faults = sum(1 for c in cases if c.label == "sensor_fault")

    detection_rate = tp / n_faults if n_faults else float("nan")
    fp_rate = fp / n_true_events if n_true_events else float("nan")

    return ValidationMetrics(
        n_cases=len(cases),
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        true_negatives=tn,
        detection_rate=detection_rate,
        false_positive_rate=fp_rate,
        avg_processing_time_ms=total_time / len(cases) if cases else 0.0,
        by_event_type=by_event,
    )


def load_episodes_from_json(path: str) -> List[LabeledCase]:
    """
    Load real historical episodes once Data Engineering provides them.
    Expected JSON structure (list of objects):
    {
      "label": "true_event" | "sensor_fault" | "normal",
      "event_type": "flood",
      "current": {...StationReading fields...},
      "neighbors": [{...}, ...],
      "history": [{...}, ...]
    }
    """
    with open(path) as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        cases.append(LabeledCase(
            label=item["label"],
            event_type=item.get("event_type", "unknown"),
            current=StationReading.from_dict(item["current"]),
            neighbors=[StationReading.from_dict(n) for n in item.get("neighbors", [])],
            history=[StationReading.from_dict(h) for h in item.get("history", [])],
        ))
    return cases


# ---------------------------------------------------------------------------
# Synthetic dataset generator — lets Phase 7 run end-to-end before real
# historical data is wired in. NOT a substitute for real validation.
# ---------------------------------------------------------------------------

def _mk_reading(station_id, t, lat, lon, temp, hum, pres, wind, wdir, rain):
    return StationReading(
        station_id=station_id, timestamp=t, location=Coordinates(lat, lon),
        temperature_c=temp, humidity_pct=hum, pressure_hpa=pres,
        wind_speed_ms=wind, wind_direction_deg=wdir, rainfall_mm=rain,
    )


def generate_synthetic_cases(seed: int = 42, n_per_type: int = 15) -> List[LabeledCase]:
    rnd = random.Random(seed)
    cases: List[LabeledCase] = []
    base_t = datetime(2026, 6, 1, 10, 0)

    def base_neighbors(t, lat=12.9, lon=77.6, temp=30, hum=55, pres=1008, wind=3, wdir=180, rain=0):
        return [
            _mk_reading(f"N{i}", t, lat + rnd.uniform(-0.05, 0.05), lon + rnd.uniform(-0.05, 0.05),
                        temp + rnd.uniform(-1, 1), hum + rnd.uniform(-3, 3), pres + rnd.uniform(-1, 1),
                        wind + rnd.uniform(-0.5, 0.5), wdir + rnd.uniform(-10, 10), rain)
            for i in range(4)
        ]

    def history_series(station_id, lat, lon, start_vals, drift, n=6, step_min=5):
        hist = []
        vals = dict(start_vals)
        t = base_t - timedelta(minutes=step_min * n)
        for i in range(n):
            hist.append(_mk_reading(station_id, t, lat, lon, vals["temp"], vals["hum"],
                                     vals["pres"], vals["wind"], vals["wdir"], vals["rain"]))
            for k in drift:
                vals[k] += drift[k]
            t += timedelta(minutes=step_min)
        return hist

    # 1. normal cases
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        cur = _mk_reading("A", t, 12.9, 77.6, 30 + rnd.uniform(-1, 1), 55 + rnd.uniform(-3, 3),
                           1008, 3, 180, 0)
        cases.append(LabeledCase("normal", "routine", cur, base_neighbors(t),
                                  history_series("A", 12.9, 77.6, {"temp": 30, "hum": 55, "pres": 1008, "wind": 3, "wdir": 180, "rain": 0}, {"temp": 0.1, "hum": 0, "pres": 0, "wind": 0, "wdir": 0, "rain": 0})))

    # 2. localized flood / heavy rain event: humidity up, pressure down, wind up, rain starting
    #    at station A while neighbors stay normal -> should be POSSIBLE_REAL_EVENT, not fault
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        hist = history_series("A", 12.9, 77.6,
                               {"temp": 30, "hum": 60, "pres": 1006, "wind": 3, "wdir": 200, "rain": 0},
                               {"temp": -0.3, "hum": 5, "pres": -0.4, "wind": 1.2, "wdir": 2, "rain": 2})
        last = hist[-1]
        cur = _mk_reading("A", t, 12.9, 77.6, last.temperature_c - 0.3, min(last.humidity_pct + 5, 97),
                           last.pressure_hpa - 0.4, last.wind_speed_ms + 1.2, last.wind_direction_deg + 2,
                           last.rainfall_mm + 3)
        cases.append(LabeledCase("true_event", "flood", cur, base_neighbors(t), hist))

    # 3. cyclone-like event: strong pressure drop + high wind, consistent trend, physically valid
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        hist = history_series("A", 12.9, 77.6,
                               {"temp": 27, "hum": 85, "pres": 985, "wind": 20, "wdir": 220, "rain": 5},
                               {"temp": -0.1, "hum": 0.5, "pres": -0.8, "wind": 1.5, "wdir": 1, "rain": 1})
        last = hist[-1]
        cur = _mk_reading("A", t, 12.9, 77.6, last.temperature_c - 0.1, min(last.humidity_pct + 0.5, 98),
                           last.pressure_hpa - 0.8, last.wind_speed_ms + 1.5, last.wind_direction_deg + 1,
                           last.rainfall_mm + 1)
        cases.append(LabeledCase("true_event", "cyclone", cur, base_neighbors(t), hist))

    # 4. heat wave: temperature high but climbing gradually and consistent with neighbors trending too
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        hist = history_series("A", 12.9, 77.6,
                               {"temp": 42, "hum": 20, "pres": 1002, "wind": 2, "wdir": 90, "rain": 0},
                               {"temp": 0.3, "hum": -0.2, "pres": 0, "wind": 0, "wdir": 0, "rain": 0})
        last = hist[-1]
        cur = _mk_reading("A", t, 12.9, 77.6, last.temperature_c + 0.3, max(last.humidity_pct - 0.2, 10),
                           last.pressure_hpa, last.wind_speed_ms, last.wind_direction_deg, 0)
        neighbors = base_neighbors(t, temp=44, hum=18, pres=1002, wind=2, wdir=90, rain=0)
        cases.append(LabeledCase("true_event", "heat_wave", cur, neighbors, hist))

    # 5. sensor failure: implausible single-reading spike, no corroboration, physically borderline
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        hist = history_series("A", 12.9, 77.6,
                               {"temp": 30, "hum": 55, "pres": 1008, "wind": 3, "wdir": 180, "rain": 0},
                               {"temp": 0, "hum": 0, "pres": 0, "wind": 0, "wdir": 0, "rain": 0})
        cur = _mk_reading("A", t, 12.9, 77.6, 30 + rnd.choice([25, -20]), 55, 1008, 3, 180, 0)
        cases.append(LabeledCase("sensor_fault", "sensor_spike", cur, base_neighbors(t), hist))

    # 6. communication failure -> stale/frozen sensor (flat line for a long time, deviates from neighbors)
    for i in range(n_per_type):
        t = base_t + timedelta(minutes=5 * i)
        frozen = {"temp": 30, "hum": 55, "pres": 1008, "wind": 3, "wdir": 180, "rain": 0}
        hist = [
            _mk_reading("A", base_t - timedelta(minutes=5 * (8 - j)), 12.9, 77.6,
                        frozen["temp"], frozen["hum"], frozen["pres"], frozen["wind"], frozen["wdir"], frozen["rain"])
            for j in range(8)
        ]
        cur = _mk_reading("A", t, 12.9, 77.6, 30, 55, 1008, 3, 180, 0)
        # neighbors have since drifted, so the frozen station now deviates
        neighbors = base_neighbors(t, temp=37, hum=80, pres=1002, wind=8, wdir=200, rain=4)
        cases.append(LabeledCase("sensor_fault", "comm_failure", cur, neighbors, hist))

    return cases
