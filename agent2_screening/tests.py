"""
Basic sanity tests for Agent2.

Run with:  python -m pytest tests/  (or just: python tests/test_agent.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from agent2_screening.pipeline import Agent2
from agent2_screening.schema import StationReading, Coordinates
from agent2_screening.validation import generate_synthetic_cases, run_validation


def mk(station, t, temp, hum, pres, wind, wdir, rain=0, lat=12.9, lon=77.6):
    return StationReading(station, t, Coordinates(lat, lon), temp, hum, pres, wind, wdir, rain)


def test_flood_like_event_not_called_a_fault():
    """The core Phase 5 regression test: a real localized event should NOT
    be labeled LIKELY_SENSOR_FAULT just because it deviates from neighbors."""
    agent = Agent2()
    t = datetime(2026, 6, 1, 10, 0)

    history = [
        mk("A", t - timedelta(minutes=15), 30.5, 65, 1006.5, 4.0, 195, 0),
        mk("A", t - timedelta(minutes=10), 30.2, 72, 1006.0, 5.5, 198, 1),
        mk("A", t - timedelta(minutes=5), 29.9, 82, 1005.3, 7.0, 200, 3),
    ]
    current = mk("A", t, 29.6, 92, 1004.5, 9.0, 202, 6)
    neighbors = [
        mk("B", t, 33.0, 55, 1008.0, 3.0, 180),
        mk("C", t, 34.0, 54, 1008.2, 3.2, 178),
        mk("D", t, 33.5, 56, 1007.9, 2.9, 182),
        mk("E", t, 34.2, 53, 1008.1, 3.1, 179),
    ]

    report = agent.evaluate(current, neighbors, history)
    assert report.spatial.status == "DEVIATION"
    assert report.reasoning.verdict != "LIKELY_SENSOR_FAULT", (
        f"Real event misclassified as fault: {report.reasoning.explanation}"
    )
    print("PASS: flood-like event ->", report.reasoning.verdict, "| score", report.score.total, report.final_label)


def test_sensor_spike_flagged():
    agent = Agent2()
    t = datetime(2026, 6, 1, 10, 0)
    history = [mk("A", t - timedelta(minutes=5 * i), 30, 55, 1008, 3, 180, 0) for i in range(5, 0, -1)]
    current = mk("A", t, 30 + 25, 55, 1008, 3, 180, 0)   # implausible +25C jump
    neighbors = [mk(n, t, 30, 55, 1008, 3, 180, 0) for n in "BCDE"]

    report = agent.evaluate(current, neighbors, history)
    assert report.final_label in ("REVIEW", "SUSPICIOUS")
    print("PASS: sensor spike ->", report.reasoning.verdict, "| score", report.score.total, report.final_label)


def test_normal_reading_is_clean():
    agent = Agent2()
    t = datetime(2026, 6, 1, 10, 0)
    history = [mk("A", t - timedelta(minutes=5 * i), 30, 55, 1008, 3, 180, 0) for i in range(5, 0, -1)]
    current = mk("A", t, 30.3, 56, 1007.8, 3.1, 182, 0)
    neighbors = [mk(n, t, 30, 55, 1008, 3, 180, 0) for n in "BCDE"]

    report = agent.evaluate(current, neighbors, history)
    assert report.final_label == "CLEAN"
    print("PASS: normal reading -> CLEAN")


def test_synthetic_validation_runs():
    agent = Agent2()
    cases = generate_synthetic_cases(n_per_type=10)
    metrics = run_validation(agent, cases)
    print(metrics.summary())
    assert metrics.n_cases == len(cases)
    assert metrics.avg_processing_time_ms < 50  # should be fast, pure Python logic


if __name__ == "__main__":
    test_flood_like_event_not_called_a_fault()
    test_sensor_spike_flagged()
    test_normal_reading_is_clean()
    test_synthetic_validation_runs()
    print("\nAll tests passed.")
