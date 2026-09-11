"""
End-to-end demo of Agent 4.

Simulates a stream of readings for a few sensors -- one healthy, one that
spikes once, one that freezes, one with comms dropouts, one drifting, and
one experiencing a genuine extreme event -- and shows what Agent 4 outputs
for each, plus a sensor health report at the end.

Run: python demo.py
"""
import json
import numpy as np
from datetime import datetime, timedelta

from agent4 import Agent4

RNG = np.random.default_rng(7)
EXPECTED_RANGE = (-5.0, 48.0)


def make_series(n, base=25.0, amp=3.0, noise=0.3):
    t = np.arange(n)
    return base + amp * np.sin(2 * np.pi * t / 24.0) + RNG.normal(0, noise, n)


def run_case(agent, sensor_id, series, missing_mask=None, gap_steps=1, label_hint=""):
    now = datetime(2026, 9, 9, 6, 0, 0)
    window = series[:-1].tolist()
    value = float(series[-1])
    ts = (now + timedelta(minutes=5 * len(series))).isoformat()

    result = agent.explain_flag(
        sensor_id=sensor_id,
        timestamp=ts,
        value=value,
        window=series.tolist(),
        expected_range=EXPECTED_RANGE,
        missing_mask=missing_mask,
        gap_steps=gap_steps,
        prior_window=make_series(8).tolist(),
        upstream_model_score=round(float(RNG.uniform(0.7, 0.99)), 3),
    )
    print(f"\n=== {sensor_id} ({label_hint}) ===")
    print(json.dumps(result.to_dict(), indent=2))
    return result


def main():
    agent = Agent4()

    # 1. Sensor spike
    s = make_series(9)
    s[-1] += 15
    run_case(agent, "IMD-TEMP-001", s, label_hint="expected: SENSOR_SPIKE")

    # 2. Frozen value
    s = make_series(9)
    s[-5:] = s[-6]
    run_case(agent, "IMD-TEMP-002", s, label_hint="expected: FROZEN_VALUE")

    # 3. Comms error (dropouts / fill values)
    s = make_series(9)
    mask = [False] * 9
    s[-3], s[-2] = -999.0, -999.0
    mask[-3], mask[-2] = True, True
    run_case(agent, "IMD-TEMP-003", s, missing_mask=mask, gap_steps=4, label_hint="expected: COMMS_ERROR")

    # 4. Drift
    s = make_series(9)
    s[-6:] += np.linspace(0, 6, 6)
    run_case(agent, "IMD-TEMP-004", s, label_hint="expected: DRIFT")

    # 5. Genuine extreme event
    s = make_series(9, base=25.0)
    s[-6:] += np.linspace(0, 14, 6) + RNG.normal(0, 0.4, 6)
    run_case(agent, "IMD-TEMP-005", s, label_hint="expected: GENUINE_EXTREME")

    # 6. Simulate sensor 002 repeatedly freezing over time -> should trend toward DEGRADING/CRITICAL
    print("\n=== IMD-TEMP-002 repeated freezes over time (degradation tracking) ===")
    for i in range(12):
        s = make_series(9)
        if i % 2 == 0:
            s[-5:] = s[-6]  # keeps freezing
        r = run_case(agent, "IMD-TEMP-002", s, label_hint=f"iteration {i}")

    print("\n=== Sensor health report ===")
    print(json.dumps(agent.sensor_health_report(), indent=2))


if __name__ == "__main__":
    main()
