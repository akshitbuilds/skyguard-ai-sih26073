import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from agent4 import Agent4

RNG = np.random.default_rng(0)


def make_series(n, base=25.0, amp=3.0, noise=0.3):
    t = np.arange(n)
    return base + amp * np.sin(2 * np.pi * t / 24.0) + RNG.normal(0, noise, n)


def test_spike_detected():
    agent = Agent4()
    s = make_series(9)
    s[-1] += 18
    r = agent.explain_flag("T1", "2026-09-09T00:00:00", float(s[-1]), s.tolist())
    assert r.root_cause == "SENSOR_SPIKE"
    assert 0.0 <= r.confidence <= 1.0
    assert len(r.top_features) > 0


def test_frozen_detected():
    agent = Agent4()
    s = make_series(9)
    s[-6:] = s[-7]
    r = agent.explain_flag("T2", "2026-09-09T00:00:00", float(s[-1]), s.tolist())
    assert r.root_cause == "FROZEN_VALUE"


def test_output_has_narrative_and_degradation():
    agent = Agent4()
    s = make_series(9)
    s[-1] += 15
    r = agent.explain_flag("T3", "2026-09-09T00:00:00", float(s[-1]), s.tolist())
    assert isinstance(r.narrative, str) and len(r.narrative) > 20
    assert r.degradation is not None
    assert r.degradation["sensor_id"] == "T3"


def test_degradation_tracker_flags_repeat_freezes():
    agent = Agent4()
    for i in range(10):
        s = make_series(9)
        s[-5:] = s[-6]  # freeze every time
        agent.explain_flag("T4", f"2026-09-09T00:0{i}:00", float(s[-1]), s.tolist())
    status = agent.degradation_tracker.get_status("T4")
    assert status["status"] in ("WATCH", "DEGRADING", "CRITICAL")


if __name__ == "__main__":
    test_spike_detected()
    test_frozen_detected()
    test_output_has_narrative_and_degradation()
    test_degradation_tracker_flags_repeat_freezes()
    print("All tests passed.")
