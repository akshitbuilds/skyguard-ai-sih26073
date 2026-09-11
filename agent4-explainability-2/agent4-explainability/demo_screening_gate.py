"""
End-to-end demo of the score-based genuine_event / sensor_fault
ScreeningGate, using constructed mock Agent 2 / Agent 3 score inputs
(their real fields aren't wired in yet -- see README "Integration
contract with Agent 2 & Agent 3").

Run: python demo_screening_gate.py
"""
import json
from agent4 import ScreeningGate

CASES = [
    dict(
        sensor_id="IMD-TEMP-014",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.88,
        ml_anomaly_score=0.91,
        physically_coherent=True,
        screening_flag="high_priority",
        label_hint="Case A: high spatial + high ML + coherent -> expect genuine_event",
    ),
    dict(
        sensor_id="IMD-RAIN-022",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.08,
        ml_anomaly_score=0.93,
        physically_coherent=False,
        screening_flag="standard",
        label_hint="Case B: high ML alone + incoherent (standard priority) -> expect sensor_fault",
    ),
    dict(
        sensor_id="IMD-RAIN-022",
        timestamp="2026-09-09T07:00:00",
        spatial_deviation_score=0.05,
        ml_anomaly_score=0.97,
        physically_coherent=False,
        screening_flag="high_priority",
        label_hint="Case B, but high_priority + overwhelming evidence -> gate should still allow fault dismissal",
    ),
    dict(
        sensor_id="IMD-WIND-007",
        timestamp="2026-09-09T07:15:00",
        spatial_deviation_score=0.5,
        ml_anomaly_score=0.7,
        physically_coherent=False,
        screening_flag="high_priority",
        label_hint="Non-negotiable case: high_priority + ambiguous/moderate fault evidence -> must NOT silently dismiss",
    ),
    dict(
        sensor_id="IMD-WIND-007",
        timestamp="2026-09-09T07:15:00",
        spatial_deviation_score=0.5,
        ml_anomaly_score=0.7,
        physically_coherent=False,
        screening_flag="standard",
        label_hint="Same ambiguous signal, but standard priority -> model call used directly (still logged)",
    ),
]


def main():
    gate = ScreeningGate()
    results = []
    for case in CASES:
        hint = case.pop("label_hint")
        r = gate.classify(**case)
        print(f"\n=== {case['sensor_id']} | {case['screening_flag']} | {hint} ===")
        print(json.dumps(r.to_dict(), indent=2))
        results.append(r.to_dict())
    return results


if __name__ == "__main__":
    main()
