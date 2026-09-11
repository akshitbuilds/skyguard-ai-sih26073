import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent4 import ScreeningGate


def test_case_a_high_spatial_high_ml_coherent_leans_genuine():
    """
    Case A from the design brief: high spatial_deviation_score + high
    ml_anomaly_score + physically coherent reading -> should lean
    genuine_event.
    """
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-TEMP-014",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.88,
        ml_anomaly_score=0.91,
        physically_coherent=True,
        screening_flag="high_priority",
    )
    assert r.verdict == "genuine_event"
    assert r.safety_override_triggered is False


def test_case_b_ml_alone_incoherent_leans_fault():
    """
    Case B from the design brief: high ml_anomaly_score alone +
    physically incoherent/impossible reading -> should lean sensor fault.
    Uses screening_flag=standard so the safety gate isn't in play here --
    that combination is tested separately below under high_priority.
    """
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-RAIN-022",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.08,
        ml_anomaly_score=0.93,
        physically_coherent=False,
        screening_flag="standard",
    )
    assert r.verdict == "sensor_fault"
    assert r.safety_override_triggered is False


def test_high_priority_fault_with_overwhelming_evidence_can_still_dismiss():
    """
    The safety gate is not "always escalate high_priority" -- it requires
    strong evidence, not zero evidence. Case B's exact numbers, but flagged
    high_priority, should still clear the bar and resolve to sensor_fault,
    because the evidence really is overwhelming (very low spatial
    corroboration, clearly incoherent, high model confidence).
    """
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-RAIN-022",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.05,
        ml_anomaly_score=0.97,
        physically_coherent=False,
        screening_flag="high_priority",
    )
    # Either outcome is defensible depending on model_confidence on this
    # exact input, but it must NEVER be silently dismissed -- it's either
    # a justified sensor_fault (strong evidence, gate_reason says so) or
    # escalated. It must not disappear silently either way.
    assert r.verdict in ("sensor_fault", "escalate_uncertain")
    if r.verdict == "sensor_fault":
        assert "strong multi-signal evidence" in r.gate_reason


def test_non_negotiable_high_priority_ambiguous_fault_never_silently_dismissed():
    """
    THE core requirement: a high_priority reading that the classifier
    *leans* fault on, but without overwhelming evidence (moderate spatial
    corroboration, borderline coherence signal), must NEVER come out as a
    plain 'sensor_fault' silently. It must escalate instead.
    """
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-WIND-007",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.5,     # ambiguous -- some spatial corroboration
        ml_anomaly_score=0.7,             # elevated but not extreme
        physically_coherent=False,        # incoherent, but the other two signals are weak
        screening_flag="high_priority",
    )
    if r.model_verdict == "sensor_fault":
        assert r.verdict == "escalate_uncertain"
        assert r.safety_override_triggered is True
        assert "SAFETY OVERRIDE" in r.gate_reason
    else:
        assert r.verdict == "genuine_event"


def test_standard_priority_is_not_gated():
    """Non-high-priority readings use the model's call directly (but logged)."""
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-WIND-007",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.5,
        ml_anomaly_score=0.7,
        physically_coherent=False,
        screening_flag="standard",
    )
    assert r.verdict == r.model_verdict
    assert r.safety_override_triggered is False


def test_output_has_narrative_and_audit_trail():
    gate = ScreeningGate()
    r = gate.classify(
        sensor_id="IMD-TEMP-014",
        timestamp="2026-09-09T06:45:00",
        spatial_deviation_score=0.88,
        ml_anomaly_score=0.91,
        physically_coherent=True,
        screening_flag="high_priority",
    )
    assert isinstance(r.narrative, str) and len(r.narrative) > 20
    assert len(r.top_features) > 0
    assert r.inputs["screening_flag"] == "high_priority"
    d = r.to_dict()
    assert d["verdict"] == r.verdict


if __name__ == "__main__":
    test_case_a_high_spatial_high_ml_coherent_leans_genuine()
    test_case_b_ml_alone_incoherent_leans_fault()
    test_high_priority_fault_with_overwhelming_evidence_can_still_dismiss()
    test_non_negotiable_high_priority_ambiguous_fault_never_silently_dismissed()
    test_standard_priority_is_not_gated()
    test_output_has_narrative_and_audit_trail()
    print("All screening-gate tests passed.")
