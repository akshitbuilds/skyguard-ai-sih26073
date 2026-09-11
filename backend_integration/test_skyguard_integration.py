
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_runner import run_scenario
import main

EXPECTED={
 "normal":"none","spike":"sensor_spike","frozen":"sensor_stuck",
 "dropout":"sensor_dropout","drift":"sensor_drift","genuine_event":"genuine_event",
}
for name,expected in EXPECTED.items():
    cfg,results,selected=run_scenario(name)
    assert selected.anomaly_type==expected,(name,selected.anomaly_type,expected)
    if name=="genuine_event":
        assert selected.corrected_value is None
        assert selected.explanation["evidence_signature"]["regional_event_index"]>=0.60
    if name in ("spike","frozen","drift","dropout"):
        assert selected.corrected_value is not None
        assert selected.sensor_health_status in ("amber", "red"), (name, selected.sensor_health_status)
        assert selected.explanation.get("final_verdict") == "sensor_fault", (name, selected.explanation)
        assert selected.explanation.get("raw_gate_verdict") is not None
    if name == "drift":
        # The injected Mahuva fault is a pressure calibration drift; the
        # correction layer must repair the diagnosed channel rather than a
        # merely correlated/noisy channel.
        assert selected.corrected_value.pressure_hpa != selected.pressure_hpa
        assert selected.corrected_value.humidity_pct == selected.humidity_pct
    if name == "genuine_event":
        assert selected.sensor_health_status == "green"
        assert selected.explanation.get("final_verdict") == "genuine_event"
        if main.runtime_status()["agent3"]["available"]:
            assert selected.engine == "LSTM_AUTOENCODER", selected.engine
        else:
            print("  NOTE: Agent 3 LSTM test skipped because TensorFlow/model runtime is unavailable in this environment.")
    if name == "normal":
        assert selected.sensor_health_status == "green"
        assert selected.explanation.get("final_verdict") == "nominal"
    assert selected.agent2_detail and "unavailable" not in str(selected.agent2_detail.get("explanation","")).lower()
    print(f"PASS {name}: {selected.anomaly_type} / {selected.confidence_score:.0%} / {selected.alert_severity} / {selected.sensor_health_status} / {selected.engine}")
print("ALL SCREENING SCENARIOS PASSED")
