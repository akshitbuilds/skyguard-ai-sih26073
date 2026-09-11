
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_runner import SCENARIOS, run_scenario
for name in sys.argv[1:] or ["normal","spike","frozen","dropout","drift","genuine_event"]:
    cfg, results, selected=run_scenario(name)
    print("\n" + "="*76)
    print(cfg["label"], "|", cfg["timestamp"])
    print("="*76)
    print("Selected:",selected.station_id, selected.anomaly_type,
          f"confidence={selected.confidence_score:.0%}",
          "severity="+str(selected.alert_severity))
    print("ML:",selected.ml_anomaly_score,"screening:",selected.screening_flag)
    print("Reason:",selected.explanation.get("root_cause_narrative",""))
    print("Corrected:",selected.corrected_value.model_dump() if selected.corrected_value else "BLOCKED / NONE")
    sig=selected.explanation.get("evidence_signature",{})
    print("Event corroboration:",sig.get("event_corroboration"),
          "regional index:",sig.get("regional_event_index"))
