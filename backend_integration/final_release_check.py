"""One-command SIH26073 release gate.

Run from repository root:
    python backend_integration/final_release_check.py

This is intentionally stricter than the demo runner: it fails if a required
real agent is unavailable or if the six judge scenarios violate key safety
invariants.
"""
from __future__ import annotations
import math
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend_integration")
for path in (ROOT, BACKEND):
    if path not in sys.path:
        sys.path.insert(0, path)

import main
from demo_runner import run_scenario

EXPECTED = {
    "normal": "none",
    "spike": "sensor_spike",
    "frozen": "sensor_stuck",
    "dropout": "sensor_dropout",
    "drift": "sensor_drift",
    "genuine_event": "genuine_event",
}


def fail(message: str):
    raise AssertionError(message)


def check_runtime():
    status = main.runtime_status()
    print("RUNTIME")
    print(status)
    if not status["ready"]:
        fail("Runtime is not release-ready: all required real agents must be available.")
    if status["agent3"]["engine"] != "LSTM_AUTOENCODER":
        fail(f"Agent 3 is not using the real LSTM-AE: {status['agent3']}")
    if status["agent4"]["engine"] != "A4_GATE_SHAP":
        fail(f"Agent 4 is not using the real SHAP safety path: {status['agent4']}")


def check_scenarios():
    print("\nSCENARIOS")

    for name, expected in EXPECTED.items():

        # Run every scenario in a fresh Python process.
        # This prevents TensorFlow/model state from leaking
        # between deterministic replay scenarios.
        import subprocess
        import json

        code = f"""
import json
from backend_integration.demo_runner import run_scenario

_, results, selected = run_scenario("{name}")

out = {{
    "anomaly_type": selected.anomaly_type,
    "confidence_score": float(selected.confidence_score),
    "sensor_health_status": selected.sensor_health_status,
    "engine": selected.engine,
    "corrected_value": (
        selected.corrected_value.model_dump()
        if selected.corrected_value is not None
        else None
    ),
    "agent2_detail": selected.agent2_detail,
    "explanation": selected.explanation or {{}},
    "pressure_hpa": selected.pressure_hpa,
    "humidity_pct": selected.humidity_pct,
    "drift_count": sum(
        1 for r in results
        if r.anomaly_type == "sensor_drift"
    ),
}}

print("SKYGUARD_RESULT=" + json.dumps(out, default=str))
"""

        env = dict(os.environ)
        env["PYTHONPATH"] = (
            ROOT
            + os.pathsep
            + BACKEND
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if proc.returncode != 0:
            fail(
                f"{name}: isolated scenario failed:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )

        marker = "SKYGUARD_RESULT="
        line = next(
            (
                x
                for x in proc.stdout.splitlines()
                if x.startswith(marker)
            ),
            None,
        )

        if line is None:
            fail(
                f"{name}: no scenario result returned.\n"
                f"{proc.stdout}\n{proc.stderr}"
            )

        selected = json.loads(line[len(marker):])

        # -----------------------------
        # EXPECTED CLASSIFICATION
        # -----------------------------

        if selected["anomaly_type"] != expected:
            fail(
                f"{name}: expected {expected}, "
                f"got {selected['anomaly_type']}"
            )

        # Agent 2 must be real
        if (
            not selected["agent2_detail"]
            or "unavailable"
            in str(selected["agent2_detail"]).lower()
        ):
            fail(f"{name}: Agent 2 unavailable/fake evidence")

        # Agent 3 must be real LSTM
        if selected["engine"] != "LSTM_AUTOENCODER":
            fail(
                f"{name}: expected real LSTM-AE, "
                f"got {selected['engine']}"
            )

        exp = selected["explanation"] or {}
        final_verdict = exp.get("final_verdict")

        # -----------------------------
        # SENSOR FAULTS
        # -----------------------------

        if expected in {
            "sensor_spike",
            "sensor_stuck",
            "sensor_drift",
            "sensor_dropout",
        }:

            if selected["sensor_health_status"] not in {
                "amber",
                "red",
            }:
                fail(
                    f"{name}: fault has invalid health state "
                    f"{selected['sensor_health_status']}"
                )

            if final_verdict != "sensor_fault":
                fail(
                    f"{name}: explanation "
                    f"final_verdict={final_verdict}"
                )

            if selected["corrected_value"] is None:
                fail(f"{name}: correction missing")

        # -----------------------------
        # GENUINE WEATHER EVENT
        # -----------------------------

        elif expected == "genuine_event":

            if selected["corrected_value"] is not None:
                fail(
                    "genuine_event: correction must be blocked"
                )

            if selected["sensor_health_status"] != "green":
                fail(
                    "genuine_event: health should remain "
                    "protected/green, got "
                    f"{selected['sensor_health_status']}"
                )

            if final_verdict != "genuine_event":
                fail(
                    f"genuine_event: "
                    f"final_verdict={final_verdict}"
                )

            sig = exp.get("evidence_signature", {})

            if float(
                sig.get("regional_event_index", 0)
            ) < 0.60:
                fail(
                    "genuine_event: regional event shield "
                    "evidence too weak"
                )

        # -----------------------------
        # NORMAL
        # -----------------------------

        else:

            if selected["corrected_value"] is not None:
                fail(
                    "normal: correction must be absent"
                )

            if selected["sensor_health_status"] != "green":
                fail(
                    "normal: unexpected health state "
                    f"{selected['sensor_health_status']}"
                )

            if final_verdict != "nominal":
                fail(
                    f"normal: final_verdict={final_verdict}"
                )

        # -----------------------------
        # DRIFT-SPECIFIC SAFETY CHECK
        # -----------------------------

        if name == "drift":

            corrected = selected["corrected_value"]

            if corrected["pressure_hpa"] == selected["pressure_hpa"]:
                fail(
                    "drift: pressure channel was not corrected"
                )

            if corrected["humidity_pct"] != selected["humidity_pct"]:
                fail(
                    "drift: humidity channel was "
                    "incorrectly overwritten"
                )

            if selected["drift_count"] != 1:
                fail(
                    "drift: expected exactly one drift station"
                )

        print(
            f"PASS {name}: "
            f"{selected['anomaly_type']} / "
            f"{selected['confidence_score']:.0%} / "
            f"{selected['sensor_health_status']} / "
            f"{selected['engine']}"
        )

def check_correction_benchmark():
    path = os.path.join(ROOT, "agent5_dashboard", "correction_accuracy_results.csv")
    df = pd.read_csv(path)
    if len(df) != 5463:
        fail(f"Correction benchmark row count changed: {len(df)}")
    tolerances = {"temperature_c": 2.0, "pressure_hpa": 3.0, "humidity_pct": 8.0}
    within = []
    for field, tol in tolerances.items():
        within.append(float((df[f"{field}_abs_err"] <= tol).mean()))
    overall = sum(within) / len(within)
    print(f"\nCORRECTION BENCHMARK: {len(df)}/{len(df)} evaluable; avg within tolerance={overall:.1%}")
    if not math.isclose(overall, 0.825, abs_tol=0.01):
        fail(f"Correction benchmark headline changed unexpectedly: {overall:.3%}")


if __name__ == "__main__":
    print("=" * 70)
    print("SKYGUARD AI — SIH26073 FINAL RELEASE GATE")
    print("=" * 70)
    check_runtime()
    check_scenarios()
    check_correction_benchmark()
    print("\nALL FINAL RELEASE CHECKS PASSED")
