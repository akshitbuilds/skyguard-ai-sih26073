# SkyGuard AI — Final Hardening Changelog

## Backend hardening

- Agent 2 imports are now independent of the launch directory.
- Final mode fails closed if Agent 2 or Agent 4 is unavailable; no synthetic screening evidence is silently substituted.
- Agent 3 preserves the real `LSTM_AUTOENCODER` engine field through the Pydantic contract.
- If a loaded LSTM model fails during inference, the adapter raises an explicit error instead of silently switching detectors.
- Fusion drift detection now requires persistent, monotonic residual bias with sign consistency, reducing regional-weather false positives.
- Drift correction selects the channel supported by the drift residual evidence instead of using a raw z-score from a correlated channel.
- Agent 4 exposes `final_verdict` separately from `raw_gate_verdict` so model disagreement is auditable without contradicting the final decision.
- Sensor health has deterministic minimum states for confirmed faults: drift/spike/stuck → at least WATCH; dropout → CRITICAL.
- Defensive correction handles missing/non-finite historical values per field.
- Release preflight and a single `final_release_check.py` gate were added.
- Unused stub backend files were removed.
- scikit-learn is pinned to 1.8.0 to match the committed Agent 4 artifacts.

## Frontend hardening

- Judge-first dark command-center UI retained and improved.
- Scenario selector has explicit dark BaseWeb styling.
- Runtime integrity strip shows actual Agent 2/3/4/5 runtime status and measured replay latency.
- LSTM stage is displayed from the backend runtime contract rather than a hardcoded label.
- Self-healing panel explicitly shows which channels changed.
- Genuine-event panel explicitly shows correction BLOCKED.
- Deployment panel avoids unmeasured energy/power claims.
- Removed stale mock-data dashboard documentation and generator from the final screening build.

## Verification in the build environment

- 14 Agent 2/Agent 4 unit tests passed.
- 6 integrated screening scenarios passed with the real Agent 2 + Agent 4 path.
- Drift scenario now produces exactly one drift station: AWS_MAHUVA.
- Drift health is amber/WATCH rather than STABLE.
- Drift final explanation verdict is `sensor_fault` and preserves raw gate verdict separately.
- Drift correction changes pressure while preserving temperature/humidity.
- Missing-value correction regression no longer crashes.
- Full LSTM runtime verification must be executed on the final Windows environment because this build environment does not include TensorFlow. The user's final environment must pass `python backend_integration/preflight.py` and `python backend_integration/final_release_check.py` before the GitHub push.
