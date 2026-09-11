# SkyGuard AI — SIH26073

**Agentic Genesis · SkyGuard AI**  
Self-aware, self-healing anomaly detection for Automatic Weather Stations.

## What is now integrated

`Agent 1 → Agent 2 → Agent 3 → Evidence Fusion → Agent 4 → Agent 5`

- **Agent 1:** historical AWS stream with temperature, pressure, humidity.
- **Agent 2:** real physical + spatial + temporal consistency screening.
- **Agent 3:** committed LSTM autoencoder (12-hour windows) when TensorFlow/model artifacts are available.
- **Hybrid Evidence Fusion:** auditable hard checks for dropout, frozen channels, isolated spikes, calibration drift, and regional event corroboration.
- **Agent 4:** real RandomForest score gate, SHAP explanations, safety override, and degradation tracker.
- **Agent 5:** real temporal/spatial correction with same-timestamp healthy neighbors and channel-preserving correction.
- **Dashboard:** screening-round Command Center + deterministic replay scenarios.

## The differentiator

**Anomaly ≠ sensor fault.**

SkyGuard does not blindly correct every unusual reading. It first asks whether the pattern is:
1. an isolated sensor failure,
2. a degrading sensor,
3. a communication failure, or
4. a genuine regional weather event.

A genuine-event shield protects a spatially corroborated, multi-hour weather transition from automatic correction.

## Run

From the repository root:

```bash
pip install -r requirements.txt
uvicorn backend_integration.main:app --reload --port 8000
```

Dashboard:

```bash
streamlit run agent5_dashboard/app.py
```

Release preflight + full screening gate:

```bash
python backend_integration/preflight.py
python backend_integration/final_release_check.py
```

The release gate intentionally fails closed if Agent 2, the real Agent-3 LSTM-AE, or Agent 4 SHAP/safety runtime is unavailable.

CLI replay (does not require Streamlit):

```bash
python backend_integration/run_demo.py
```

## Screening scenarios

The replay uses the committed Agent-1 historical data rather than random mock values:

- NORMAL NETWORK
- ISOLATED SENSOR SPIKE
- FROZEN SENSOR
- COMMUNICATION DROPOUT
- CALIBRATION DRIFT
- GENUINE EXTREME WEATHER (Tauktae window)

## Important model honesty

The committed Agent-3 LSTM was evaluated by its own report and is **not** treated as a perfect detector. The system's design deliberately combines its anomaly score with independent physical, spatial, temporal and rule-based evidence. If TensorFlow is not installed, the backend labels the fallback explicitly instead of pretending it is the LSTM.

## API

- `GET /health`
- `POST /process`
- `POST /process_batch`
- `POST /reset`
- `GET /demo/scenarios`
- `GET /demo/scenario/{scenario}`

## Architecture

```text
AWS stream
   │
   ▼
[Agent 1] Ingestion
   │
   ▼
[Agent 2] Physics + Spatial + Temporal
   │
   ▼
[Agent 3] LSTM Autoencoder
   │
   ▼
[Evidence Fusion]
   ├── dropout / frozen / spike / drift signatures
   └── regional genuine-event shield
   │
   ▼
[Agent 4] RF gate + SHAP + safety gate + degradation
   │
   ▼
[Agent 5] correction + severity
   │
   ├──────────────► FastAPI
   └──────────────► Streamlit Command Center
```

## Judge-first documentation

- [`docs/JUDGE_SCORECARD.md`](docs/JUDGE_SCORECARD.md) — maps the implementation to the SIH26073 evaluation weights.
- [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) — exact six-scenario screening sequence and judge answers.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — end-to-end architecture and safety principle.

## Screening build principle

The Command Center is deliberately designed around one question: **"Is this anomaly a faulty sensor, or is it the weather?"** The dashboard therefore makes the final decision, evidence, correction action and genuine-event protection visible before exposing lower-level implementation details.
