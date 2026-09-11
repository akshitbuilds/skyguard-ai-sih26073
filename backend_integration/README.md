# Backend Integration — SkyGuard AI

This directory contains the integrated SIH26073 orchestration layer.

## Runtime pipeline

1. Agent 1 — incoming AWS reading
2. Agent 2 — physical, spatial and temporal consistency screening
3. Agent 3 — committed LSTM autoencoder on a 12-hour station window
4. Evidence Fusion — deterministic signatures + regional genuine-event shield
5. Agent 4 — score gate, SHAP explanation, safety gate and degradation tracking
6. Agent 5 — correction and alert severity

The implementation deliberately keeps the agents modular. The fusion layer is
an orchestration/safety layer; it does not replace Agent 2, Agent 3, Agent 4,
or Agent 5.

## Run

```bash
uvicorn main:app --reload --port 8000
```

or from the repository root:

```bash
uvicorn backend_integration.main:app --reload --port 8000
```

## Screening replay

```bash
python run_demo.py
python test_skyguard_integration.py
```

The replay is deterministic and uses the real Agent-1 historical stream.
For a final release, run `python final_release_check.py`; it fails closed if a required real agent is unavailable and verifies the six judge scenarios plus correction invariants.

## Agent 3

The release build requires TensorFlow and the committed LSTM artifacts. The adapter can expose an explicitly labeled deterministic fallback for development/warm-up, but the release preflight fails closed unless the real `LSTM_AUTOENCODER` is available.

## Important integration detail

Agent 2's `spatial_deviation_score` means "how much this station differs from
neighbors." Agent 4's original score model expects a corroboration polarity.
`agent4_adapter.py` performs the explicit `1 - deviation` polarity mapping
before invoking Agent 4.

## Same-timestamp processing

`POST /process_batch` and the scenario replay process a network tick
synchronously. This prevents accidental comparison of one station's current
reading against another station's previous tick.

## Endpoints

- `GET /health`
- `POST /process`
- `POST /process_batch`
- `POST /reset`
- `GET /demo/scenarios`
- `GET /demo/scenario/{scenario}`
