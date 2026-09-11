# SkyGuard AI — Screening Command Center

Team **Agentic Genesis** · Smart India Hackathon 2026 · **SIH26073**

This dashboard is the operator-facing end of the integrated five-agent pipeline:

**Agent 1 → Agent 2 → Agent 3 LSTM-AE → Evidence Fusion → Agent 4 SHAP/Safety → Agent 5 Correction**

## Run

From the repository root:

```bash
pip install -r requirements.txt
python backend_integration/preflight.py
streamlit run agent5_dashboard/app.py
```

The screening dashboard intentionally uses the committed Agent-1 historical stream and deterministic judge scenarios so every demo decision is reproducible and auditable.

## Dashboard responsibilities

- Final diagnosis, confidence and severity
- Runtime integrity of Agents 2–5
- Network station health and digital twin
- Physical, spatial, temporal/ML and event-shield evidence
- SHAP-backed root-cause explanation
- Raw → corrected comparison
- Genuine-event protection with correction blocked
- Validation and deployment architecture

The dashboard does **not** generate fake anomaly scores or replace backend decisions with UI rules.

## Correction behavior

`backend_integration/correction.py` performs defensive temporal + spatial estimation and handles missing/non-finite historical values safely. The caller passes the diagnosed fault field so a single-channel fault does not overwrite healthy channels.

`genuine_event` is never passed to the correction estimator.

## Production boundary

The screening build uses an in-memory station state and replay data. The FastAPI boundary is intentionally separated from the state implementation so a production deployment can replace the state layer with Redis/TimescaleDB or another time-series store without changing the agent contracts.
