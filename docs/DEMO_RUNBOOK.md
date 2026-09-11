# SkyGuard AI — Screening Demo Runbook

## Start

Terminal 1:

```bash
uvicorn backend_integration.main:app --reload --port 8000
```

Terminal 2:

```bash
streamlit run agent5_dashboard/app.py
```

Optional preflight:

```bash
python backend_integration/preflight.py
```

Optional regression:

```bash
python backend_integration/test_skyguard_integration.py
```

## Demo order

### 1. Normal network

**Message:** "We first establish a trusted network state. No unnecessary alert and no correction."

### 2. Isolated sensor spike

**Message:** "One station jumps while its neighbors remain stable. Spatial + temporal + ML evidence converge on a sensor fault. Only the affected channel is corrected."

### 3. Frozen sensor

**Message:** "The failure signature is different: a channel becomes flat. The system classifies the root cause instead of simply saying anomaly."

### 4. Communication dropout

**Message:** "The telemetry itself disappears. This becomes a critical data-integrity event and a last-trustworthy estimate is surfaced."

### 5. Calibration drift

**Message:** "The station slowly separates from the network baseline. We surface maintenance before the sensor becomes unusable."

### 6. Genuine extreme weather — final shot

**Message:** "Now the dangerous case. An unusual reading is not necessarily bad data. Multiple stations move together over multiple hours, so the genuine-event shield protects the observation and blocks correction."

Finish with:

> **Anomaly does not equal sensor fault. The same system that repairs a broken sensor knows when not to repair the weather.**

## Judge defense

**Why not just LSTM?**

Because an LSTM can detect unusual temporal behavior but cannot by itself determine whether the cause is a faulty instrument or a real regional event. SkyGuard combines temporal ML with physical, spatial and explicit safety evidence.

**Why multiple agents?**

Each agent has an auditable responsibility. A single model is not given unchecked authority over a high-risk correction decision.

**How do you explain a decision?**

Agent 4 exposes SHAP features for the tree-based decision layer and converts the result into a human-readable narrative. The dashboard also shows the underlying evidence and safety gate.

**How do you avoid corrupting genuine weather?**

The regional event shield looks for multi-station, multi-hour corroboration and physical coherence. When that evidence is strong, automatic correction is blocked.

**How is correction performed?**

Agent 5 combines temporal and healthy-neighbor spatial estimates, excludes unhealthy neighbors, applies safety bounds and replaces only the diagnosed channel where possible.

**Can this scale?**

The screening build uses in-memory state for deterministic demonstration. The API boundary is already separated from the state layer, so Redis/TimescaleDB can replace the prototype store without changing the agent contracts.
