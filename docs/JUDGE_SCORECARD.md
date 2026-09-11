# SkyGuard AI — SIH26073 Judge Scorecard

SIH26073 evaluates the solution on the following published criteria:

| Criterion | Weight | What SkyGuard demonstrates |
|---|---:|---|
| Innovation & Novelty | 25% | Neuro-symbolic evidence fusion, genuine-event safety shield, multi-agent decision trace, self-healing correction, degradation monitoring |
| Detection Accuracy | 20% | Agent 2 consistency checks + Agent 3 LSTM-AE + deterministic fault signatures + event corroboration |
| Real-Time Capability | 15% | FastAPI streaming endpoint, in-memory station state, per-tick batch processing |
| Explainability | 10% | Agent 4 SHAP + human-readable root-cause narrative + visible evidence/gate reasoning |
| Scalability | 10% | Station-independent contracts, batch endpoint, replaceable state layer, stateless API boundary |
| Practical Deployability | 10% | FastAPI service, Streamlit command center, requirements, deterministic replay, production state-store seam |
| Visualization / UI | 5% | Network digital twin, station health, alert severity, decision trace, evidence and correction views |
| Energy Efficiency | 5% | Edge-ready architecture; optional edge inference path is documented without claiming an unbuilt ESP32 deployment |

## The central judge question

**How do you distinguish a genuine weather event from a faulty sensor when both can look anomalous?**

SkyGuard answers with independent evidence:

1. **Temporal evidence** — does the observation violate the station's learned sequence pattern?
2. **Spatial evidence** — do neighboring stations disagree with the observation?
3. **Physical evidence** — are temperature, pressure and humidity jointly coherent?
4. **Fault signatures** — spike, frozen channel, dropout or persistent drift.
5. **Regional event evidence** — do multiple stations move together over multiple hours?
6. **Safety policy** — if the regional pattern is corroborated, do not automatically overwrite the observation.

## The 30-second judge demonstration

1. Run **Isolated Sensor Spike** → fault identified → correction approved.
2. Run **Genuine Extreme Weather** → regional corroboration → correction blocked.
3. Say: **"The same system that repairs a broken sensor knows when not to repair the weather."**

## Claims discipline

The repository deliberately avoids claiming universal accuracy. The committed Agent-3 model is imperfect on its own; SkyGuard therefore uses multiple independent evidence layers and explicit safety rules rather than allowing a single model to make the highest-risk decision.
