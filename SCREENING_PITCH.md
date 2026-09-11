# SkyGuard AI — Screening Round Runbook

## 45-second opening

> "The hardest part of weather-station anomaly detection is not finding unusual data. It is deciding whether an unusual signal is a broken sensor or the beginning of a real disaster. SkyGuard AI solves that distinction with five cooperating agents."

## Live sequence

### 1. NORMAL
Show the network with all stations green/nominal.

Say:
> "First, the system establishes a normal network baseline. No alert, no correction."

### 2. SENSOR SPIKE
Run **ISOLATED SENSOR SPIKE**.

Say:
> "Veraval suddenly reports a large humidity jump. Agent 2 sees the spatial mismatch, Agent 3 raises the temporal anomaly score, and Evidence Fusion confirms it is isolated. Agent 4 explains the decision with SHAP evidence. Agent 5 repairs only the diagnosed channel."

Point at:
- raw vs corrected
- confidence
- spatial deviation
- ML anomaly score
- explanation

### 3. FROZEN SENSOR
Run **FROZEN SENSOR**.

Say:
> "This is a different failure mode: the value stops changing. SkyGuard detects the flatline streak and identifies a stuck channel rather than calling it a spike."

### 4. COMMUNICATION DROPOUT
Run **COMMUNICATION DROPOUT**.

Say:
> "Now the telemetry disappears. This is a communication failure, so the severity becomes critical and an estimated value is surfaced for downstream systems."

### 5. DRIFT
Run **CALIBRATION DRIFT**.

Say:
> "This is where we go beyond anomaly detection. A slow spatial residual trend indicates calibration drift, so the system can recommend maintenance before complete failure."

### 6. GENUINE EVENT — THE MONEY SHOT
Run **GENUINE EXTREME WEATHER**.

Say:
> "Now the dangerous case. The reading is unusual, but the surrounding network moves with it over multiple hours. SkyGuard activates the genuine-event shield. It trusts the weather signal and BLOCKS automatic correction."

Then say:
> "That is our key principle: anomaly does not equal sensor fault."

## Judge questions

**Why multiple agents?**  
Each agent has one auditable responsibility: ingestion, consistency screening, temporal ML detection, explanation/safety, and correction/alerting. No single model gets unchecked authority.

**Why not just use LSTM?**  
An LSTM can identify unusual temporal behavior, but unusual weather is not necessarily faulty data. Spatial and physical evidence are needed to distinguish a sensor failure from a regional weather transition.

**What if the classifier is wrong during a disaster?**  
The Agent-4 safety gate prevents silent dismissal of high-priority readings unless the strong-evidence conditions are met. The fusion layer adds a second, explicit regional event shield.

**How is explainability handled?**  
Agent 4 uses SHAP over tree models and converts the leading features into human-readable reasons. The dashboard also exposes the intermediate evidence and gate decision.

**How do you repair bad data?**  
Agent 5 combines temporal and spatial estimates, excludes unhealthy neighbors, and preserves channels that were not diagnosed as faulty. Genuine events are never automatically corrected.

**Can it run in real time?**  
Yes. The FastAPI path is streaming/in-memory for the prototype; the architecture can replace the state store with Redis/TimescaleDB and run Agent 3 on edge hardware later.

## Evidence already validated in this repository

- Real Agent 2 pipeline integrated and exercised.
- Real Agent 4 screening-gate tests pass.
- Real Agent 5 correction evaluation: **5,463/5,463 labeled sensor faults evaluable (100% coverage)**; average percentage within the configured per-variable tolerance: **82.5%**.
- Six screening scenarios pass end-to-end in `test_skyguard_integration.py`.
- Agent 3 is wired to the committed LSTM artifacts and is explicitly labeled `LSTM_AUTOENCODER` when TensorFlow is available; otherwise the demo uses an explicitly labeled deterministic fallback.

## Do not claim

Do not claim "100% detection accuracy" or "guaranteed to win." The committed Agent-3 evaluation itself shows that the LSTM alone is imperfect. The credible claim is that SkyGuard combines independent evidence and safety policies so that a single model is not allowed to make the highest-risk decision by itself.
