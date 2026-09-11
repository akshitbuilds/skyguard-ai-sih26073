# Agent 4 — Root-Cause + Explainable Scoring

Downstream of the anomaly-detection model. That model decides **whether** a
sensor reading is anomalous — Agent 4 decides **why**, and whether the
sensor itself is trending toward failure. This is what turns a red flag into
something an IMD engineer or a judge can actually act on.

## What it does

1. **Root-cause classification** — every flagged reading is sorted into one of:
   - `SENSOR_SPIKE` — implausible single-step jump, snaps back
   - `FROZEN_VALUE` — value stuck / flatlined
   - `COMMS_ERROR` — dropouts, fill values, missing samples
   - `DRIFT` — slow, monotonic bias creeping away from baseline
   - `GENUINE_EXTREME` — a real extreme weather event, physically consistent

2. **SHAP explainability** — a `RandomForestClassifier` + `shap.TreeExplainer`
   gives an exact, per-prediction breakdown of which signals drove the
   call, rendered into a plain-English narrative (not just numbers).

3. **Sensor degradation tracker** — a rolling per-sensor log flags sensors
   whose flag-rate or flatline streaks are trending upward over time
   (`STABLE` → `WATCH` → `DEGRADING` → `CRITICAL`), so a sensor that's
   slowly dying gets caught before it produces a wall of bad data.

## Score-based front gate (`ScreeningGate`) — genuine_event vs sensor_fault

Upstream of the raw-signal root-cause classifier above sits a second,
independent stage: `agent4.ScreeningGate`. This is what talks directly to
Agent 2 (`spatial_deviation_score`) and Agent 3 (`ml_anomaly_score`) once
their real outputs are wired in, plus a `physically_coherent` check and the
upstream `screening_flag`. It answers one question first: **is this a real
event or a sensor fault, at all** — before anything reaches the detailed
`SENSOR_SPIKE` / `FROZEN_VALUE` / `COMMS_ERROR` / `DRIFT` taxonomy, which
still assumes a fault and just needs to characterize it.

```python
from agent4 import ScreeningGate

gate = ScreeningGate()  # auto-trains from synthetic score data on first run

result = gate.classify(
    sensor_id="IMD-TEMP-014",
    timestamp="2026-09-09T06:45:00",
    spatial_deviation_score=0.88,   # Agent 2's output
    ml_anomaly_score=0.91,          # Agent 3's output
    physically_coherent=True,       # physical-plausibility check
    screening_flag="high_priority", # upstream triage priority
)

print(result.verdict)        # "genuine_event" | "sensor_fault" | "escalate_uncertain"
print(result.narrative)
```

Always read `result.verdict` (the final, post-safety-gate outcome), not
`result.model_verdict` (the raw classifier call before the gate runs).

### Non-negotiable safety rule

**A `screening_flag: "high_priority"` reading is never silently classified
as a dismissible `sensor_fault` without strong evidence.** Concretely, for
a high-priority reading the classifier's own "sensor_fault" call is only
allowed to stand if ALL of the following hold:

1. classifier confidence ≥ 0.85
2. the reading is physically incoherent
3. no meaningful spatial corroboration (`spatial_deviation_score` ≤ 0.35)

If any of those three is missing, the reading is **not** dismissed — it
comes out as `escalate_uncertain` instead of `sensor_fault`, so it gets
routed to a human/analyst rather than closed out. This logic lives as
plain, auditable code in `apply_safety_gate()` inside
`agent4/screening_gate.py`, deliberately separate from the model itself,
because the design principle here is: **missing a real event is worse than
a false alarm.** See `tests/test_screening_gate.py` for the enforcement
test, and `examples/screening_gate_example_output.json` for a worked
example of the override actually firing.

### Semantics of the score fields (please confirm against Agent 2/3's actual definitions)

- `spatial_deviation_score` (0–1): built here as *spatial corroboration* —
  how strongly nearby sensors show a correlated deviation at the same time
  (high = a real, spatially-extended event; low = an isolated outlier).
  If Agent 2's field instead measures "how much this sensor differs from
  its neighbors" (the opposite direction), the sign needs flipping at the
  integration point — flag this to Agent 2 before wiring in real data.
- `ml_anomaly_score` (0–1): Agent 3's raw anomaly confidence, independent
  of cause.
- `physically_coherent` (bool): plausibility / cross-signal consistency
  check.
- `screening_flag`: `"high_priority" | "standard" | "low_priority"`.

**Swapping in real data:** replace `data/generate_score_synthetic.py`'s
`build_score_dataset()` with a loader over real historical Agent 2/3 scores
+ analyst-confirmed genuine_event/sensor_fault labels once that history
exists, keep using `agent4.score_features.extract_score_features` so
train/serve stays consistent, then delete
`artifacts/score_gate_model.joblib` to retrain.

Run it: `python demo_screening_gate.py` · `python tests/test_screening_gate.py`

## Architecture

```
agent4/
  features.py              # shared feature extraction (train + inference use the SAME function)
  root_cause_classifier.py # RandomForest wrapper: train / predict / save / load
  explainability.py        # SHAP wrapper + human-readable narrative generator
  degradation_tracker.py   # rolling per-sensor trend tracker
  pipeline.py               # Agent4 — the orchestrator you actually call
  schemas.py                # RootCauseLabel enum + output dataclasses + score-gate dataclasses
  score_features.py         # feature extraction for the score-based gate (Agent 2/3 inputs)
  score_classifier.py       # RandomForest wrapper for genuine_event vs sensor_fault
  score_explainability.py   # SHAP wrapper + narrative for the score-based gate
  screening_gate.py         # ScreeningGate — orchestrator + the non-negotiable safety gate
data/
  generate_synthetic.py       # labeled synthetic training set for the raw-signal classifier
  generate_score_synthetic.py # labeled synthetic training set for the score-based gate
demo.py                      # runnable end-to-end demo (raw-signal classifier)
demo_screening_gate.py       # runnable end-to-end demo (score-based gate + safety cases)
examples/screening_gate_example_output.json  # captured output from the 5 demo cases
tests/test_pipeline.py       # sanity tests (raw-signal classifier)
tests/test_screening_gate.py # sanity tests incl. the non-negotiable safety-gate case
```

## Integration contract with the ML model

Agent 4 does not need to know how the upstream model works. Once it flags a
reading, call:

```python
from agent4 import Agent4

agent = Agent4()  # auto-trains from synthetic data on first run, caches to artifacts/

result = agent.explain_flag(
    sensor_id="IMD-TEMP-014",
    timestamp="2026-09-09T06:45:00",
    value=42.4,
    window=[24.9, 25.1, 25.3, 25.0, 24.8, 25.2, 25.4, 42.4],  # trailing raw values, most recent last
    expected_range=(-5.0, 48.0),          # physically plausible range for this variable
    upstream_model_score=0.93,             # optional: your model's own anomaly score, passed through
)

print(result.narrative)
print(result.root_cause, result.confidence)
```

`explain_flag` accepts optional `missing_mask`, `gap_steps`, and
`prior_window` — pass these if your pipeline has them, they sharpen the
`COMMS_ERROR` / `DRIFT` calls. If you don't have them, sensible defaults are
used and the classifier still runs.

Output is a plain dataclass (`ExplainedFlag.to_dict()`) — JSON-serializable,
ready to hand to a dashboard or Agent 5.

**Swapping in real data:** replace `data/generate_synthetic.py`'s
`build_dataset()` with a loader over real historical flagged windows +
analyst-assigned labels. Keep using `agent4.features.extract_features` so
the feature vector stays identical between training and inference, then
just delete `artifacts/root_cause_model.joblib` and it retrains on next run.

## Running it

```bash
pip install -r requirements.txt
python demo.py              # full walkthrough, all 5 root-cause types + degradation trend
python tests/test_pipeline.py
```

## Design notes / why these choices

- **RandomForest, not a deep model**: with no real labeled incident data yet,
  a small synthetic-trained forest is honest about its confidence, trains in
  under a second, and pairs with an *exact* SHAP explainer (no sampling
  approximation) — important for a live demo where explanations need to be
  fast and trustworthy.
- **Feature extraction is a single shared function** (`features.py`) used by
  both training and inference, so there's no train/serve skew.
- **Degradation status logic is a simple, statable rule** (rate deltas +
  trend slope + flatline streaks), not another black box — a judge can
  follow exactly why a sensor got flagged as degrading.
