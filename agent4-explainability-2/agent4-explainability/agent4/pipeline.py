from __future__ import annotations
from typing import Optional

from .root_cause_classifier import RootCauseClassifier, DEFAULT_MODEL_PATH
from .explainability import Explainer, build_narrative
from .degradation_tracker import SensorDegradationTracker
from .features import extract_features
from .schemas import ExplainedFlag


class Agent4:
    """
    Root-Cause + Explainable Scoring agent.

    INTEGRATION CONTRACT with the anomaly-detection model (Agent 1-3):
    that model decides WHETHER a reading is anomalous. Once it flags one,
    call `Agent4.explain_flag(...)` with:

      - sensor_id, timestamp, value              (required)
      - window: list of recent raw values ending at `value`  (required, >=4 points recommended)
      - expected_range: (min, max) physically plausible values for this variable
      - missing_mask / gap_steps / prior_window: optional, improve comms-error & drift detection
      - upstream_model_score: optional float (e.g. the ML model's own anomaly score /
        reconstruction error), just passed through into the output for the dashboard --
        Agent 4's own confidence is independent of it.

    Returns an ExplainedFlag: root cause label, confidence, SHAP-based top
    features with plain-English meaning, a ready-to-display narrative
    sentence, and this sensor's degradation status.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.classifier = RootCauseClassifier.load_or_train(model_path)
        self.explainer = Explainer(self.classifier.model)
        self.degradation_tracker = SensorDegradationTracker()

    def explain_flag(
        self,
        sensor_id: str,
        timestamp: str,
        value: float,
        window: list[float],
        expected_range: tuple[float, float] = (-5.0, 48.0),
        missing_mask: Optional[list[bool]] = None,
        gap_steps: int = 1,
        prior_window: Optional[list[float]] = None,
        upstream_model_score: Optional[float] = None,
    ) -> ExplainedFlag:
        features = extract_features(
            window=window,
            expected_range=expected_range,
            missing_mask=missing_mask,
            gap_steps=gap_steps,
            prior_window=prior_window,
        )

        label, confidence, proba_dict = self.classifier.predict(features)
        top_features = self.explainer.explain(features, label)

        self.degradation_tracker.update(
            sensor_id=sensor_id,
            timestamp=timestamp,
            is_anomaly=True,   # Agent4 only ever sees things already flagged upstream
            root_cause=label,
            flatline_run=features["flatline_run"],
        )
        degradation = self.degradation_tracker.get_status(sensor_id)

        narrative = build_narrative(sensor_id, label, confidence, top_features, degradation)

        return ExplainedFlag(
            sensor_id=sensor_id,
            timestamp=timestamp,
            value=value,
            root_cause=label,
            confidence=round(confidence, 4),
            class_probabilities={k: round(v, 4) for k, v in proba_dict.items()},
            top_features=top_features,
            narrative=narrative,
            degradation=degradation,
            upstream_model_score=upstream_model_score,
        )

    def note_clean_reading(self, sensor_id: str, timestamp: str):
        """
        Optional: call this for readings the upstream model did NOT flag,
        so the degradation tracker's rolling rate reflects true anomaly
        frequency rather than only ever-flagged events. Skipping this is
        fine -- the tracker still works off flag density alone.
        """
        self.degradation_tracker.update(
            sensor_id=sensor_id, timestamp=timestamp,
            is_anomaly=False, root_cause=None, flatline_run=0.0,
        )

    def sensor_health_report(self) -> list[dict]:
        return self.degradation_tracker.all_statuses()
