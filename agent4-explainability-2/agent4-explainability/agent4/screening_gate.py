"""
Score-based genuine_event / sensor_fault gate.

This is the front door for Agent 4 once Agent 2 (spatial_deviation_score)
and Agent 3 (ml_anomaly_score) have real fields flowing. It answers a single
binary question -- "is this flagged reading a real event or a sensor fault?"
-- from their scores plus a physical-coherence check and the upstream
screening_flag, using a RandomForest + SHAP for the statistical call.

================================================================================
NON-NEGOTIABLE DESIGN PRINCIPLE (this is the whole point of this module)
================================================================================
A screening_flag == "high_priority" reading must NEVER be silently
classified as a dismissible sensor_fault without strong, multi-signal
evidence. The statistical classifier is a single fallible signal -- for
high-stakes readings it does not get the final word by itself.

Concretely: if the classifier's own call is "sensor_fault" for a
high_priority reading, the `apply_safety_gate` function below requires
ALL of the following before it lets that call stand:
  1. classifier confidence >= FAULT_CONFIDENCE_THRESHOLD (default 0.85)
  2. the reading is physically incoherent
  3. there is no meaningful spatial corroboration
     (spatial_deviation_score <= SPATIAL_CORROBORATION_CEILING, default 0.35)

If ANY of those three is missing, the reading is NOT dismissed. It is
reclassified as `escalate_uncertain` instead of `sensor_fault` -- routed
onward for human/analyst review rather than silently closed out. This
holds even if the classifier itself is fairly confident; "fairly confident"
is not the bar for a high-priority reading.

Because missing a real event is worse than a false alarm, when the
evidence for "fault" is anything short of overwhelming for a high_priority
reading, this module always resolves toward genuine_event / escalation,
never toward silent dismissal. Standard/low_priority readings use the
classifier's call directly (still logged, just not gated) since a false
dismissal there does not carry the same cost.
================================================================================
"""

from __future__ import annotations
from typing import Optional

from .schemas import ScreeningFlag, GateVerdict, ScoreClassification
from .score_classifier import ScoreGateClassifier, DEFAULT_SCORE_MODEL_PATH
from .score_explainability import ScoreExplainer, build_score_narrative
from .score_features import extract_score_features

FAULT_CONFIDENCE_THRESHOLD = 0.85
SPATIAL_CORROBORATION_CEILING = 0.35


def apply_safety_gate(
    screening_flag: str,
    model_verdict: str,
    model_confidence: float,
    physically_coherent: bool,
    spatial_deviation_score: float,
    fault_confidence_threshold: float = FAULT_CONFIDENCE_THRESHOLD,
    spatial_corroboration_ceiling: float = SPATIAL_CORROBORATION_CEILING,
) -> tuple[str, bool, str]:
    """
    Returns (final_verdict, safety_override_triggered, gate_reason).

    This function is the enforcement point for the non-negotiable design
    principle described in the module docstring. It is intentionally kept
    separate from the ML classifier so the rule is auditable as plain code,
    not buried inside model weights.
    """
    flag = ScreeningFlag(screening_flag)

    if flag != ScreeningFlag.HIGH_PRIORITY:
        return (
            model_verdict,
            False,
            f"screening_flag={flag.value}: model verdict used as-is (still logged for audit).",
        )

    if model_verdict == GateVerdict.GENUINE_EVENT.value:
        return (
            GateVerdict.GENUINE_EVENT.value,
            False,
            "high_priority reading and the model already leans genuine_event -- no override needed.",
        )

    # From here: screening_flag == high_priority AND model_verdict == "sensor_fault".
    # This is the exact dangerous path the non-negotiable rule exists for.
    strong_evidence = (
        model_confidence >= fault_confidence_threshold
        and physically_coherent is False
        and spatial_deviation_score <= spatial_corroboration_ceiling
    )

    if strong_evidence:
        return (
            GateVerdict.SENSOR_FAULT.value,
            False,
            (
                f"high_priority reading, but strong multi-signal evidence supports fault "
                f"(model confidence {model_confidence:.2f} >= {fault_confidence_threshold}, "
                f"physically incoherent, spatial_deviation_score {spatial_deviation_score:.2f} <= "
                f"{spatial_corroboration_ceiling} i.e. no spatial corroboration) -- dismissal is justified."
            ),
        )

    return (
        GateVerdict.ESCALATE_UNCERTAIN.value,
        True,
        (
            f"SAFETY OVERRIDE: this high_priority reading's model call was 'sensor_fault' "
            f"(confidence {model_confidence:.2f}) but did not clear the strong-evidence bar "
            f"(requires confidence>={fault_confidence_threshold}, physically incoherent, and "
            f"spatial_deviation_score<={spatial_corroboration_ceiling}). Per design policy, missing a "
            f"real event is worse than a false alarm, so this is escalated rather than dismissed."
        ),
    )


class ScreeningGate:
    """
    Orchestrator: Agent 2/3 scores in, ScoreClassification out.

        gate = ScreeningGate()
        result = gate.classify(
            sensor_id="IMD-TEMP-014",
            timestamp="2026-09-09T06:45:00",
            spatial_deviation_score=0.82,
            ml_anomaly_score=0.91,
            physically_coherent=True,
            screening_flag="high_priority",
        )

    `result.verdict` is the FINAL, post-safety-gate outcome -- always use
    this, not `result.model_verdict`, when deciding what happens next
    downstream (e.g. whether to notify a human).
    """

    def __init__(self, model_path: str = DEFAULT_SCORE_MODEL_PATH):
        self.classifier = ScoreGateClassifier.load_or_train(model_path)
        self.explainer = ScoreExplainer(self.classifier.model)

    def classify(
        self,
        sensor_id: str,
        timestamp: str,
        spatial_deviation_score: float,
        ml_anomaly_score: float,
        physically_coherent: bool,
        screening_flag: str = ScreeningFlag.STANDARD.value,
    ) -> ScoreClassification:
        features = extract_score_features(
            spatial_deviation_score=spatial_deviation_score,
            ml_anomaly_score=ml_anomaly_score,
            physically_coherent=physically_coherent,
        )

        model_verdict, model_confidence, proba_dict = self.classifier.predict(features)
        top_features = self.explainer.explain(features, model_verdict)

        final_verdict, override_triggered, gate_reason = apply_safety_gate(
            screening_flag=screening_flag,
            model_verdict=model_verdict,
            model_confidence=model_confidence,
            physically_coherent=physically_coherent,
            spatial_deviation_score=spatial_deviation_score,
        )

        narrative = build_score_narrative(
            sensor_id=sensor_id,
            final_verdict=final_verdict,
            model_verdict=model_verdict,
            model_confidence=model_confidence,
            top_features=top_features,
            safety_override_triggered=override_triggered,
            gate_reason=gate_reason,
        )

        return ScoreClassification(
            sensor_id=sensor_id,
            timestamp=timestamp,
            verdict=final_verdict,
            model_verdict=model_verdict,
            model_confidence=round(model_confidence, 4),
            class_probabilities={k: round(v, 4) for k, v in proba_dict.items()},
            screening_flag=ScreeningFlag(screening_flag).value,
            safety_override_triggered=override_triggered,
            gate_reason=gate_reason,
            top_features=top_features,
            narrative=narrative,
            inputs={
                "spatial_deviation_score": spatial_deviation_score,
                "ml_anomaly_score": ml_anomaly_score,
                "physically_coherent": physically_coherent,
                "screening_flag": ScreeningFlag(screening_flag).value,
            },
        )
