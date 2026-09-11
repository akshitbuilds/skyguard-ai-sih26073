from __future__ import annotations
import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from .score_features import SCORE_FEATURE_NAMES, score_feature_vector

# Human-readable phrasing for each feature, used in the narrative.
SCORE_FEATURE_PHRASES = {
    "spatial_deviation_score": "how strongly nearby sensors corroborate this deviation (spatial consistency)",
    "ml_anomaly_score": "the upstream model's raw confidence that this reading is statistically anomalous",
    "physically_coherent": "whether the reading is physically plausible / consistent with correlated signals",
    "corroborated_event_strength": "the combined strength of spatial corroboration AND physical coherence (points toward a real event)",
    "incoherent_anomaly_strength": "the combined strength of a high anomaly score WITH physical incoherence (points toward a fault)",
}


class ScoreExplainer:
    """
    SHAP wrapper over the score-gate RandomForest, same pattern as
    `explainability.Explainer`. SHAP (exact TreeExplainer) is used rather
    than LIME because it's deterministic and exact for tree ensembles --
    no sampling variance, which matters when the explanation itself is
    part of the audit trail for an escalation decision. Falls back to the
    model's global `feature_importances_` if shap isn't installed, so the
    gate never hard-fails on a missing optional dependency.
    """

    def __init__(self, model):
        self.model = model
        self._explainer = shap.TreeExplainer(model) if _HAS_SHAP else None

    def explain(self, features: dict, predicted_label: str, top_k: int = 4):
        x = score_feature_vector(features).reshape(1, -1)
        used_shap = self._explainer is not None

        if used_shap:
            classes = list(self.model.classes_)
            class_idx = classes.index(predicted_label)
            shap_values = self._explainer.shap_values(x)
            if isinstance(shap_values, list):
                contribs = shap_values[class_idx][0]
            else:
                arr = np.asarray(shap_values)
                contribs = arr[0, :, class_idx] if arr.ndim == 3 else arr[0]
        else:
            # feature_importances_ is a GLOBAL, unsigned magnitude -- it has
            # no notion of "raised vs lowered" for this specific prediction.
            # We keep it as a fallback so the gate never hard-fails on a
            # missing optional dependency, but callers must not treat its
            # sign as meaningful (see `used_shap` / build_score_narrative).
            contribs = self.model.feature_importances_

        pairs = list(zip(SCORE_FEATURE_NAMES, contribs, x[0]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top = pairs[:top_k]

        return [
            {
                "feature": name,
                "value": round(float(val), 4),
                "shap_contribution": round(float(contrib), 4),
                "meaning": SCORE_FEATURE_PHRASES.get(name, name),
                "directional": used_shap,  # False => magnitude/importance only, no valid sign
            }
            for name, contrib, val in top
        ]


def build_score_narrative(
    sensor_id: str,
    final_verdict: str,
    model_verdict: str,
    model_confidence: float,
    top_features: list,
    safety_override_triggered: bool,
    gate_reason: str,
) -> str:
    """Plain-English narrative for a judge / IMD engineer / dashboard."""
    verdict_phrasing = {
        "genuine_event": "a genuine event, physically coherent and/or spatially corroborated",
        "sensor_fault": "a sensor fault, not a real event",
        "escalate_uncertain": "escalated as uncertain -- NOT dismissed, pending review",
    }
    verdict_str = verdict_phrasing.get(final_verdict, final_verdict)

    evidence_bits = []
    for f in top_features:
        if f.get("directional", True):
            direction = "raised" if f["shap_contribution"] > 0 else "lowered"
            evidence_bits.append(f"{f['meaning']} (value={f['value']}) {direction} confidence in this call")
        else:
            # Fallback mode (no shap installed): importances are unsigned,
            # so only claim relevance, not direction.
            evidence_bits.append(f"{f['meaning']} (value={f['value']}) was influential in this call")
    evidence_str = "; ".join(evidence_bits)

    narrative = (
        f"Sensor {sensor_id}: classified as {verdict_str}. "
        f"Underlying model call was '{model_verdict}' at {model_confidence * 100:.0f}% confidence. "
        f"Leading evidence: {evidence_str}."
    )

    if safety_override_triggered:
        narrative += f" {gate_reason}"

    return narrative
