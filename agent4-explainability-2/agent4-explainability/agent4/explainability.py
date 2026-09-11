from __future__ import annotations
import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from .features import FEATURE_NAMES, feature_vector

# Human-readable phrasing for each raw feature, used to build the narrative.
FEATURE_PHRASES = {
    "z_score": "how many standard deviations the reading sits from its recent local average",
    "delta_prev": "the jump from the previous reading",
    "roc": "the rate of change per sampling interval",
    "rolling_std": "the recent volatility (spread) of the sensor's readings",
    "rolling_mean_dev": "how far the reading deviates from its recent rolling mean, in std-devs",
    "flatline_run": "the number of consecutive near-identical readings",
    "missing_ratio": "the fraction of recent samples that were missing/interpolated",
    "gap_since_last": "the number of sampling intervals since the last valid reading",
    "out_of_range": "whether the value falls outside the physically plausible range",
    "range_breach_magnitude": "how far outside the plausible physical range the value falls",
    "monotonic_run": "the length of the consecutive upward/downward run ending at this point",
    "long_term_bias": "the shift between this window's average and the prior window's average",
}


class Explainer:
    """
    Wraps shap.TreeExplainer over the trained RandomForest so every
    classification comes with a ranked, human-readable "why".

    Falls back to the model's built-in feature_importances_ (global, not
    per-prediction) if the shap package isn't installed, so the pipeline
    never hard-fails on a missing optional dependency.
    """

    def __init__(self, model):
        self.model = model
        self._explainer = shap.TreeExplainer(model) if _HAS_SHAP else None

    def explain(self, features: dict, predicted_label: str, top_k: int = 3):
        x = feature_vector(features).reshape(1, -1)

        if self._explainer is not None:
            classes = list(self.model.classes_)
            class_idx = classes.index(predicted_label)
            shap_values = self._explainer.shap_values(x)
            # shap_values shape handling across shap versions (list-per-class vs 3D array)
            if isinstance(shap_values, list):
                contribs = shap_values[class_idx][0]
            else:
                arr = np.asarray(shap_values)
                contribs = arr[0, :, class_idx] if arr.ndim == 3 else arr[0]
        else:
            # global importances as a fallback, same sign convention not guaranteed
            contribs = self.model.feature_importances_

        pairs = list(zip(FEATURE_NAMES, contribs, x[0]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top = pairs[:top_k]

        top_features = [
            {
                "feature": name,
                "value": round(float(val), 4),
                "shap_contribution": round(float(contrib), 4),
                "meaning": FEATURE_PHRASES.get(name, name),
            }
            for name, contrib, val in top
        ]
        return top_features


def build_narrative(sensor_id: str, label: str, confidence: float, top_features: list,
                     degradation: dict | None = None) -> str:
    """
    Produces a single readable paragraph suitable for a judge or IMD
    engineer -- states the verdict, the confidence, the leading evidence,
    and (if available) whether this sensor has a track record of trouble.
    """
    label_phrasing = {
        "SENSOR_SPIKE": "an implausible single-step spike, not a real weather transition",
        "FROZEN_VALUE": "a frozen/stuck sensor value (readings have stopped changing)",
        "COMMS_ERROR": "a communications/telemetry fault (missing or fill-value samples)",
        "DRIFT": "a slow sensor drift away from its baseline, not a real trend",
        "GENUINE_EXTREME": "a genuine extreme weather reading, physically consistent with a real event",
    }
    verdict = label_phrasing.get(label, label)

    evidence_bits = []
    for f in top_features:
        sign = "raised" if f["shap_contribution"] > 0 else "lowered"
        evidence_bits.append(
            f"{f['meaning']} (value={f['value']}) {sign} confidence in this call"
        )
    evidence_str = "; ".join(evidence_bits)

    narrative = (
        f"Sensor {sensor_id} flagged as {verdict}, with {confidence * 100:.0f}% model confidence. "
        f"Leading evidence: {evidence_str}."
    )

    if degradation:
        if degradation.get("status") in ("DEGRADING", "CRITICAL"):
            narrative += (
                f" Note: this sensor's flag rate has been trending upward "
                f"({degradation['recent_anomaly_rate']*100:.0f}% recent vs "
                f"{degradation['baseline_anomaly_rate']*100:.0f}% baseline) -- "
                f"{degradation['reasoning']}"
            )
        elif degradation.get("status") == "WATCH":
            narrative += f" This sensor is on watch for degradation: {degradation['reasoning']}"

    return narrative
