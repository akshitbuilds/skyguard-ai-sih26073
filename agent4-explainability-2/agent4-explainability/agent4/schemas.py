"""
Shared data contracts for Agent 4.

Agent 4 sits downstream of the anomaly-detection model (Agent 1-3 territory).
It does NOT decide whether a reading is anomalous -- it explains WHY a reading
that was already flagged is anomalous, and tracks whether a sensor is
degrading over time.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class RootCauseLabel(str, Enum):
    SENSOR_SPIKE = "SENSOR_SPIKE"          # implausible single-step jump, snaps back
    FROZEN_VALUE = "FROZEN_VALUE"          # value stuck / flatlined
    COMMS_ERROR = "COMMS_ERROR"            # dropouts, out-of-range fill values, gaps
    DRIFT = "DRIFT"                        # slow, monotonic bias creeping away from baseline
    GENUINE_EXTREME = "GENUINE_EXTREME"    # real extreme weather event, physically consistent


@dataclass
class SensorReading:
    sensor_id: str
    timestamp: str          # ISO 8601
    value: float


@dataclass
class DegradationStatus:
    sensor_id: str
    status: str                     # "STABLE" | "WATCH" | "DEGRADING" | "CRITICAL"
    recent_anomaly_rate: float
    baseline_anomaly_rate: float
    trend_slope: float
    frozen_streak_readings: int
    reasoning: str


class ScreeningFlag(str, Enum):
    """
    Upstream triage priority set before Agent 4 sees the reading (by the
    ingestion/scoring pipeline -- Agents 1-3). HIGH_PRIORITY means the
    upstream pipeline believes this could be a real, high-impact event.
    """
    HIGH_PRIORITY = "high_priority"
    STANDARD = "standard"
    LOW_PRIORITY = "low_priority"


class GateVerdict(str, Enum):
    """
    Final, post-safety-gate verdict for a flagged reading. ESCALATE_UNCERTAIN
    is a distinct outcome from SENSOR_FAULT: it means the evidence did not
    clear the bar required to safely dismiss a high-priority reading, so it
    is routed onward as if genuine, pending human/analyst review, rather than
    silently closed out.
    """
    GENUINE_EVENT = "genuine_event"
    SENSOR_FAULT = "sensor_fault"
    ESCALATE_UNCERTAIN = "escalate_uncertain"


@dataclass
class ScoreInput:
    """
    Agent 4's direct input contract with Agents 2 & 3, once their real
    outputs are wired in. All three score fields plus the screening_flag
    are expected to already exist on every flagged reading.

    spatial_deviation_score (0-1): degree to which NEARBY sensors show a
        correlated, coherent deviation at the same time -- i.e. spatial
        corroboration. High = the anomaly is echoed across the region
        (consistent with a real, physically-extended event like a storm
        system). Low = this sensor is an outlier relative to its neighbors,
        with no corroboration. (Agent 2's territory.)
    ml_anomaly_score (0-1): the upstream anomaly-detection model's own
        confidence that this reading is statistically anomalous, independent
        of *why*. (Agent 3's territory.)
    physically_coherent (bool): whether the reading is physically plausible
        on its own terms and consistent with correlated variables / known
        physical constraints for this sensor type and location.
    screening_flag (ScreeningFlag): upstream triage priority.
    """
    sensor_id: str
    timestamp: str
    spatial_deviation_score: float
    ml_anomaly_score: float
    physically_coherent: bool
    screening_flag: ScreeningFlag = ScreeningFlag.STANDARD


@dataclass
class ScoreClassification:
    """Output of the score-based genuine_event / sensor_fault gate."""
    sensor_id: str
    timestamp: str
    verdict: str                      # GateVerdict value -- FINAL, post-safety-gate
    model_verdict: str                # raw classifier call, pre-gate ("genuine_event" | "sensor_fault")
    model_confidence: float           # classifier confidence in model_verdict, 0-1
    class_probabilities: dict
    screening_flag: str
    safety_override_triggered: bool   # True if the non-negotiable gate changed the outcome
    gate_reason: str
    top_features: list                # SHAP-based, same shape as ExplainedFlag.top_features
    narrative: str
    inputs: dict                      # the ScoreInput fields, for audit trail

    def to_dict(self):
        return asdict(self)


@dataclass
class ExplainedFlag:
    sensor_id: str
    timestamp: str
    value: float
    root_cause: str                 # RootCauseLabel value
    confidence: float                # 0-1
    class_probabilities: dict
    top_features: list               # [{feature, value, shap_contribution}]
    narrative: str                   # human-readable explanation for a judge / IMD engineer
    degradation: Optional[dict] = None   # DegradationStatus as dict, if available
    upstream_model_score: Optional[float] = None  # passthrough from the ML agent, if provided

    def to_dict(self):
        return asdict(self)
