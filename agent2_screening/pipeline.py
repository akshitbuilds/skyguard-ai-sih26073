"""
Phase 8 — Integration
========================
AWS Stations -> Data Engineering Stream -> Agent2 -> CLEAN/REVIEW/SUSPICIOUS -> ML Model

`Agent2.evaluate()` is the single entry point Data Engineering / the ML
layer calls per incoming reading. It runs Phases 2-6 and returns a full
`AnomalyReport` with every intermediate result attached (for logging,
debugging, and Phase 7 offline evaluation).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional, Dict, Any

from .schema import StationReading
from .physical import run_physical_checks, RangeConfig, PhysicalCheckResult
from .spatial import run_spatial_checks, SpatialCheckResult, DEFAULT_Z_THRESHOLD
from .temporal import run_temporal_checks, TemporalCheckResult
from .reasoning import reason_about_anomaly, ReasoningResult
from .scoring import compute_confidence_score, ScoringWeights, ScoreThresholds, ConfidenceScore


@dataclass
class AnomalyReport:
    station_id: str
    timestamp: str
    physical: PhysicalCheckResult
    spatial: SpatialCheckResult
    temporal: TemporalCheckResult
    reasoning: ReasoningResult
    score: ConfidenceScore
    final_label: str                 # CLEAN / REVIEW / SUSPICIOUS  (== score.label)
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp,
            "physical": {
                "status": self.physical.status,
                "failed_variables": self.physical.failed_variables,
                "failed_relationships": self.physical.failed_relationships,
                "warnings": self.physical.warnings,
            },
            "spatial": {
                "status": self.spatial.status,
                "deviating_variables": self.spatial.deviating_variables,
            },
            "temporal": {
                "status": self.temporal.status,
                "jump_variables": self.temporal.jump_variables,
                "stale_variables": self.temporal.stale_variables,
            },
            "reasoning": {
                "verdict": self.reasoning.verdict,
                "explanation": self.reasoning.explanation,
                "supporting_variables": self.reasoning.supporting_variables,
            },
            "score": {
                "total": self.score.total,
                "physical_component": self.score.physical_component,
                "spatial_component": self.score.spatial_component,
                "temporal_component": self.score.temporal_component,
                "cross_variable_component": self.score.cross_variable_component,
            },
            "final_label": self.final_label,
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


class Agent2:
    """The full Phase 2-6 pipeline, configurable per deployment/network."""

    def __init__(
        self,
        range_config: Optional[RangeConfig] = None,
        spatial_z_threshold: float = DEFAULT_Z_THRESHOLD,
        scoring_weights: Optional[ScoringWeights] = None,
        score_thresholds: Optional[ScoreThresholds] = None,
    ):
        self.range_config = range_config or RangeConfig()
        self.spatial_z_threshold = spatial_z_threshold
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.score_thresholds = score_thresholds or ScoreThresholds()

    def evaluate(
        self,
        current: StationReading,
        neighbors: List[StationReading],
        history: Optional[List[StationReading]] = None,
    ) -> AnomalyReport:
        """
        current   : the reading being screened right now
        neighbors : most recent readings from nearby stations (Phase 3)
        history   : this SAME station's recent readings, oldest->newest,
                    NOT including `current` (Phase 4). Pass [] if unavailable
                    (Agent2 degrades gracefully to spatial-only reasoning).
        """
        start = perf_counter()
        history = history or []

        physical = run_physical_checks(current, self.range_config)
        spatial = run_spatial_checks(current, neighbors, z_threshold=self.spatial_z_threshold)
        temporal = run_temporal_checks(history, current)
        reasoning = reason_about_anomaly(physical, spatial, temporal)
        score = compute_confidence_score(
            physical, spatial, temporal, reasoning,
            weights=self.scoring_weights, thresholds=self.score_thresholds,
        )

        elapsed_ms = (perf_counter() - start) * 1000.0

        return AnomalyReport(
            station_id=current.station_id,
            timestamp=current.timestamp.isoformat(),
            physical=physical,
            spatial=spatial,
            temporal=temporal,
            reasoning=reasoning,
            score=score,
            final_label=score.label,
            processing_time_ms=elapsed_ms,
        )

    def evaluate_batch(
        self,
        readings: List[StationReading],
        neighbor_map: Dict[str, List[StationReading]],
        history_map: Optional[Dict[str, List[StationReading]]] = None,
    ) -> List[AnomalyReport]:
        """Convenience wrapper for evaluating a batch/stream tick across many stations."""
        history_map = history_map or {}
        return [
            self.evaluate(
                current=r,
                neighbors=neighbor_map.get(r.station_id, []),
                history=history_map.get(r.station_id, []),
            )
            for r in readings
        ]
