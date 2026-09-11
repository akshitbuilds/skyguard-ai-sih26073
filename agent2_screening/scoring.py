"""
Phase 6 — Confidence scoring
==============================
Combines the four signal layers into a single 0-100 "suspicion score"
using configurable weights, then maps it to a CLEAN / REVIEW /
SUSPICIOUS label using configurable thresholds.

NOTE: the default weights (30/30/20/20) and thresholds (30/60) mirror
the design doc's starting point. Phase 7 (validation.py) is exactly
the mechanism for replacing these defaults with values fitted to real
historical AWS data, e.g. via ROC-curve threshold selection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .physical import PhysicalCheckResult
from .spatial import SpatialCheckResult
from .temporal import TemporalCheckResult
from .reasoning import ReasoningResult
from .schema import VARIABLES


@dataclass
class ScoringWeights:
    physical: float = 0.30
    spatial: float = 0.30
    temporal: float = 0.20
    cross_variable: float = 0.20

    def normalized(self) -> "ScoringWeights":
        total = self.physical + self.spatial + self.temporal + self.cross_variable
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            physical=self.physical / total,
            spatial=self.spatial / total,
            temporal=self.temporal / total,
            cross_variable=self.cross_variable / total,
        )


@dataclass
class ScoreThresholds:
    clean_max: float = 30.0
    review_max: float = 60.0
    # anything above review_max -> SUSPICIOUS


@dataclass
class ConfidenceScore:
    total: float
    physical_component: float
    spatial_component: float
    temporal_component: float
    cross_variable_component: float
    label: str  # CLEAN / REVIEW / SUSPICIOUS


def _physical_subscore(physical: PhysicalCheckResult) -> float:
    """0-100: how much the physical layer contributes to suspicion."""
    if physical.failed_variables or physical.failed_relationships:
        return 100.0
    if physical.warnings:
        return 40.0
    return 0.0


def _spatial_subscore(spatial: SpatialCheckResult) -> float:
    if not spatial.deviating_variables:
        return 0.0
    # scale with how many variables deviate and how far (avg |z|, capped)
    zs = [
        abs(v.z_score) for v in spatial.per_variable.values()
        if v.deviates and v.z_score is not None
    ]
    magnitude = min(sum(zs) / max(len(zs), 1) / 4.0, 1.0) if zs else 0.6
    coverage = min(len(spatial.deviating_variables) / len(VARIABLES), 1.0)
    return 100.0 * (0.6 * magnitude + 0.4 * coverage)


def _temporal_subscore(temporal: TemporalCheckResult) -> float:
    if temporal.jump_variables:
        return 90.0
    if temporal.stale_variables:
        return 70.0
    return 0.0


def _cross_variable_subscore(reasoning: ReasoningResult) -> float:
    """
    This is where Phase 5's reasoning gets folded back into the score.
    Strong multi-variable corroboration of a REAL event should PULL THE
    SCORE DOWN (less suspicious of the sensor), while corroboration of a
    fault should push it up.
    """
    mapping = {
        "NO_ANOMALY": 0.0,
        "POSSIBLE_REAL_EVENT": 10.0,      # corroborated -> low sensor-fault suspicion
        "SUSPICIOUS": 55.0,
        "LIKELY_SENSOR_FAULT": 95.0,
    }
    return mapping.get(reasoning.verdict, 50.0)


def compute_confidence_score(
    physical: PhysicalCheckResult,
    spatial: SpatialCheckResult,
    temporal: TemporalCheckResult,
    reasoning: ReasoningResult,
    weights: ScoringWeights = ScoringWeights(),
    thresholds: ScoreThresholds = ScoreThresholds(),
) -> ConfidenceScore:
    w = weights.normalized()

    p = _physical_subscore(physical)
    s = _spatial_subscore(spatial)
    t = _temporal_subscore(temporal)
    c = _cross_variable_subscore(reasoning)

    total = w.physical * p + w.spatial * s + w.temporal * t + w.cross_variable * c

    if total <= thresholds.clean_max:
        label = "CLEAN"
    elif total <= thresholds.review_max:
        label = "REVIEW"
    else:
        label = "SUSPICIOUS"

    return ConfidenceScore(
        total=round(total, 2),
        physical_component=round(p, 2),
        spatial_component=round(s, 2),
        temporal_component=round(t, 2),
        cross_variable_component=round(c, 2),
        label=label,
    )
