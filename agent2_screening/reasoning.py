"""
Phase 5 — Don't immediately reject spatial anomalies
=======================================================
The core fix for the "flood/localized-event false alarm" problem:
a spatial deviation is a QUESTION, not a VERDICT.

    Neighbor difference
           |
    Is it physically possible?              (Phase 2)
           |
    Is it changing consistently over time?  (Phase 4)
           |
    Is there supporting evidence from
    other variables (cross-variable)?       (Phase 2 relationships + a
                                              same-direction check across
                                              humidity/pressure/wind/rain)
           |
    SUSPICIOUS  vs  POSSIBLE_REAL_EVENT  vs  LIKELY_SENSOR_FAULT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from .physical import PhysicalCheckResult
from .spatial import SpatialCheckResult
from .temporal import TemporalCheckResult

# variables whose simultaneous deviation in a "storm-like" direction
# supports a real meteorological event rather than a bad sensor
EVENT_SUPPORT_VARS = {"humidity_pct", "pressure_hpa", "wind_speed_ms", "rainfall_mm"}


@dataclass
class ReasoningResult:
    verdict: str                     # LIKELY_SENSOR_FAULT / SUSPICIOUS / POSSIBLE_REAL_EVENT / NO_ANOMALY
    explanation: str
    supporting_variables: List[str] = field(default_factory=list)


def reason_about_anomaly(
    physical: PhysicalCheckResult,
    spatial: SpatialCheckResult,
    temporal: TemporalCheckResult,
) -> ReasoningResult:

    # No spatial deviation at all -> nothing to reason about.
    if spatial.status == "CONSISTENT":
        return ReasoningResult(
            verdict="NO_ANOMALY",
            explanation="No significant deviation from neighboring stations.",
        )

    deviating = set(spatial.deviating_variables)

    # Step 1: is it physically impossible? -> straightforward fault signal,
    # but we still call it SUSPICIOUS (not an outright hard fault) because
    # Phase 6 scoring / the downstream ML model makes the final call.
    if physical.status == "FAIL":
        return ReasoningResult(
            verdict="LIKELY_SENSOR_FAULT",
            explanation=(
                f"Spatial deviation in {sorted(deviating)} co-occurs with a physical "
                f"impossibility ({physical.failed_variables + physical.failed_relationships}). "
                "This pattern is far more consistent with sensor malfunction than a real event."
            ),
            supporting_variables=sorted(deviating),
        )

    # Step 2: is the deviating variable changing erratically (sudden jump) with
    # no build-up, or is it stale? Erratic single-reading spikes with no other
    # corroborating variable look like noise/fault. A stale reading that
    # "deviates" from neighbors just because the sensor is frozen is also
    # a fault signal, not a real event.
    jump_vars = set(temporal.jump_variables)
    stale_vars = set(temporal.stale_variables)

    if deviating & stale_vars:
        return ReasoningResult(
            verdict="LIKELY_SENSOR_FAULT",
            explanation=(
                f"{sorted(deviating & stale_vars)} deviates from neighbors while showing no "
                "temporal change at all — consistent with a frozen/stuck sensor."
            ),
            supporting_variables=sorted(deviating & stale_vars),
        )

    erratic_jump_only = (deviating & jump_vars) and not (deviating & EVENT_SUPPORT_VARS - jump_vars)

    # Step 3: cross-variable support — do OTHER variables also point the same
    # direction a real event would (e.g. humidity up, pressure down, wind up,
    # rain starting)? This is the flood/storm early-warning case.
    support = sorted(deviating & EVENT_SUPPORT_VARS)

    if len(support) >= 2 and not (deviating & stale_vars):
        return ReasoningResult(
            verdict="POSSIBLE_REAL_EVENT",
            explanation=(
                f"Spatial deviation in {sorted(deviating)} is corroborated by {len(support)} "
                f"independent variables moving together ({support}) in a pattern consistent with "
                "a genuine localized weather event (e.g. developing storm/heavy rain cell). "
                "Recommend escalation to the ML model rather than discarding as a fault."
            ),
            supporting_variables=support,
        )

    if erratic_jump_only:
        return ReasoningResult(
            verdict="SUSPICIOUS",
            explanation=(
                f"{sorted(deviating & jump_vars)} shows an implausibly fast jump with no "
                "corroborating change in related variables — looks more like noise/glitch "
                "than a real event, but is not physically impossible so it is not auto-rejected."
            ),
            supporting_variables=sorted(deviating & jump_vars),
        )

    # Default: a single-variable, physically-valid, temporally-plausible
    # deviation without strong corroboration — genuinely ambiguous.
    return ReasoningResult(
        verdict="SUSPICIOUS",
        explanation=(
            f"{sorted(deviating)} deviates from neighboring stations; physically valid and "
            "not an obvious sensor glitch, but lacks enough independent corroboration to "
            "confirm a real event. Needs further review."
        ),
        supporting_variables=sorted(deviating),
    )
