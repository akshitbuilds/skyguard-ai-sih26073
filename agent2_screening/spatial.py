"""
Phase 3 — Spatial consistency
===============================
Compares station A against its nearby neighbors (B, C, D, E, ...) for
each variable. Produces a deviation score per variable and an overall
SPATIAL = CONSISTENT / DEVIATION verdict — but (per Phase 5) this is
NOT a final fault verdict, just a signal fed into the reasoning layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import List, Dict, Optional
from .schema import StationReading, VARIABLES

# How many "neighbor standard deviations" away counts as a deviation,
# with an absolute-difference fallback for variables where neighbor
# spread is naturally near-zero (e.g. all neighbors report identical humidity).
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_ABS_THRESHOLD = {
    "temperature_c": 4.0,     # °C
    "humidity_pct": 20.0,     # %
    "pressure_hpa": 3.0,      # hPa
    "wind_speed_ms": 4.0,     # m/s
    "wind_direction_deg": 90.0,  # degrees
    "rainfall_mm": 5.0,       # mm
}


@dataclass
class VariableDeviation:
    variable: str
    value: Optional[float]
    neighbor_mean: Optional[float]
    neighbor_std: Optional[float]
    abs_difference: Optional[float]
    z_score: Optional[float]
    deviates: bool


@dataclass
class SpatialCheckResult:
    per_variable: Dict[str, VariableDeviation] = field(default_factory=dict)
    deviating_variables: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "DEVIATION" if self.deviating_variables else "CONSISTENT"


def run_spatial_checks(
    target: StationReading,
    neighbors: List[StationReading],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    abs_thresholds: Optional[Dict[str, float]] = None,
) -> SpatialCheckResult:
    abs_thresholds = abs_thresholds or DEFAULT_ABS_THRESHOLD
    result = SpatialCheckResult()

    if not neighbors:
        # No neighbors resolved — can't make a spatial judgement.
        for var in VARIABLES:
            result.per_variable[var] = VariableDeviation(
                variable=var, value=getattr(target, var), neighbor_mean=None,
                neighbor_std=None, abs_difference=None, z_score=None, deviates=False,
            )
        return result

    for var in VARIABLES:
        target_val = getattr(target, var)
        neighbor_vals = [getattr(n, var) for n in neighbors if getattr(n, var) is not None]

        if target_val is None or not neighbor_vals:
            result.per_variable[var] = VariableDeviation(
                variable=var, value=target_val, neighbor_mean=None,
                neighbor_std=None, abs_difference=None, z_score=None, deviates=False,
            )
            continue

        # circular handling for wind direction
        if var == "wind_direction_deg":
            diffs = [_circular_diff(target_val, v) for v in neighbor_vals]
            abs_diff = mean(diffs)
            n_mean = mean(neighbor_vals)
            n_std = pstdev(neighbor_vals) if len(neighbor_vals) > 1 else 0.0
        else:
            n_mean = mean(neighbor_vals)
            n_std = pstdev(neighbor_vals) if len(neighbor_vals) > 1 else 0.0
            abs_diff = abs(target_val - n_mean)

        z = (abs_diff / n_std) if n_std > 1e-6 else None
        threshold_abs = abs_thresholds.get(var, 0.0)

        deviates = abs_diff > threshold_abs and (z is None or z > z_threshold)

        result.per_variable[var] = VariableDeviation(
            variable=var, value=target_val, neighbor_mean=n_mean, neighbor_std=n_std,
            abs_difference=abs_diff, z_score=z, deviates=deviates,
        )
        if deviates:
            result.deviating_variables.append(var)

    return result


def _circular_diff(a: float, b: float) -> float:
    """Smallest angular difference between two compass bearings (0-360)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)
