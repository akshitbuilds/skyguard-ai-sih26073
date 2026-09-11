"""
Phase 4 — Temporal consistency
=================================
Looks at a station's own recent history for each variable and asks:
  - is this a continuous, physically-plausible trend? (NORMAL)
  - did it jump implausibly fast between consecutive readings? (SUDDEN_JUMP)
  - has the value not changed at all across many readings, longer than
    real weather noise would allow? (STALE — usually a frozen/dead sensor)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .schema import StationReading, VARIABLES

# Maximum physically-plausible change *per minute* for each variable.
# Tune with Phase 7 historical data; these are conservative starting points.
MAX_RATE_PER_MIN = {
    "temperature_c": 1.0,       # °C/min (extreme frontal passages can hit ~0.5-1/min)
    "humidity_pct": 8.0,        # %/min
    "pressure_hpa": 1.0,        # hPa/min (very fast even for intense cyclones)
    "wind_speed_ms": 5.0,       # m/s/min (gust ramp-up)
    "rainfall_mm": 15.0,        # mm/min (extreme convective burst)
}

# Number of consecutive identical readings (within STALE_EPSILON) before
# flagging a variable as stale, given readings arrive ~5 min apart.
STALE_MIN_READINGS = 6   # ~30 minutes of a perfectly flat line
STALE_EPSILON = {
    "temperature_c": 0.05,
    "humidity_pct": 0.5,
    "pressure_hpa": 0.05,
    "wind_speed_ms": 0.05,
    "rainfall_mm": 0.0,   # zero rainfall for a long time is normal, so we exclude it below
}


@dataclass
class VariableTemporalResult:
    variable: str
    verdict: str                    # NORMAL / SUDDEN_JUMP / STALE / INSUFFICIENT_DATA
    max_rate_observed: Optional[float] = None
    detail: str = ""


@dataclass
class TemporalCheckResult:
    per_variable: Dict[str, VariableTemporalResult] = field(default_factory=dict)
    jump_variables: List[str] = field(default_factory=list)
    stale_variables: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.jump_variables:
            return "SUDDEN_JUMP"
        if self.stale_variables:
            return "STALE"
        return "NORMAL"


def run_temporal_checks(
    history: List[StationReading],
    current: StationReading,
    rate_limits: Optional[Dict[str, float]] = None,
) -> TemporalCheckResult:
    """
    `history` = prior readings for the SAME station, sorted oldest->newest,
    NOT including `current`. Typically the last 30-60 minutes.
    """
    rate_limits = rate_limits or MAX_RATE_PER_MIN
    series = sorted(history, key=lambda r: r.timestamp) + [current]
    result = TemporalCheckResult()

    for var in VARIABLES:
        points = [(r.timestamp, getattr(r, var)) for r in series if getattr(r, var) is not None]
        if len(points) < 2:
            result.per_variable[var] = VariableTemporalResult(
                variable=var, verdict="INSUFFICIENT_DATA", detail="Fewer than 2 data points"
            )
            continue

        # rate-of-change check between the last two points
        (t_prev, v_prev), (t_now, v_now) = points[-2], points[-1]
        minutes = max((t_now - t_prev).total_seconds() / 60.0, 1e-6)

        if var == "wind_direction_deg":
            diff = min(abs(v_now - v_prev) % 360, 360 - abs(v_now - v_prev) % 360)
        else:
            diff = abs(v_now - v_prev)
        rate = diff / minutes

        limit = rate_limits.get(var)
        verdict = "NORMAL"
        detail = f"Rate {rate:.2f}/min over {minutes:.1f} min"

        if limit is not None and rate > limit:
            verdict = "SUDDEN_JUMP"
            detail = f"Rate {rate:.2f}/min exceeds limit {limit}/min"
            result.jump_variables.append(var)

        # staleness check: look at the last N readings for flatness
        elif var != "rainfall_mm" and len(points) >= STALE_MIN_READINGS:
            recent_vals = [v for _, v in points[-STALE_MIN_READINGS:]]
            spread = max(recent_vals) - min(recent_vals)
            eps = STALE_EPSILON.get(var, 0.0)
            if spread <= eps:
                verdict = "STALE"
                detail = f"No change (spread {spread:.3f}) across last {STALE_MIN_READINGS} readings"
                result.stale_variables.append(var)

        result.per_variable[var] = VariableTemporalResult(
            variable=var, verdict=verdict, max_rate_observed=rate, detail=detail
        )

    return result
