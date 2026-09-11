"""
Phase 2 — Physical consistency rules
=====================================
Two layers:
  1. Single-variable "does this even make physical sense" range checks.
  2. Cross-variable relationship checks (things that shouldn't happen
     together, or a combination that's *extremely* unlikely).

Ranges are deliberately generous (they must not reject genuine extreme
weather — cyclones, heat waves, etc.) and are meant to be tuned per
network/region using Phase 7 historical validation. Defaults below are
reasonable for a general mid-latitude / tropical AWS network (e.g. IMD-
style deployment in India).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .schema import StationReading


@dataclass
class RangeConfig:
    temperature_c: tuple = (-40.0, 55.0)
    humidity_pct: tuple = (0.0, 100.0)
    pressure_hpa: tuple = (870.0, 1085.0)      # extreme cyclone low ~870, extreme high ~1085
    wind_speed_ms: tuple = (0.0, 113.0)         # 113 m/s ~ strongest recorded surface gust
    wind_direction_deg: tuple = (0.0, 360.0)
    rainfall_mm: tuple = (0.0, 300.0)           # per reporting interval; tune to interval length


@dataclass
class PhysicalCheckResult:
    variable_checks: Dict[str, str] = field(default_factory=dict)   # var -> PASS/FAIL/MISSING
    relationship_checks: Dict[str, str] = field(default_factory=dict)  # name -> PASS/FAIL/WARN
    failed_variables: List[str] = field(default_factory=list)
    failed_relationships: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "FAIL" if (self.failed_variables or self.failed_relationships) else "PASS"


def _in_range(value: Optional[float], bounds: tuple) -> bool:
    if value is None:
        return True  # missing values are handled separately, not a physical failure
    lo, hi = bounds
    return lo <= value <= hi


def check_ranges(reading: StationReading, cfg: RangeConfig) -> Dict[str, str]:
    results = {}
    checks = {
        "temperature_c": (reading.temperature_c, cfg.temperature_c),
        "humidity_pct": (reading.humidity_pct, cfg.humidity_pct),
        "pressure_hpa": (reading.pressure_hpa, cfg.pressure_hpa),
        "wind_speed_ms": (reading.wind_speed_ms, cfg.wind_speed_ms),
        "wind_direction_deg": (reading.wind_direction_deg, cfg.wind_direction_deg),
        "rainfall_mm": (reading.rainfall_mm, cfg.rainfall_mm),
    }
    for var, (value, bounds) in checks.items():
        if value is None:
            results[var] = "MISSING"
        elif _in_range(value, bounds):
            results[var] = "PASS"
        else:
            results[var] = "FAIL"
    return results


def check_relationships(reading: StationReading) -> Dict[str, str]:
    """
    Cross-variable sanity checks. These are WARN-level (suspicious but not
    an automatic FAIL) unless physically impossible, because genuine
    weather can produce unusual-looking but real combinations.
    """
    r: Dict[str, str] = {}
    t, h, p, ws, wd, rain = (
        reading.temperature_c,
        reading.humidity_pct,
        reading.pressure_hpa,
        reading.wind_speed_ms,
        reading.wind_direction_deg,
        reading.rainfall_mm,
    )

    # Temperature + Humidity: near-100% humidity with very high temperature and
    # zero rainfall for an extended dry spell is unusual but not impossible
    # (pre-monsoon mugginess). Flag only the truly implausible extreme.
    if t is not None and h is not None:
        if t > 45 and h > 95:
            r["temp_humidity"] = "WARN"
        else:
            r["temp_humidity"] = "PASS"

    # Temperature + Pressure: extremely low pressure with very high temperature
    # at the surface (deep cyclonic low usually brings cooler, cloudy conditions)
    if t is not None and p is not None:
        if p < 950 and t > 40:
            r["temp_pressure"] = "WARN"
        else:
            r["temp_pressure"] = "PASS"

    # Rainfall + Humidity: meaningful rainfall with very low humidity is
    # physically implausible (air must be near saturation to precipitate)
    if rain is not None and h is not None:
        if rain > 2.0 and h < 40:
            r["rainfall_humidity"] = "FAIL"
        else:
            r["rainfall_humidity"] = "PASS"

    # Wind speed + direction: direction should be present whenever there is
    # meaningful wind, and should be near-meaningless (but not "wrong") at calm
    if ws is not None and wd is not None:
        if ws > 1.0 and not (0 <= wd <= 360):
            r["wind_speed_direction"] = "FAIL"
        elif ws <= 0.2 and wd not in (0, None):
            # calm wind with a specific direction reported — sensor quirk, not fatal
            r["wind_speed_direction"] = "WARN"
        else:
            r["wind_speed_direction"] = "PASS"
    elif ws is not None and ws > 1.0 and wd is None:
        r["wind_speed_direction"] = "WARN"  # direction missing while wind is blowing

    return r


def run_physical_checks(reading: StationReading, cfg: Optional[RangeConfig] = None) -> PhysicalCheckResult:
    cfg = cfg or RangeConfig()
    var_results = check_ranges(reading, cfg)
    rel_results = check_relationships(reading)

    failed_vars = [v for v, status in var_results.items() if status == "FAIL"]
    failed_rels = [k for k, status in rel_results.items() if status == "FAIL"]
    warnings = [k for k, status in rel_results.items() if status == "WARN"]

    return PhysicalCheckResult(
        variable_checks=var_results,
        relationship_checks=rel_results,
        failed_variables=failed_vars,
        failed_relationships=failed_rels,
        warnings=warnings,
    )
