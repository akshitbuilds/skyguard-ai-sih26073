"""
Adapter between our flat shared schema and Agent 2's real pipeline
(agent2_screening/pipeline.py, schema.py, physical.py, spatial.py,
temporal.py, reasoning.py, scoring.py).

STATUS: verified against the real Agent 2 code (all 7 files), not just
guessed from the interface. Two things that were open assumptions are
now confirmed:

  CONFIRMED: score.physical_component / score.spatial_component are
  0-100 "how anomalous" values (0 = clean, 100 = maximally suspicious),
  per scoring.py's _physical_subscore/_spatial_subscore. Our flat
  schema's physical_consistency_score is the OPPOSITE polarity (higher
  = MORE consistent/normal) by original design, so we invert it below
  (and both are rescaled from 0-100 to 0-1 to match our schema).

  CONFIRMED: final_label is exactly "CLEAN" / "REVIEW" / "SUSPICIOUS",
  per scoring.py's compute_confidence_score.

Agent 2's schema.py also carries wind_speed_ms / wind_direction_deg /
rainfall_mm fields we don't have data for (PS26073 only gives us
temperature/pressure/humidity) -- these come through as None and every
check in physical.py/spatial.py/temporal.py already guards on None,
so they're silently skipped rather than causing false FAILs. Confirmed
by running the real pipeline end-to-end, not just by reading the code.
"""

from typing import Dict, List, Optional
import os
import sys

# Make sibling agent packages importable regardless of launch directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from agent2_screening.schema import StationReading as A2Reading
    from agent2_screening.pipeline import Agent2
    AGENT2_AVAILABLE = True
except ImportError as e:
    AGENT2_AVAILABLE = False
    _import_error = e

_agent2 = Agent2() if AGENT2_AVAILABLE else None

_LABEL_MAP = {
    "CLEAN": "clean",
    "REVIEW": "watch",
    "SUSPICIOUS": "high_priority",
}


def _to_a2_reading(flat: dict) -> "A2Reading":
    """flat has station_id, timestamp (str), latitude, longitude,
    temperature_c, pressure_hpa, humidity_pct -- Agent2's own
    StationReading.from_dict() already handles this shape directly."""
    return A2Reading.from_dict(flat)


def engine_status() -> dict:
    """Compatibility status endpoint used by manual diagnostics/preflight."""
    return {
        "available": bool(AGENT2_AVAILABLE),
        "engine": "REAL_AGENT2" if AGENT2_AVAILABLE else "UNAVAILABLE",
        "error": None if AGENT2_AVAILABLE else str(_import_error),
    }


def run_agent2_screening(current: dict, neighbors: List[dict], history: List[dict]) -> dict:
    """
    current, neighbors, history: flat dicts matching our shared schema.
    Returns the fields our flat schema expects: screening_flag,
    spatial_deviation_score, physical_consistency_score -- plus an
    extra 'agent2_detail' dict with the full rich report (explanation,
    failed variables etc.) for Agent 4 / the dashboard to use if useful.
    """
    if not AGENT2_AVAILABLE:
        raise RuntimeError(
            f"agent2_screening package not importable ({_import_error}). "
            "Need physical.py, spatial.py, temporal.py, reasoning.py, "
            "scoring.py alongside pipeline.py and schema.py."
        )

    a2_current = _to_a2_reading(current)
    a2_neighbors = [_to_a2_reading(n) for n in neighbors]
    a2_history = [_to_a2_reading(h) for h in history]

    report = _agent2.evaluate(current=a2_current, neighbors=a2_neighbors, history=a2_history)
    d = report.to_dict()

    screening_flag = _LABEL_MAP.get(d["final_label"], "watch")  # unknown label -> don't silently clear it

    # Agent 2's score components are 0-100 (see scoring.py), our shared schema
    # expects 0-1 floats -- normalize.
    spatial_deviation_score = d["score"]["spatial_component"] / 100.0
    # ASSUMPTION 1 (see module docstring) -- CONFIRMED against scoring.py:
    # physical_component is 0 (clean) to 100 (FAIL), i.e. "how anomalous",
    # opposite polarity from our "consistency" field, hence the inversion.
    physical_consistency_score = 1.0 - (d["score"]["physical_component"] / 100.0)

    return {
        "screening_flag": screening_flag,
        "spatial_deviation_score": spatial_deviation_score,
        "physical_consistency_score": physical_consistency_score,
        "agent2_detail": {
            "verdict": d["reasoning"]["verdict"],
            "explanation": d["reasoning"]["explanation"],
            "failed_variables": d["physical"]["failed_variables"],
            "deviating_variables": d["spatial"]["deviating_variables"],
            "jump_variables": d["temporal"]["jump_variables"],
            "stale_variables": d["temporal"]["stale_variables"],
            "score_total": d["score"]["total"],
            "processing_time_ms": d["processing_time_ms"],
        },
    }
