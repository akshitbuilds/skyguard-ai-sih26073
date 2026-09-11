"""
Agent 2 — AWS (Automatic Weather Station) Anomaly Pre-Screening Agent
======================================================================

Implements Phases 1-8 of the design:

    Phase 1  Input schema              -> agent2_screening.schema
    Phase 2  Physical consistency      -> agent2_screening.physical
    Phase 3  Spatial consistency       -> agent2_screening.spatial
    Phase 4  Temporal consistency      -> agent2_screening.temporal
    Phase 5  Reasoning / non-rejection -> agent2_screening.reasoning
    Phase 6  Confidence scoring        -> agent2_screening.scoring
    Phase 7  Historical validation     -> agent2_screening.validation
    Phase 8  Integration pipeline      -> agent2_screening.pipeline

Quick start:

    from agent2_screening import Agent2
    agent = Agent2()
    report = agent.evaluate(station_reading, neighbor_readings, history)
    print(report.final_label, report.confidence_score)
"""

from .pipeline import Agent2
from .schema import StationReading, Coordinates

__all__ = ["Agent2", "StationReading", "Coordinates"]
__version__ = "1.0.0"
