"""
Shared schema/constants for SkyGuard AI - Agent 5 (Correction & Alert + Dashboard).
This mirrors the record contract handed off by Agents 1-4. Keeping it in one
the integrated backend and dashboard can both target the same field contract.
"""

SCREENING_FLAGS = ["pass", "spatial_outlier", "physical_inconsistency", "flagged"]

ANOMALY_TYPES = [
    "none",
    "sensor_stuck",       # sensor frozen at a constant value
    "sensor_spike",       # single/short implausible spike
    "sensor_drift",       # slow calibration drift away from truth
    "sensor_dropout",     # missing / garbage readings
    "genuine_event",      # real weather (storm, front passage, etc.) - NOT a fault
]

HEALTH_STATUS = ["green", "amber", "red"]
ALERT_SEVERITY = ["none", "low", "medium", "high", "critical"]

# Health/severity ordering used for sorting the alert feed and map coloring
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
HEALTH_ORDER = {"red": 0, "amber": 1, "green": 2}

HEALTH_COLOR = {
    "green": "#2ecc71",
    "amber": "#f39c12",
    "red": "#e74c3c",
}

SEVERITY_COLOR = {
    "critical": "#c0392b",
    "high": "#e67e22",
    "medium": "#f1c40f",
    "low": "#3498db",
    "none": "#95a5a6",
}

RECORD_FIELDS = [
    "station_id", "timestamp", "latitude", "longitude",
    "temperature_c", "pressure_hpa", "humidity_pct",
    "spatial_deviation_score", "physical_consistency_score", "screening_flag",
    "ml_anomaly_score", "reconstruction_error",
    "anomaly_type", "confidence_score",
    "explanation", "sensor_health_status",
    "corrected_value", "alert_severity",
]
