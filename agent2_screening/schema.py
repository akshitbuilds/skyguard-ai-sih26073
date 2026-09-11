"""
Phase 1 — Input schema
=======================
This is the data contract Agent 2 expects from the Data Engineering
streaming pipeline (e.g. Kafka / MQTT topic per station, or a batch
JSON payload). One `StationReading` = one station, one timestamp,
plus whatever `neighbors` were resolved for it upstream (or Agent2
can resolve neighbors itself if given a full network — see spatial.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class Coordinates:
    latitude: float
    longitude: float


@dataclass
class StationReading:
    """One measurement packet from one AWS station."""

    station_id: str
    timestamp: datetime
    location: Coordinates

    temperature_c: Optional[float] = None      # degrees Celsius
    humidity_pct: Optional[float] = None        # 0-100 %
    pressure_hpa: Optional[float] = None         # hectopascals (sea-level or station-level, see meta)
    wind_speed_ms: Optional[float] = None        # metres/second
    wind_direction_deg: Optional[float] = None   # 0-360, meteorological convention
    rainfall_mm: Optional[float] = None          # mm accumulated in the reporting interval

    # optional metadata that helps downstream checks but isn't required
    elevation_m: Optional[float] = None
    sensor_meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp.isoformat(),
            "latitude": self.location.latitude,
            "longitude": self.location.longitude,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "pressure_hpa": self.pressure_hpa,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_deg": self.wind_direction_deg,
            "rainfall_mm": self.rainfall_mm,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StationReading":
        ts = d["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return StationReading(
            station_id=d["station_id"],
            timestamp=ts,
            location=Coordinates(latitude=d["latitude"], longitude=d["longitude"]),
            temperature_c=d.get("temperature_c"),
            humidity_pct=d.get("humidity_pct"),
            pressure_hpa=d.get("pressure_hpa"),
            wind_speed_ms=d.get("wind_speed_ms"),
            wind_direction_deg=d.get("wind_direction_deg"),
            rainfall_mm=d.get("rainfall_mm"),
            elevation_m=d.get("elevation_m"),
            sensor_meta=d.get("sensor_meta", {}),
        )


VARIABLES = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_direction_deg",
    "rainfall_mm",
]
