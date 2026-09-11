"""
Corrected/imputed value estimation for SkyGuard AI - Agent 5.

Approach:
  1. TEMPORAL: linear trend on the station's recent trusted history.
  2. SPATIAL: inverse-distance-weighted (IDW) estimate from healthy stations.
  3. Blend both when available; fall back safely when one/both are unavailable.

The estimator is deliberately explainable and defensive:
- missing/non-finite values are ignored rather than crashing the replay;
- each field is estimated independently, so a dropout in one channel does not
  invalidate the other channels;
- spatial neighbors are used only when that field is actually available;
- genuine_event is blocked by the caller before this estimator is invoked.
"""

from __future__ import annotations

import math

CLEAN_LOOKBACK_EXCLUDING_TAIL = 6
FAULT_TAIL_TO_EXCLUDE = 10
NEIGHBOR_RADIUS_KM = 800
MAX_NEIGHBORS = 4
TEMPORAL_WEIGHT_WHEN_BOTH = 0.15

# Maximum plausible one-step movement from the last trusted value.
MAX_PLAUSIBLE_DELTA = {
    "temperature_c": 8.0,
    "pressure_hpa": 12.0,
    "humidity_pct": 25.0,
}

FIELDS = ["temperature_c", "pressure_hpa", "humidity_pct"]


def _finite_number(value):
    """Return a finite float or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlmb / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _temporal_estimate(station_id, histories):
    """
    Fit a simple linear trend on trusted pre-fault history and extrapolate.

    Missing/non-finite field values are ignored per field. This is important
    for communication-dropout records where one or more channels can be None.
    """
    series = histories.get(station_id, [])

    if len(series) <= FAULT_TAIL_TO_EXCLUDE + 2:
        return None

    clean_tail = series[
        -(FAULT_TAIL_TO_EXCLUDE + CLEAN_LOOKBACK_EXCLUDING_TAIL):-FAULT_TAIL_TO_EXCLUDE
    ]

    if len(clean_tail) < 3:
        return None

    # Use the first valid timestamp as the time origin.
    t0 = None
    for point in clean_tail:
        if point.get("timestamp") is not None:
            t0 = point["timestamp"]
            break

    if t0 is None:
        return None

    target_timestamp = series[-1].get("timestamp")
    if target_timestamp is None:
        return None

    try:
        target_t = (target_timestamp - t0).total_seconds()
    except (TypeError, AttributeError):
        return None

    result = {}

    for field in FIELDS:
        pairs = []

        for point in clean_tail:
            timestamp = point.get("timestamp")
            value = _finite_number(point.get(field))

            if timestamp is None or value is None:
                continue

            try:
                x = (timestamp - t0).total_seconds()
            except (TypeError, AttributeError):
                continue

            if math.isfinite(x):
                pairs.append((x, value))

        # Need enough observations for a meaningful trend.
        if len(pairs) < 3:
            continue

        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(pairs)

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        denom = sum((x - mean_x) ** 2 for x in xs)

        if denom == 0:
            slope = 0.0
        else:
            slope = sum(
                (x - mean_x) * (y - mean_y)
                for x, y in pairs
            ) / denom

        intercept = mean_y - slope * mean_x
        estimate = intercept + slope * target_t

        if not math.isfinite(estimate):
            continue

        # Sanity-check against the last trusted value for this field.
        last_clean_value = _finite_number(clean_tail[-1].get(field))
        if last_clean_value is not None:
            if abs(estimate - last_clean_value) > MAX_PLAUSIBLE_DELTA[field]:
                continue

        result[field] = round(estimate, 2)

    return result or None


def _spatial_estimate(station_id, current_snapshot, meta, healthy_station_ids):
    """
    IDW estimate from nearby healthy stations.

    Each field is estimated independently. A neighbor missing humidity, for
    example, can still contribute its valid temperature and pressure.
    """
    if station_id not in meta:
        return None

    try:
        lat0 = float(meta[station_id]["latitude"])
        lon0 = float(meta[station_id]["longitude"])
    except (KeyError, TypeError, ValueError):
        return None

    candidates = []

    for other_id in healthy_station_ids:
        if (
            other_id == station_id
            or other_id not in meta
            or other_id not in current_snapshot
        ):
            continue

        try:
            lat = float(meta[other_id]["latitude"])
            lon = float(meta[other_id]["longitude"])
            distance = _haversine_km(lat0, lon0, lat, lon)
        except (KeyError, TypeError, ValueError):
            continue

        if math.isfinite(distance) and distance <= NEIGHBOR_RADIUS_KM:
            candidates.append((distance, other_id))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:MAX_NEIGHBORS]

    result = {}

    for field in FIELDS:
        usable = []

        for distance, other_id in candidates:
            value = _finite_number(current_snapshot[other_id].get(field))
            if value is None:
                continue

            weight = 1.0 / max(distance, 1.0) ** 2
            usable.append((weight, value))

        if not usable:
            continue

        total_weight = sum(weight for weight, _ in usable)

        if total_weight <= 0:
            continue

        result[field] = round(
            sum(weight * value for weight, value in usable) / total_weight,
            2,
        )

    return result or None


def estimate_corrected_value(
    station_id,
    histories,
    current_snapshot,
    meta,
    healthy_station_ids=None,
):
    """
    Main entry point.

    Returns a dict containing whichever fields can be safely estimated,
    or None when no estimate can be made.
    """
    if healthy_station_ids is None:
        healthy_station_ids = [
            sid for sid in meta if sid != station_id
        ]

    temporal = _temporal_estimate(station_id, histories)
    spatial = _spatial_estimate(
        station_id,
        current_snapshot,
        meta,
        healthy_station_ids,
    )

    if temporal and spatial:
        result = {}

        for field in FIELDS:
            t = _finite_number(temporal.get(field))
            s = _finite_number(spatial.get(field))

            if t is not None and s is not None:
                result[field] = round(
                    TEMPORAL_WEIGHT_WHEN_BOTH * t
                    + (1.0 - TEMPORAL_WEIGHT_WHEN_BOTH) * s,
                    2,
                )
            elif s is not None:
                result[field] = round(s, 2)
            elif t is not None:
                result[field] = round(t, 2)

        if result:
            return result

    if spatial:
        return {
            field: round(value, 2)
            for field, value in spatial.items()
            if _finite_number(value) is not None
        } or None

    if temporal:
        return {
            field: round(value, 2)
            for field, value in temporal.items()
            if _finite_number(value) is not None
        } or None

    # Last resort: most recent value before the corrupted tail.
    series = histories.get(station_id, [])

    if len(series) > FAULT_TAIL_TO_EXCLUDE:
        fallback = series[-FAULT_TAIL_TO_EXCLUDE - 1]
        result = {}

        for field in FIELDS:
            value = _finite_number(fallback.get(field))
            if value is not None:
                result[field] = round(value, 2)

        return result or None

    return None
