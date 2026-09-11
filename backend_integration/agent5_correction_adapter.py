
"""Adapter for Agent 5 correction with explicit same-timestamp neighbors."""
from datetime import datetime
from typing import Dict, Optional
import statistics
from correction import estimate_corrected_value, FIELDS
from station_state import StationState

FAULT_TYPES_TO_CORRECT={"sensor_stuck","sensor_spike","sensor_drift","sensor_dropout"}

def _parse_history(raw_history:list,current_timestamp:str)->list:
    parsed=[]
    for r in raw_history:
        parsed.append({"timestamp":datetime.fromisoformat(r["timestamp"]),**{f:r[f] for f in FIELDS}})
    parsed.append({"timestamp":datetime.fromisoformat(current_timestamp),**{f:0.0 for f in FIELDS}})
    return parsed

def maybe_correct(reading:dict, anomaly_type:str, state:StationState,
                  latest_screening:Dict[str,str], current_snapshot:Optional[Dict[str,dict]]=None,
                  healthy_ids:Optional[list[str]]=None, fault_field:Optional[str]=None)->Optional[dict]:
    if anomaly_type not in FAULT_TYPES_TO_CORRECT: return None
    sid=reading["station_id"]
    history=state.get_history(sid)
    histories={sid:_parse_history(history,reading["timestamp"])}
    if current_snapshot is None:
        current_snapshot={oid:{f:r[f] for f in FIELDS} for oid,r in state.get_neighbors_current(sid).items()
                          if all(r.get(f) is not None for f in FIELDS)}
    else:
        current_snapshot={oid:{f:r[f] for f in FIELDS} for oid,r in current_snapshot.items()
                          if oid!=sid and all(r.get(f) is not None for f in FIELDS)}
    if healthy_ids is None:
        healthy_ids=state.healthy_neighbor_ids(sid,latest_screening)
    estimated=estimate_corrected_value(sid,histories,current_snapshot,state.meta,healthy_ids)
    if estimated is None:
        return None

    # Seasonal anchor: same station + same hour-of-day over recent history.
    # This is especially effective for diurnal temperature/humidity cycles
    # when a sensor has been frozen for many hours.
    def seasonal_value(field):
        try:
            ts=datetime.fromisoformat(reading["timestamp"])
            vals=[float(x[field]) for x in state.get_history(sid)
                  if x.get(field) is not None and str(x.get(field))!="nan"
                  and datetime.fromisoformat(x["timestamp"]).hour==ts.hour]
            return float(statistics.median(vals[-14:])) if len(vals)>=3 else None
        except Exception:
            return None

    if fault_field and fault_field in FIELDS and anomaly_type != "sensor_dropout":
        merged={f:reading[f] for f in FIELDS}
        seasonal=seasonal_value(fault_field)
        merged[fault_field]=round(float(seasonal),2) if seasonal is not None else estimated[fault_field]
        return merged
    if anomaly_type=="sensor_dropout":
        # Prefer the same-hour seasonal anchor for each missing channel when available.
        out={}
        for f in FIELDS:
            sv=seasonal_value(f)
            out[f]=round(float(sv),2) if sv is not None else estimated[f]
        return out
    return estimated
