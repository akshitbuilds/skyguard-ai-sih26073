
"""Hybrid Evidence Fusion for SIH26073.

The original agents remain authoritative for their own jobs. This thin
orchestration layer adds auditable, deterministic "hard evidence" checks
before the final fault/event decision:
  - communication sentinel/missing values
  - flatline/frozen signatures
  - isolated step-change signatures
  - persistent drift signatures
  - multi-station event corroboration

This is intentionally neuro-symbolic: learned scores can be wrong, but the
safety-critical evidence rules are inspectable and can veto an unsafe call.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

FIELDS = ("temperature_c", "pressure_hpa", "humidity_pct")
RANGES = {
    "temperature_c": (-20.0, 60.0),
    "pressure_hpa": (850.0, 1100.0),
    "humidity_pct": (0.0, 100.0),
}

def _finite(v):
    return v is not None and np.isfinite(float(v))

def _delta(a, b):
    if not (_finite(a) and _finite(b)): return np.nan
    return float(a) - float(b)

def field_flatline_run(history: List[dict], current: dict, field: str, eps=1e-6) -> int:
    vals = [r.get(field) for r in history[-12:]] + [current.get(field)]
    run = 1
    for i in range(len(vals)-1, 0, -1):
        if _finite(vals[i]) and _finite(vals[i-1]) and abs(float(vals[i])-float(vals[i-1])) <= eps:
            run += 1
        else:
            break
    return run

def _robust_z(history: List[dict], current: dict, field: str) -> float:
    vals = [float(r[field]) for r in history[-12:] if _finite(r.get(field))]
    if len(vals) < 4 or not _finite(current.get(field)): return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(np.asarray(vals)-med)))
    scale = max(1.4826*mad, 0.25 if field=="temperature_c" else (0.15 if field=="pressure_hpa" else 1.0))
    return abs(float(current[field])-med)/scale

def _previous_before_current(history: List[dict], current: dict) -> Optional[dict]:
    """Return the latest valid historical row strictly before current timestamp."""
    cur_ts=pd.Timestamp(current.get("timestamp"))
    candidates=[]
    for row in history or []:
        try:
            ts=pd.Timestamp(row.get("timestamp"))
        except Exception:
            continue
        if ts < cur_ts:
            candidates.append((ts,row))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None

def _neighbor_delta_profile(history_by_station: Dict[str,List[dict]], neighbors: Dict[str,dict],
                            current: dict, field: str):
    cur = current.get(field)
    prev_self = None
    hs = history_by_station.get(current["station_id"], [])
    for r in reversed(hs):
        if str(r.get("timestamp")) == str(current.get("timestamp")):
            continue
        if _finite(r.get(field)):
            prev_self = r[field]; break
    target_delta = _delta(cur, prev_self)
    neighbor_deltas=[]
    for sid,ncur in neighbors.items():
        hs2=history_by_station.get(sid,[])
        prev=None
        for r in reversed(hs2):
            if str(r.get("timestamp")) == str(current.get("timestamp")):
                continue
            if _finite(r.get(field)):
                prev=r[field]; break
        d=_delta(ncur.get(field),prev)
        if np.isfinite(d): neighbor_deltas.append(float(d))
    med=float(np.median(neighbor_deltas)) if neighbor_deltas else np.nan
    return target_delta, med, neighbor_deltas

def event_corroboration(history_by_station: Dict[str,List[dict]], neighbors: Dict[str,dict],
                        current: dict) -> Tuple[float, dict]:
    """Score whether the *pattern* is spatially extended rather than isolated."""
    scores=[]; details={}
    for f in FIELDS:
        td, nd, nds = _neighbor_delta_profile(history_by_station, neighbors, current, f)
        if not np.isfinite(td) or not np.isfinite(nd) or len(nds)<2:
            continue
        same_sign = float(np.sign(td) == np.sign(nd) and np.sign(td) != 0)
        ratio = abs(td)/(abs(nd)+0.5)
        magnitude_match = 1.0 if 0.3 <= ratio <= 3.0 else max(0.0, 1.0-abs(math.log(max(ratio,1e-6)))/3.0)
        s = 0.6*same_sign + 0.4*magnitude_match
        scores.append(s)
        details[f]={"target_delta":round(float(td),3),"neighbor_median_delta":round(float(nd),3),
                    "magnitude_ratio":round(float(ratio),3),"score":round(float(s),3)}
    return (float(np.mean(scores)) if scores else 0.0), details


def regional_event_index(history_by_station: Dict[str,List[dict]], current_ts: str, hours: int = 6) -> float:
    """Network-wide weather-transition evidence over the last few samples."""
    flags=[]
    target=pd.Timestamp(current_ts)
    for sid, rows in history_by_station.items():
        valid=[r for r in rows if str(r.get("timestamp")) <= str(current_ts)]
        if len(valid) < hours+1: continue
        window=valid[-(hours+1):]
        if any(not all(_finite(r.get(f)) for f in FIELDS) for r in window): continue
        first,last=window[0],window[-1]
        dt={f:float(last[f]-first[f]) for f in FIELDS}
        votes=int(dt["temperature_c"] < -2.0)+int(dt["humidity_pct"] > 8.0)+int(dt["pressure_hpa"] < -0.5)
        flags.append(votes>=2)
    return float(np.mean(flags)) if flags else 0.0

def evidence_signature(history: List[dict], current: dict,
                       neighbors: Dict[str,dict], history_by_station: Dict[str,List[dict]],
                       agent2: dict|None=None, ml_score: float=0.0) -> dict:
    invalid = any((not _finite(current.get(f))) for f in FIELDS)
    range_breach = any(_finite(current.get(f)) and not (RANGES[f][0] <= float(current[f]) <= RANGES[f][1]) for f in FIELDS)
    flat = {f: field_flatline_run(history,current,f) for f in FIELDS}
    z = {f: _robust_z(history,current,f) for f in FIELDS}
    previous = _previous_before_current(history, current)
    td = {f: abs(_delta(current.get(f), previous.get(f))) if previous else 0.0 for f in FIELDS}
    event_score, event_details = event_corroboration(history_by_station, neighbors, current)
    regional_event = regional_event_index(history_by_station, current["timestamp"])
    event_isolated = any(
        d.get("magnitude_ratio", 0.0) >= 5.0 and abs(d.get("target_delta",0.0)) >= 10.0
        for d in event_details.values()
    )
    max_z=max(z.values()) if z else 0.0
    finite_deltas=[v for v in td.values() if np.isfinite(v)]
    max_delta=max(finite_deltas) if finite_deltas else 0.0
    isolated_spike = (
        (max_delta >= 12.0 and (event_score < 0.65 or event_isolated))
        or (max_z >= 8.0 and max_delta >= 6.0 and (event_score < 0.65 or event_isolated))
    )
    frozen = max(flat.values()) >= 4
    # Persistent spatial residual is a stronger drift signal than raw slope:
    # normal diurnal weather can trend for hours, but a degrading sensor tends
    # to move steadily away from its neighboring network baseline.
    drift_votes=0
    residual_slopes={}
    residual_changes={}
    drift_monotonicity={}
    drift_bias_sign_consistency={}
    station_id=current["station_id"]
    timestamp_map={}
    for oid, rows in history_by_station.items():
        for rr in rows:
            timestamp_map.setdefault(str(rr.get("timestamp")), {})[oid]=rr
    for f in FIELDS:
        vals=[]
        target_rows=[rr for rr in history_by_station.get(station_id,[]) if str(rr.get("timestamp")) <= str(current["timestamp"])]
        for rr in target_rows[-8:]:
            tm=str(rr.get("timestamp")); tv=rr.get(f)
            others=[x.get(f) for oid,x in timestamp_map.get(tm,{}).items() if oid!=station_id and _finite(x.get(f))]
            if _finite(tv) and len(others)>=2:
                vals.append(float(tv)-float(np.median(others)))
        if len(vals)>=6:
            slope=float(np.polyfit(np.arange(len(vals)),np.asarray(vals),1)[0])
            change=abs(vals[-1]-vals[0])
            residual_slopes[f]=slope; residual_changes[f]=change
            slope_thr={"temperature_c":0.50,"pressure_hpa":0.30,"humidity_pct":4.0}[f]
            change_thr={"temperature_c":2.0,"pressure_hpa":2.0,"humidity_pct":4.0}[f]
            # Drift must look like a persistent one-direction residual bias,
            # not merely a normal regional temperature/pressure transition.
            diffs=np.diff(np.asarray(vals,dtype=float))
            nonzero=diffs[np.abs(diffs)>1e-9]
            monotonic_fraction=(float(np.mean(nonzero >= 0)) if len(nonzero) else 0.0)
            if np.mean(diffs < 0) > 0.5:
                monotonic_fraction=float(np.mean(nonzero <= 0)) if len(nonzero) else 0.0
            residual_signs=np.sign(np.asarray(vals,dtype=float))
            residual_signs=residual_signs[residual_signs != 0]
            if len(residual_signs):
                positive=float(np.mean(residual_signs > 0))
                negative=float(np.mean(residual_signs < 0))
                sign_consistency=max(positive,negative)
            else:
                sign_consistency=0.0
            drift_monotonicity[f]=round(monotonic_fraction,3)
            drift_bias_sign_consistency[f]=round(sign_consistency,3)
            if (abs(slope)>=slope_thr and change>=change_thr
                and monotonic_fraction>=0.75 and sign_consistency>=0.80):
                drift_votes += 1
    drift = drift_votes >= 1 and not isolated_spike and not frozen
    return {
        "invalid_or_missing": bool(invalid),
        "range_breach": bool(range_breach),
        "flatline_runs": flat,
        "robust_z": {k:round(v,2) for k,v in z.items()},
        "absolute_deltas": {k:round(float(v),2) if np.isfinite(v) else None for k,v in td.items()},
        "event_corroboration": round(event_score,3),
        "regional_event_index": round(regional_event,3),
        "event_isolated_variable": bool(event_isolated),
        "event_details": event_details,
        "isolated_spike": bool(isolated_spike),
        "frozen_signature": bool(frozen),
        "drift_signature": bool(drift),
        "drift_residual_slopes": {k:round(v,3) for k,v in residual_slopes.items()},
        "drift_residual_changes": {k:round(v,2) for k,v in residual_changes.items()},
        "drift_monotonicity": drift_monotonicity,
        "drift_bias_sign_consistency": drift_bias_sign_consistency,
        "ml_score": round(float(ml_score),4),
    }

def decide(signature: dict, agent2: dict|None, gate_verdict: str|None) -> tuple[str,float,str]:
    """Return normalized shared anomaly type, confidence, and reason."""
    if signature["invalid_or_missing"]:
        return "sensor_dropout", 0.99, "Missing/non-finite telemetry detected across the AWS stream."
    if signature["range_breach"]:
        return "sensor_dropout", 0.98, "A telemetry value breached the configured physical validity envelope."
    # A spatially corroborated, physically coherent pattern gets the event shield.
    a2ver=(agent2 or {}).get("verdict")
    if (signature["event_corroboration"] >= 0.65 and signature["regional_event_index"] >= 0.60
        and not signature["event_isolated_variable"]
        and a2ver in ("NO_ANOMALY","POSSIBLE_REAL_EVENT","SUSPICIOUS")):
        return "genuine_event", 0.94, "Multi-station, multi-hour weather transition is spatially corroborated; protect the weather signal from correction."
    if signature["frozen_signature"]:
        return "sensor_stuck", 0.98, "A physical variable has remained effectively unchanged for four or more consecutive samples."
    if signature["isolated_spike"]:
        return "sensor_spike", 0.97, "A large local step-change is not sufficiently corroborated by neighboring stations."
    if signature["drift_signature"]:
        return "sensor_drift", 0.88, "A sustained monotonic change is inconsistent with a transient spike and is flagged for calibration review."
    if gate_verdict == "sensor_fault":
        return "sensor_spike", 0.72, "The Agent-4 gate supports a sensor fault, but no stronger deterministic signature was available."
    if gate_verdict in ("genuine_event","escalate_uncertain"):
        return "genuine_event", 0.72, "Safety policy protects an uncertain/high-priority anomaly from silent correction."
    if signature["ml_score"] >= 0.90:
        return "sensor_spike", 0.68, "The temporal ML detector reports a high anomaly score; downstream evidence is insufficient for a more specific class."
    return "none", 0.90, "No independent fault signature or high-confidence anomaly evidence was found."

def choose_focus_field(signature: dict, anomaly_type: str) -> str:
    if anomaly_type=="sensor_dropout": return "temperature_c"
    if anomaly_type=="sensor_stuck":
        return max(signature["flatline_runs"], key=signature["flatline_runs"].get)
    if anomaly_type=="sensor_spike":
        return max(signature["absolute_deltas"], key=lambda k: signature["absolute_deltas"][k] or -1)
    if anomaly_type=="sensor_drift":
        # Correct the channel whose residual bias actually satisfies the same
        # persistent-drift evidence used by the detector. This avoids choosing
        # a correlated channel merely because its raw z-score/change is larger.
        changes=signature.get("drift_residual_changes") or {}
        slopes=signature.get("drift_residual_slopes") or {}
        mono=signature.get("drift_monotonicity") or {}
        sign_consistency=signature.get("drift_bias_sign_consistency") or {}
        slope_thr={"temperature_c":0.50,"pressure_hpa":0.30,"humidity_pct":4.0}
        change_thr={"temperature_c":2.0,"pressure_hpa":2.0,"humidity_pct":4.0}
        candidates=[
            f for f in changes
            if abs(slopes.get(f,0.0)) >= slope_thr[f]
            and abs(changes.get(f,0.0)) >= change_thr[f]
            and mono.get(f,0.0) >= 0.75
            and sign_consistency.get(f,0.0) >= 0.80
        ]
        if candidates:
            return max(candidates, key=lambda k: (abs(changes.get(k,0.0)), abs(slopes.get(k,0.0))))
        if changes:
            return max(changes, key=lambda k: (abs(changes.get(k,0.0)), abs(slopes.get(k,0.0))))
        return max(signature["robust_z"], key=signature["robust_z"].get)
    return "temperature_c"
