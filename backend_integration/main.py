
"""SkyGuard AI Command Backend — SIH26073.

Production-style demo orchestrator:
Agent 1 stream -> Agent 2 physics/spatial/temporal -> Agent 3 LSTM-AE
-> Hybrid Evidence Fusion -> Agent 4 score gate + SHAP + degradation
-> Agent 5 correction + severity -> dashboard/API.

The batch path is deliberately tick-synchronous so spatial evidence compares
stations at the SAME timestamp rather than accidentally comparing one station
to a previous tick.
"""
from __future__ import annotations
import os, sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BACKEND_ROOT, ".."))
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import StationReading
from station_state import StationState
from agent2_adapter import run_agent2_screening, AGENT2_AVAILABLE
from agent3_adapter import run_agent3, engine_status as agent3_status
from agent4_adapter import run_agent4, status as agent4_status
from agent5_correction_adapter import maybe_correct
from fusion import evidence_signature, decide, choose_focus_field

STATION_META={
    "AWS_DIU":{"latitude":20.7141,"longitude":70.9822,"name":"Diu"},
    "AWS_VERAVAL":{"latitude":20.9077,"longitude":70.3679,"name":"Veraval"},
    "AWS_MAHUVA":{"latitude":21.0901,"longitude":71.7690,"name":"Mahuva"},
    "AWS_PORBANDAR":{"latitude":21.6422,"longitude":69.6093,"name":"Porbandar"},
    "AWS_BHAVNAGAR":{"latitude":21.7629,"longitude":72.1533,"name":"Bhavnagar"},
}
state=StationState({k:{x:y for x,y in v.items() if x!="name"} for k,v in STATION_META.items()})
_latest_screening:Dict[str,str]={}

app=FastAPI(title="SkyGuard AI Command Center",version="2.0")

def _severity(anomaly:str,confidence:float,gate:dict|None,health:str)->str:
    if anomaly=="none": return "none"
    if anomaly=="sensor_dropout": return "critical"
    if anomaly=="sensor_spike": return "high" if confidence>=.85 else "medium"
    if anomaly=="sensor_stuck": return "medium"
    if anomaly=="sensor_drift": return "low" if health!="red" else "medium"
    if anomaly=="genuine_event": return "high" if confidence>=.9 else "medium"
    return "medium"

def _as_reading(d): return StationReading(**d)

def run_tick(readings:List[StationReading])->List[StationReading]:
    if not readings: return []
    a4_runtime = agent4_status()
    if not a4_runtime.get("available"):
        raise RuntimeError(
            f"Agent 4 is unavailable; final explainability/safety path cannot run: {a4_runtime.get('error')}"
        )
    if not AGENT2_AVAILABLE:
        raise RuntimeError("Agent 2 is unavailable; final screening path cannot run.")
    # Current tick network snapshot; every agent sees the same simultaneous values.
    incoming={r.station_id:r.model_dump() for r in readings}
    histories=state.snapshot_histories()
    # If a station is new, its history is empty.
    processed=[]
    latest_screening=dict(_latest_screening)

    for reading in readings:
        r=reading.model_dump()
        sid=r["station_id"]
        hist=[h for h in state.get_history(sid) if all(
            h.get(f) is not None and pd.notna(h.get(f)) for f in ("temperature_c","pressure_hpa","humidity_pct")
        )]
        # Agent 3 needs the station's latest pre-current reading included so
        # its 12-point window is truly consecutive. StationState.get_history()
        # intentionally excludes the latest current state, so use the snapshot
        # (history + current) for the ML window only.
        current_ts=pd.to_datetime(r["timestamp"], errors="coerce")
        agent3_hist=[]
        for h in histories.get(sid, []):
            h_ts=pd.to_datetime(h.get("timestamp"), errors="coerce")
            if pd.notna(h_ts) and pd.notna(current_ts) and h_ts < current_ts and all(
                h.get(f) is not None and pd.notna(h.get(f)) for f in ("temperature_c","pressure_hpa","humidity_pct")
            ):
                agent3_hist.append(h)
        # Agent 2 contract: neighbors is a LIST of reading dictionaries.
        # Fusion/Agent 4 contract: neighbors is a DICT keyed by station id.
        # Keep these representations separate; iterating a dict would yield
        # station-id strings and crash Agent 2 schema parsing.
        agent2_neighbors=[v for oid,v in incoming.items() if oid!=sid and all(
            v.get(f) is not None and pd.notna(v.get(f)) for f in ("temperature_c","pressure_hpa","humidity_pct")
        )]
        fusion_neighbors={oid:v for oid,v in incoming.items() if oid!=sid and all(
            v.get(f) is not None and pd.notna(v.get(f)) for f in ("temperature_c","pressure_hpa","humidity_pct")
        )}
        current_valid=all(r.get(f) is not None and pd.notna(r.get(f)) for f in ("temperature_c","pressure_hpa","humidity_pct"))
        if AGENT2_AVAILABLE and current_valid:
            a2=run_agent2_screening(r,agent2_neighbors,hist)
        elif AGENT2_AVAILABLE:
            a2={
                "screening_flag":"high_priority","spatial_deviation_score":0.0,
                "physical_consistency_score":0.0,
                "agent2_detail":{"verdict":"LIKELY_SENSOR_FAULT","explanation":"Non-finite telemetry bypassed statistical spatial check and was escalated by the integration safety layer.",
                                 "failed_variables":["temperature_c","pressure_hpa","humidity_pct"],
                                 "deviating_variables":[],"jump_variables":[],"stale_variables":[],"score_total":100.0,
                                 "processing_time_ms":0.0},
            }
        else:
            raise RuntimeError(
                "Agent 2 is unavailable. SkyGuard final mode is fail-closed: "
                "refusing to classify readings with synthetic screening evidence."
            )
        r.update(a2)
        latest_screening[sid]=r["screening_flag"]
        a3=run_agent3(r,agent3_hist)
        r.update(a3)
        # Include current tick in per-station history for event corroboration.
        hnet={k:list(v) for k,v in histories.items()}
        for oid,cur in incoming.items():
            if oid in hnet: hnet[oid]=hnet[oid]+[cur]
            else: hnet[oid]=[cur]
        sig=evidence_signature(hist,r,fusion_neighbors,hnet,
                              r.get("agent2_detail"),r.get("ml_anomaly_score") or 0.0)
        gate_hint=None
        # Agent 4 sees normalized corroboration polarity internally.
        a4=run_agent4(r,hist,fusion_neighbors,hnet,
                      r.get("agent2_detail") or {},sig,"none",0.0)
        gate_hint=(a4.get("gate") or {}).get("verdict")
        final_type,conf,reason=decide(sig,r.get("agent2_detail"),gate_hint)
        # A low-priority reading with no independent anomaly signature must not
        # be promoted merely because a binary gate has a probabilistic prior.
        if (r.get("screening_flag")=="clean"
            and float(r.get("ml_anomaly_score") or 0.0) < 0.90
            and (sig.get("regional_event_index",0.0) < 0.60 or sig.get("event_isolated_variable",False))
            and not sig.get("frozen_signature",False)
            and not sig.get("isolated_spike",False)
            and not sig.get("drift_signature",False)):
            final_type,conf,reason="none",0.92,"All independent evidence layers are nominal."
        if (final_type=="genuine_event" and float(r.get("ml_anomaly_score") or 0.0) < 0.90
            and not (sig.get("regional_event_index",0.0) >= 0.60 and
                     sig.get("event_corroboration",0.0) >= 0.65 and
                     not sig.get("event_isolated_variable",False))):
            final_type,conf,reason="none",0.92,"Agent-4 probabilistic event prior was not corroborated by the regional multi-hour event shield."
        # Re-run A4 with final type so SHAP/degradation output matches the displayed decision.
        a4=run_agent4(r,hist,fusion_neighbors,hnet,
                      r.get("agent2_detail") or {},sig,final_type,conf)
        r["anomaly_type"]=final_type
        r["confidence_score"]=conf
        r["_fault_field"]=choose_focus_field(sig, final_type)
        r["explanation"]=a4["explanation"]
        r["explanation"]["fusion_decision_reason"]=reason
        r["sensor_health_status"]=a4["sensor_health_status"]
        r["degradation"]=a4.get("degradation")
        r["_fusion_reason"]=reason
        r["_agent4_gate"]=a4.get("gate")
        r["_raw_gate_verdict"]=a4.get("raw_gate_verdict")
        processed.append(r)

    # Agent 5 gets a clean, same-timestamp healthy-neighbor snapshot.
    current_snapshot=incoming
    healthy_ids=[r["station_id"] for r in processed if r["anomaly_type"] in ("none","genuine_event")]
    out=[]
    for r in processed:
        corrected=maybe_correct(r,r["anomaly_type"],state,latest_screening,current_snapshot,healthy_ids,
                                fault_field=r.get("_fault_field"))
        r["corrected_value"]=corrected
        r["alert_severity"]=_severity(r["anomaly_type"],r["confidence_score"],r.get("_agent4_gate"),r["sensor_health_status"])
        # Add an audit-friendly pipeline trace.
        r["pipeline_trace"]={
            "agent1":"ingested",
            "agent2":"real_screening",
            "agent3":r.get("engine","DETERMINISTIC_FALLBACK"),
            "fusion":"evidence_fusion",
            "agent4":"real_gate+SHAP+degradation",
            "agent5":"real_correction+severity",
        }
        r.pop("_agent4_gate",None); r.pop("_raw_gate_verdict",None); r.pop("_fusion_reason",None); r.pop("_fault_field",None)
        out.append(_as_reading(r))
    _latest_screening.update(latest_screening)
    # Commit only AFTER the whole tick has been evaluated.
    for r in out: state.record(r.model_dump())
    return out

def run_pipeline(reading:StationReading)->StationReading:
    return run_tick([reading])[0]

def runtime_status() -> dict:
    """Truthful runtime readiness used by the dashboard and preflight tests."""
    a3 = agent3_status()
    a4 = agent4_status()
    return {
        "agent2": {"available": bool(AGENT2_AVAILABLE), "engine": "REAL_AGENT2" if AGENT2_AVAILABLE else "UNAVAILABLE"},
        "agent3": a3,
        "agent4": a4,
        "agent5": {"available": True, "engine": "TEMPORAL+SPATIAL_CORRECTION"},
        "ready": bool(AGENT2_AVAILABLE and a3.get("available") and a4.get("available")),
    }

@app.get("/")
def root():
    runtime = runtime_status()
    return {"status":"ok" if runtime["ready"] else "degraded",
            "service":"SkyGuard AI Command Center",
            "problem_statement":"SIH26073",
            "runtime":runtime}

@app.get("/health")
def health(): return root()

@app.post("/reset")
def reset():
    global state,_latest_screening
    state=StationState({k:{x:y for x,y in v.items() if x!="name"} for k,v in STATION_META.items()})
    _latest_screening={}
    return {"status":"reset"}

@app.post("/process",response_model=StationReading)
def process(reading:StationReading):
    try:return run_pipeline(reading)
    except Exception as e: raise HTTPException(status_code=500,detail=str(e))

@app.post("/process_batch",response_model=List[StationReading])
def process_batch(readings:List[StationReading]):
    try:return run_tick(readings)
    except Exception as e: raise HTTPException(status_code=500,detail=str(e))

@app.get("/network")
def network():
    return {"stations":[r.model_dump() for r in state.current_snapshot().values()]}

@app.get("/demo/scenarios")
def demo_scenarios():
    return [
      {"id":"normal","label":"Normal network","description":"All stations nominal"},
      {"id":"spike","label":"Sensor spike","description":"One station reports an isolated implausible jump"},
      {"id":"frozen","label":"Frozen sensor","description":"One channel stops changing"},
      {"id":"dropout","label":"Communication dropout","description":"Missing telemetry"},
      {"id":"drift","label":"Sensor drift","description":"Slow calibration bias develops"},
      {"id":"genuine_event","label":"Genuine extreme event","description":"Tauktae weather signal is protected from correction"},
    ]

@app.get("/demo/scenario/{scenario}")
def demo_scenario(scenario: str):
    try:
        from demo_runner import run_scenario
        cfg,results,selected=run_scenario(scenario)
        return {"scenario":cfg,"selected":selected.model_dump(),"network":[r.model_dump() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
