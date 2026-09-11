
"""Deterministic screening-round scenarios built from Agent 1's real dataset."""
from __future__ import annotations
import os, sys
import pandas as pd

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATA=os.path.join(ROOT,"agent1_ingestion")
SCENARIOS={
    "normal":{"station_id":"AWS_DIU","timestamp":"2021-08-01T00:00:00","label":"NORMAL NETWORK",
              "description":"All five stations behave normally; no correction is issued."},
    "spike":{"station_id":"AWS_VERAVAL","timestamp":"2021-01-01T06:00:00","label":"ISOLATED SENSOR SPIKE",
             "description":"Veraval humidity jumps ~25 points while neighbors do not; the raw value is replaced by a corrected estimate."},
    "frozen":{"station_id":"AWS_BHAVNAGAR","timestamp":"2021-04-19T10:00:00","label":"FROZEN SENSOR",
              "description":"Bhavnagar humidity has been flat for many samples; SkyGuard identifies a stuck channel."},
    "dropout":{"station_id":"AWS_PORBANDAR","timestamp":"2021-11-29T05:00:00","label":"COMMUNICATION DROPOUT",
               "description":"All three telemetry channels disappear; the event is critical and values are imputed."},
    "drift":{"station_id":"AWS_MAHUVA","timestamp":"2021-04-23T23:00:00","label":"CALIBRATION DRIFT",
             "description":"A slow pressure bias separates from the network baseline; maintenance is recommended."},
    "genuine_event":{"station_id":"AWS_DIU","timestamp":"2021-05-16T18:00:00","label":"GENUINE EXTREME WEATHER",
                     "description":"Cyclone Tauktae-era regional transition is corroborated across stations; correction is blocked."},
}

def _rows_for_tick(df,ts):
    g=df[df["ts"]==pd.Timestamp(ts)].sort_values("station_id")
    return g

def run_scenario(name:str):
    from backend_integration import main
    from backend_integration.schema import StationReading
    run_tick=main.run_tick
    if name not in SCENARIOS: raise ValueError(f"Unknown scenario: {name}")
    # reset backend state using the public endpoint implementation
    main.reset()
    cfg=SCENARIOS[name]
    df=pd.read_csv(os.path.join(DATA,"faulty_dataset.csv"))
    df["ts"]=pd.to_datetime(df["timestamp"])
    t=pd.Timestamp(cfg["timestamp"])
    # Seed 7 days of prior stream without running agents; the real incoming
    # tick is still evaluated through the complete five-agent pipeline.
    seed_start=t-pd.Timedelta(days=7)
    seed=df[(df.ts>=seed_start)&(df.ts<t)]
    records=[]
    for _,row in seed.sort_values(["ts","station_id"]).iterrows():
        records.append({
            "station_id":row.station_id,"timestamp":row.timestamp,
            "latitude":float(row.latitude),"longitude":float(row.longitude),
            "temperature_c":float(row.temperature_c) if pd.notna(row.temperature_c) else None,
            "pressure_hpa":float(row.pressure_hpa) if pd.notna(row.pressure_hpa) else None,
            "humidity_pct":float(row.humidity_pct) if pd.notna(row.humidity_pct) else None,
        })
    main.state.seed(records)
    # Keep current station readings at t-1 while evaluating the simultaneous target tick.
    rows=_rows_for_tick(df,t)
    readings=[]
    for _,row in rows.iterrows():
        readings.append(StationReading(
            station_id=row.station_id,timestamp=row.timestamp,
            latitude=float(row.latitude),longitude=float(row.longitude),
            temperature_c=float(row.temperature_c) if pd.notna(row.temperature_c) else None,
            pressure_hpa=float(row.pressure_hpa) if pd.notna(row.pressure_hpa) else None,
            humidity_pct=float(row.humidity_pct) if pd.notna(row.humidity_pct) else None,
        ))
    results=run_tick(readings)
    selected=next(r for r in results if r.station_id==cfg["station_id"])
    return cfg, results, selected
