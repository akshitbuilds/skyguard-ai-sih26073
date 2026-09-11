"""
Integration test -- NOT a stub, actually exercises the real pipeline
across multiple ticks so StationState has real history/neighbor data
to work with (the single-shot /demo endpoint alone can't prove this,
since it only ever sends one reading with no history behind it).

Run: python3 test_integration.py
"""
from datetime import datetime, timedelta
from schema import StationReading
import main  # reuses the same `state` and `run_pipeline` the real server uses

STATIONS = list(main.STATION_META.keys())
t0 = datetime(2021, 5, 10, 0, 0, 0)

print(f"Agent 2 real pipeline available: {main.AGENT2_AVAILABLE}\n")

# 15 normal hourly ticks across all 5 stations, so real history + neighbor
# state builds up exactly like it would from replay.py
for hour in range(15):
    ts = (t0 + timedelta(hours=hour)).isoformat()
    for i, sid in enumerate(STATIONS):
        meta = main.STATION_META[sid]
        reading = StationReading(
            station_id=sid,
            timestamp=ts,
            latitude=meta["latitude"],
            longitude=meta["longitude"],
            temperature_c=28.0 + i * 0.3 + 0.05 * hour,
            pressure_hpa=1008.0 + i * 0.2,
            humidity_pct=60.0 + i * 0.5,
        )
        main.run_pipeline(reading)

print(f"History built for AWS_DIU: {len(main.state.get_history('AWS_DIU'))} readings")
print(f"Neighbors visible to AWS_DIU: {list(main.state.get_neighbors_current('AWS_DIU').keys())}\n")

# Now inject an obvious fault at AWS_DIU (tick 16) and force anomaly_type
# so Agent 5's REAL correction logic actually triggers (Agent 3/4 are still
# stubs and won't set this on their own yet)
ts = (t0 + timedelta(hours=15)).isoformat()
meta = main.STATION_META["AWS_DIU"]
faulty_reading = StationReading(
    station_id="AWS_DIU",
    timestamp=ts,
    latitude=meta["latitude"],
    longitude=meta["longitude"],
    temperature_c=55.0,   # way outside normal range - simulated spike
    pressure_hpa=1008.0,
    humidity_pct=60.0,
    anomaly_type="sensor_spike",  # normally Agent 4 sets this; forced here for the test
)
result = main.run_pipeline(faulty_reading)

print("=== Result on the faulty AWS_DIU reading ===")
print(f"  screening_flag:      {result.screening_flag}")
print(f"  anomaly_type:        {result.anomaly_type}")
print(f"  raw temperature_c:   55.0  (the injected fault)")
print(f"  corrected_value:     {result.corrected_value}")
if result.corrected_value:
    true_temp = 28.0 + 0*0.3 + 0.05*15  # what the "clean" trend would have given
    print(f"  (expected ~{true_temp:.1f} from trend/neighbors -- corrected value should land near this, not near 55.0)")
