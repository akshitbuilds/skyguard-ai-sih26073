import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# List of 5 stations with exact coordinates
STATIONS = [
    {"station_id": "AWS_DIU",       "lat": 20.7141, "lon": 70.9822},
    {"station_id": "AWS_VERAVAL",   "lat": 20.9077, "lon": 70.3679},
    {"station_id": "AWS_MAHUVA",    "lat": 21.0901, "lon": 71.7690},
    {"station_id": "AWS_PORBANDAR", "lat": 21.6422, "lon": 69.6093},
    {"station_id": "AWS_BHAVNAGAR", "lat": 21.7629, "lon": 72.1533},
]

url = "https://archive-api.open-meteo.com/v1/archive"
all_station_dfs = []

# NOTE: extended from a single month to a full year so the model actually
# sees seasonal variation (summer / monsoon / winter), not just May.
# This window deliberately still includes 16-18 May 2021, when Cyclone
# Tauktae made landfall near Una/Diu -- see PROTECTED_EVENTS in
# fault_injector.py, which uses this real event as a genuine-event
# validation case instead of treating it as ordinary data.
for station in STATIONS:
    print(f"Fetching data for {station['station_id']}...")

    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure"],
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    temps = hourly.Variables(0).ValuesAsNumpy()
    humidities = hourly.Variables(1).ValuesAsNumpy()
    pressures = hourly.Variables(2).ValuesAsNumpy()

    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Column names match schema.json used by fault_injector.py and replay.py
    df_station = pd.DataFrame({
        "station_id": station["station_id"],
        "timestamp": dates.strftime('%Y-%m-%dT%H:%M:%S'),
        "temperature_c": [round(t, 2) for t in temps],
        "pressure_hpa": [round(p, 2) for p in pressures],
        "humidity_pct": [round(h, 2) for h in humidities],
        "latitude": station["lat"],
        "longitude": station["lon"]
    })

    all_station_dfs.append(df_station)

# Combine into single DataFrame
final_dataframe = pd.concat(all_station_dfs, ignore_index=True)

# Save as clean_dataset.csv so fault_injector.py can read it directly
output_csv_path = "clean_dataset.csv"
final_dataframe.to_csv(output_csv_path, index=False)

print(f"\nSuccessfully generated {output_csv_path} with {len(final_dataframe)} total rows.")
print("Ready for fault_injector.py.")
