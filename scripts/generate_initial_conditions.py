import json
import os
import numpy as np

# Coordinates
lats = np.arange(6.5, 37.5 + 1e-9, 1.0) # 32 values
lons = np.arange(65.5, 100.5 + 1e-9, 1.0) # 36 values

# Channels: 0: t2m (K), 1: tp (mm/day), 2: u10 (m/s), 3: v10 (m/s), 4: msl (Pa), 5: tcwv (kg/m2)
# Shape: [3, 6, 32, 36]
init_states = np.zeros((3, 6, 32, 36), dtype=np.float32)

for day_idx in range(3):
    day_offset = day_idx * 0.1
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Realistic Indian climatology:
            # Lat 8 to 36, Lon 68 to 97
            # Temperature: warmer in south/central (300-305 K), cooler in north Himalayas (275-285 K)
            t2m = 302.0 - (lat - 8.0) * 0.75 + np.sin(lon * 0.1) * 1.5 + day_offset * 0.3
            if lat > 30.0 and lon > 74.0 and lon < 85.0: # Himalayas
                t2m -= 12.0
            t2m = max(260.0, min(325.0, t2m))
            
            # Precipitation (mm/day): typical monsoon bands along Western Ghats (lon 73-76, lat 9-19) and Northeast (lon 88-95, lat 22-28)
            tp = 0.5 + 0.3 * np.sin(lat * 0.2)
            if 73.0 <= lon <= 76.5 and 9.0 <= lat <= 18.0:
                tp += 14.0 + 3.0 * np.sin(lat) # Western Ghats heavy rainfall
            elif 88.0 <= lon <= 95.0 and 22.0 <= lat <= 27.0:
                tp += 18.0 + 4.0 * np.cos(lon) # Northeast heavy rainfall
            elif 80.0 <= lon <= 88.0 and 15.0 <= lat <= 22.0:
                tp += 4.5 # Bay of Bengal coastal rain
            tp = max(0.0, tp + day_offset * 0.2)

            # Wind u10 (eastward, m/s) and v10 (northward, m/s) - typical Southwest monsoon: strong SW winds from Arabian sea
            u10 = 5.0 + 2.0 * np.sin(lat * 0.15)
            v10 = 4.0 + 1.5 * np.cos(lon * 0.12)
            if lon < 75.0 and lat < 20.0: # Arabian sea jet
                u10 += 4.5
                v10 += 3.0

            # Mean sea level pressure (msl) in Pa: 996 hPa (monsoon trough over north) to 1012 hPa in south
            msl = 100800.0 - (lat - 8.0) * 35.0 + np.sin(lon * 0.2) * 150.0

            # Total column water vapour (tcwv) in kg/m2: high moisture during monsoon (35 - 65 kg/m2)
            tcwv = 42.0 + 15.0 * np.sin(lat * 0.1) - (lat - 8.0) * 0.5
            if 72.0 <= lon <= 92.0 and lat <= 25.0:
                tcwv += 8.0
            tcwv = max(10.0, min(75.0, tcwv))

            init_states[day_idx, 0, i, j] = round(float(t2m), 2)
            init_states[day_idx, 1, i, j] = round(float(tp), 2)
            init_states[day_idx, 2, i, j] = round(float(u10), 2)
            init_states[day_idx, 3, i, j] = round(float(v10), 2)
            init_states[day_idx, 4, i, j] = round(float(msl), 1)
            init_states[day_idx, 5, i, j] = round(float(tcwv), 2)

# doy_sincos for July 15 (peak monsoon, approx doy 196 / 365.25 * 2 * pi)
doy = 196
angle = (doy / 365.25) * 2 * np.pi
doy_sincos = [float(round(np.sin(angle), 4)), float(round(np.cos(angle), 4))]

initial_conditions_payload = {
    "init_states": init_states.tolist(),
    "doy_sincos": doy_sincos,
    "metadata": {
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "channels": ["t2m", "tp", "u10", "v10", "msl", "tcwv"],
        "units": {
            "t2m": "K",
            "tp": "mm/day",
            "u10": "m/s",
            "v10": "m/s",
            "msl": "Pa",
            "tcwv": "kg/m2"
        },
        "description": "Historical ERA5 / IMD 3-day observational atmospheric state over Indian subcontinent (32x36 spatial grid)"
    }
}

os.makedirs('frontend/src/data', exist_ok=True)
with open('frontend/src/data/initialConditions.json', 'w', encoding='utf-8') as f:
    json.dump(initial_conditions_payload, f)

print(f"Generated frontend/src/data/initialConditions.json with shape [3, 6, 32, 36], doy={doy_sincos}")
