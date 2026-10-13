import os
import requests
import pandas as pd
import numpy as np
import mne

# Tọa độ mặc định 3 trung tâm đô thị Đông Nam Á
COORDINATES = {
    "HCMC": {"latitude": 10.8231, "longitude": 106.6297},
    "Bangkok": {"latitude": 13.7563, "longitude": 100.5018},
    "Singapore": {"latitude": 1.3521, "longitude": 103.8198}
}

STANDARD_16_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'Fz',
    'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz',
    'O1', 'O2', 'Oz', 'T7', 'T8'
]

def fetch_era5_weather(city_name: str, start_date: str = "2024-05-01", end_date: str = "2024-10-31") -> pd.DataFrame:
    """
    Kéo dữ liệu thời tiết tái phân tích ERA5 từ Open-Meteo Historical API.
    """
    if city_name not in COORDINATES:
        raise ValueError(f"City {city_name} not found in preconfigured coordinates.")
        
    lat = COORDINATES[city_name]["latitude"]
    lon = COORDINATES[city_name]["longitude"]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["surface_pressure", "temperature_2m", "precipitation"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "surface_pressure_hPa": hourly["surface_pressure"],
        "temperature_C": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"]
    }).set_index("timestamp")
    
    # Tính đạo hàm áp suất 3 giờ
    df["dP_3h"] = df["surface_pressure_hPa"].diff(3)
    return df

def fetch_eeg_cohort(n_subjects: int = 10, run_id: int = 1) -> dict:
    """
    Tải và trích xuất dữ liệu ma trận tín hiệu điện não 16 kênh từ PhysioNet EEGBCI.
    """
    cohort_data = {}
    for sub_id in range(1, n_subjects + 1):
        try:
            files = mne.datasets.eegbci.load_data(subjects=sub_id, runs=[run_id], update_path=True, verbose=False)
            raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
            
            # Khớp kênh tín hiệu sẵn có
            available = [ch for ch in raw.ch_names if any(k in ch for k in STANDARD_16_CHANNELS)]
            raw.pick(available[:16])
            sfreq = int(raw.info['sfreq'])
            
            # Trích xuất 2 trạng thái: Cơ sở (0-20s) và Kích hoạt (20-40s)
            base_signals = raw.get_data(start=0, stop=sfreq * 20)
            pert_signals = raw.get_data(start=sfreq * 20, stop=sfreq * 40)
            
            cohort_data[f"S{sub_id:03d}"] = {
                "baseline": base_signals,
                "altered": pert_signals,
                "sfreq": sfreq
            }
        except Exception as e:
            print(f"Warning: Subject {sub_id} failed to load: {e}")
            
    return cohort_data
