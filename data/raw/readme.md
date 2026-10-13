# Raw Data Directory (`data/raw/`)

This directory contains pristine, unmodified source datasets and acquisition protocols supporting the findings of the study:
> **"Topological Characterization of Micro-Atmospheric Phase Transitions and Their Cross-Scale Coupling to Neural Network Dynamics in Tropical Urban Environments"**

In accordance with open-science and reproducible research standards, all files archived here are maintained in read-only form and serve as direct inputs for the preprocessing and feature-extraction pipelines in `src/`.

---

## 1. Directory Structure

```text
data/raw/
├── README.md                      # Data provenance & technical specifications (this file)
└── hcmc_recent_weather_48h.csv    # 48-hour localized observational verification sample
```
---
## 2. Dataset Descriptions & Provenance

### A. Atmospheric Surface Reanalysis (ECMWF ERA5)
* **Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF), retrieved via the Copernicus Climate Data Store and the Open-Meteo Historical Weather API.
* **Temporal Horizon:** Continuous hourly time series spanning **May 1, 2024 to October 31, 2024** ($N = 4,416$ consecutive hours per city, capturing the full 2024 Southwest Monsoon season).
* **Target Coordinates:**
  * **Ho Chi Minh City, Vietnam:** $10.8231^\circ\text{ N}, 106.6297^\circ\text{ E}$
  * **Bangkok, Thailand:** $13.7563^\circ\text{ N}, 100.5018^\circ\text{ E}$
  * **Singapore:** $1.3521^\circ\text{ N}, 103.8198^\circ\text{ E}$
* **Key Variables:**
  * `surface_pressure_hPa`: Hourly atmospheric surface pressure ($\text{hPa}$).
  * `temperature_C`: Hourly 2-meter ambient dry-bulb temperature ($^\circ\text{C}$).
  * `precipitation_mm`: Total hourly convective and stratiform precipitation ($\text{mm}$).

### B. Human Electrophysiology (PhysioNet EEGBCI)
* **Provider:** [PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/).
* **Hardware & Acquisition Montage:** 64-channel scalp montage following the international 10–20 system, sampled at 160 Hz via BCI2000 acquisition hardware.
* **Cohort Scope:** $N = 10$ independent human subjects (`S001` through `S010`).
* **Selected 16-Channel Configuration for Functional Metric Graphs:**
  * **Frontal:** `Fp1`, `Fp2`, `F3`, `F4`, `Fz`
  * **Central & Parietal:** `C3`, `C4`, `Cz`, `P3`, `P4`, `Pz`
  * **Occipital & Temporal:** `O1`, `O2`, `Oz`, `T7`, `T8`
---

## 3. Sample File Specification: `hcmc_recent_weather_48h.csv`

This file provides a lightweight, standalone 48-hour observational benchmark recorded during an acute barometric depressurization event in Ho Chi Minh City (October 12–13, 2024):

* **Record Count:** 48 consecutive hourly records.
* **Schema:**
  * `timestamp`: ISO-8601 formatted datetime string (`YYYY-MM-DD HH:MM:SS`).
  * `surface_pressure_hPa`: Observed surface atmospheric pressure ($\text{hPa}$).
  * `temperature_C`: Observed ambient temperature ($^\circ\text{C}$).
* **Intended Purpose:** Serves as an instant smoke-test benchmark for visualization routines and the interactive `app.py` dashboard, regenerating **Figure 1** (Localized Drop Analysis) and **Figure 2** (Atmospheric Attractor Homology) without requiring multi-gigabyte network downloads.

---

## 4. Automated Acquisition & Verification

To programmatically retrieve the full 4,416-hour multi-city ERA5 reanalysis records and the 10-subject PhysioNet EDF files, run the following command from the repository root:

```bash
python src/data_loader.py --fetch-all --output-dir data/raw/
