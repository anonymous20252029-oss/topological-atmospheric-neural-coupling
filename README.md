# Topological Characterization of Micro-Atmospheric Phase Transitions and Their Cross-Scale Coupling to Neural Network Dynamics

Official repository containing the complete implementation, data processing pipelines, and interactive visualization tools for the paper:
> **"Topological Characterization of Micro-Atmospheric Phase Transitions and Their Cross-Scale Coupling to Neural Network Dynamics in Tropical Urban Environments"**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://topological-atmospheric-neural-coupling-cfigtkeee6rpmahf4y88iu.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview
This study proposes a cross-scale mathematical framework combining **Topological Data Analysis (TDA)**, **Takens' dynamical embedding**, and **Vector Autoregressive (VAR) Granger Causality** to investigate the biophysical relationship between acute barometric depressurization and neurological discomfort, systematically refuting non-ionizing electromagnetic radiation (RF-EMF 5G/6G) myths.

### Key Highlights:
- **Atmospheric Phase Space TDA:** Longitudinal hourly ERA5 reanalysis across Ho Chi Minh City, Bangkok, and Singapore ($N = 4,416$ hours per city).
- **Cortical Connectome Topology:** Real 64-channel human EEG from PhysioNet evaluated across an independent cohort ($N = 10$).
- **Predictive Causal Coupling:** Persistent Homology Entropy ($H_1$) Granger-causes clinical surges with a 48-hour lead time ($F = 21.02, p < 0.001$, Peak $\text{CCF} = +0.43$).

---

## 🚀 Live Interactive Demo
An interactive web application is deployed via Streamlit Community Cloud:
👉 **[Access the Online Dashboard](https://topological-atmospheric-neural-coupling-cfigtkeee6rpmahf4y88iu.streamlit.app/)**

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)anonymous20252029-oss/topological-atmospheric-neural-coupling.git
   cd topological-atmospheric-neural-coupling
    ```
    1.1. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
     ```

   1.2. Run the Streamlit Dashboard locally:
   ```bash
      streamlit run app.py
    ```
---
📊 Data Availability

Meteorological Data: Publicly accessed via Copernicus Climate Data Store / Open-Meteo Historical API.
Electrophysiological Data: Freely accessible through the PhysioNet EEG Motor Movement/Imagery Database.

---

📖 Citation

If you use this code or methodology in your research, please cite our paper:
```bash
@article{,
  title={Topological Characterization of Micro-Atmospheric Phase Transitions and Their Cross-Scale Coupling to Neural Network Dynamics in Tropical Urban Environments},
  author={},
  journal={International Journal of Biometeorology},
  year={2026}
}
```
---
