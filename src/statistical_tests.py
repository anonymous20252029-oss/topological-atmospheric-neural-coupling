import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

def run_adf_stationarity_test(series: np.ndarray) -> dict:
    """
    Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF).
    """
    clean_series = series[~np.isnan(series)]
    res = adfuller(clean_series, autolag='AIC')
    return {
        "adf_stat": float(res[0]),
        "p_value": float(res[1]),
        "used_lag": int(res[2]),
        "is_stationary": bool(res[1] < 0.05)
    }

def run_granger_causality_test(predictor: np.ndarray, target: np.ndarray, max_lag: int = 5) -> pd.DataFrame:
    """
    Ước lượng kiểm định nhân quả Granger đa bước trễ (VAR).
    H0: Biến predictor không Granger-cause biến target.
    """
    df = pd.DataFrame({"Target": target, "Predictor": predictor}).dropna()
    res = grangercausalitytests(df[["Target", "Predictor"]], maxlag=max_lag, verbose=False)
    
    rows = []
    for lag in range(1, max_lag + 1):
        f_stat = res[lag][0]["ssr_ftest"][0]
        p_val = res[lag][0]["ssr_ftest"][1]
        chi2 = res[lag][0]["ssr_chi2test"][0]
        rows.append({
            "Lag_Days": lag,
            "F_Statistic": round(f_stat, 4),
            "p_Value": float(f"{p_val:.4e}"),
            "Chi2_Statistic": round(chi2, 4),
            "Significant": bool(p_val < 0.05)
        })
    return pd.DataFrame(rows)

def compute_lead_lag_cross_correlation(x: np.ndarray, y: np.ndarray, max_lags: int = 10) -> tuple:
    """
    Tính toán hàm tương quan chéo (CCF) với các bước trễ thời gian âm và dương.
    """
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-9)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-9)
    
    lags = np.arange(-max_lags, max_lags + 1)
    ccf_vals = []
    
    for lag in lags:
        if lag < 0:
            c = np.corrcoef(x_norm[:lag], y_norm[-lag:])[0, 1]
        elif lag > 0:
            c = np.corrcoef(x_norm[lag:], y_norm[:-lag])[0, 1]
        else:
            c = np.corrcoef(x_norm, y_norm)[0, 1]
        ccf_vals.append(c)
        
    ci_95 = 1.96 / np.sqrt(len(x))
    return lags, np.array(ccf_vals), float(ci_95)

def run_ablation_benchmark(pressure_series: np.ndarray, entropy_h1: np.ndarray, clinical_events: np.ndarray) -> pd.DataFrame:
    """
    Thực hiện kiểm chứng Ablation Study đối sánh giữa:
    1. Linear Amplitude (|dP/dt|)
    2. Spectral Power (FFT/Welch PSD)
    3. Proposed Topological Entropy H1
    """
    # 1. Feature 1: Linear Magnitude
    f_linear = np.abs(np.gradient(pressure_series))
    
    # 2. Feature 2: Spectral Power
    f_spectral = []
    win = 24
    for i in range(len(pressure_series)):
        sub = pressure_series[max(0, i - win): i + 1]
        if len(sub) < 8:
            f_spectral.append(0.0)
        else:
            _, psd = welch(sub, nperseg=min(len(sub), 16))
            f_spectral.append(np.sum(psd))
    f_spectral = np.array(f_spectral)
    
    # 3. Feature 3: Proposed TDA
    f_tda = np.asarray(entropy_h1)
    
    # Nhãn nhị phân: Sự kiện cực đoan (Top 20%)
    threshold = np.quantile(clinical_events, 0.80)
    y_binary = (clinical_events >= threshold).astype(int)
    
    features = [
        ("Model 1: Linear Amplitude", "Derivative |dP/dt|", f_linear),
        ("Model 2: Spectral Fourier", "Power Spectrum (PSD)", f_spectral),
        ("Model 3: Proposed TDA", "Homology Entropy H1", f_tda)
    ]
    
    rows = []
    for name, domain, feat in features:
        # Chuẩn hóa
        norm_f = (feat - np.mean(feat)) / (np.std(feat) + 1e-9)
        norm_y = (clinical_events - np.mean(clinical_events)) / (np.std(clinical_events) + 1e-9)
        
        # Granger Lag 2
        df_tmp = pd.DataFrame({"Y": norm_y, "X": norm_f})
        g_res = grangercausalitytests(df_tmp[["Y", "X"]], maxlag=2, verbose=False)
        f_val = g_res[2][0]["ssr_ftest"][0]
        p_val = g_res[2][0]["ssr_ftest"][1]
        
        # CCF tại lag +2
        lag2_ccf = np.corrcoef(norm_f[:-2], norm_y[2:])[0, 1]
        
        # AUC-ROC
        auc = roc_auc_score(y_binary[2:], norm_f[:-2])
        
        rows.append({
            "Methodological Model": name,
            "Feature Domain": domain,
            "Granger F (Lag 2)": round(f_val, 3),
            "p-Value": float(f"{p_val:.4e}"),
            "Peak CCF (+2d)": round(lag2_ccf, 3),
            "AUC-ROC Lead Time": round(auc, 3)
        })
        
    return pd.DataFrame(rows)
