import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams

def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

def plot_localized_drop_analysis(df_48h: pd.DataFrame, output_path: str):
    """Xuất Figure 1: Biểu đồ sụt áp cục bộ và nhiệt độ tại TP.HCM."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Trục 1: Tốc độ sụt áp 3h
    bars = ax1.bar(df_48h.index.strftime('%m-%d %H:00'), df_48h['dP_3h'], color='#e67e22', width=0.5)
    for idx, bar in enumerate(bars):
        if df_48h['dP_3h'].iloc[idx] <= -2.0:
            bar.set_color('#c0392b')
            
    ax1.axhline(-2.0, color='#880000', linestyle='--', linewidth=1.5, label='Clinical Trigger Threshold (-2.0 hPa/3h)')
    ax1.set_ylabel("3h Pressure Change: dP/dt (hPa)", fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True)
    
    # Trục 2: Áp suất bề mặt và nhiệt độ nghịch biến
    ax2_twin = ax2.twinx()
    ax2.plot(df_48h.index.strftime('%m-%d %H:00'), df_48h['surface_pressure_hPa'], color='#2980b9', marker='o', linewidth=2, label='Surface Pressure')
    ax2_twin.plot(df_48h.index.strftime('%m-%d %H:00'), df_48h['temperature_C'], color='#d35400', marker='s', linestyle='--', linewidth=2, label='Temperature')
    
    ax2.set_ylabel("Pressure (hPa)", color='#2980b9', fontweight='bold')
    ax2_twin.set_ylabel("Temperature (°C)", color='#d35400', fontweight='bold')
    ax2.set_xticklabels(df_48h.index.strftime('%m-%d %H:00'), rotation=30, ha='right')
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_atmospheric_persistence_diagram(diagrams: list, h0_pe: float, h1_pe: float, output_path: str):
    """Xuất Figure 2: Biểu đồ Persistence Diagram của khí quyển."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_diagrams(diagrams, ax=ax, legend=True, show=False)
    ax.set_title(f"Figure 2: Persistence Diagram of Atmospheric Attractor\nEntropy H0={h0_pe:.2f}, H1={h1_pe:.2f}", fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cortical_network_shift(dist_matrix: np.ndarray, dgm_base: list, dgm_pert: list, output_path: str):
    """Xuất Figure 3: Ma trận kết nối và dịch chuyển chu trình H1 não bộ."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Trái: Ma trận khoảng cách
    im = ax1.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax1.set_title("Brain Functional Distance Matrix D=sqrt(2(1-R))", fontweight='bold', fontsize=10)
    ax1.set_xlabel("EEG Channel Index")
    ax1.set_ylabel("EEG Channel Index")
    fig.colorbar(im, ax=ax1)
    
    # Phải: Dịch chuyển vòng H1
    if len(dgm_base[1]) > 0:
        ax2.scatter(dgm_base[1][:, 0], dgm_base[1][:, 1], color='#2980b9', s=60, label='Baseline State (H1)')
    if len(dgm_pert[1]) > 0:
        ax2.scatter(dgm_pert[1][:, 0], dgm_pert[1][:, 1], color='#e74c3c', marker='^', s=60, label='Altered State (H1)')
        
    ax2.plot([0, 1.2], [0, 1.2], 'k--', alpha=0.5)
    ax2.set_xlim([-0.05, 1.1])
    ax2.set_ylim([-0.05, 1.1])
    ax2.set_title("Cortical Topological Shift", fontweight='bold', fontsize=10)
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Death")
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_multicity_longitudinal_tda(entropy_dict: dict, dist_matrix_df: pd.DataFrame, output_path: str):
    """Xuất Figure 4: Quỹ đạo entropy đa đô thị và ma trận Wasserstein liên thành phố."""
    set_publication_style()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
    
    # Trên: Chuỗi thời gian entropy
    ax1 = fig.add_subplot(gs[0])
    colors = {"HCMC": "#2980b9", "Bangkok": "#e67e22", "Singapore": "#27ae60"}
    for city, s in entropy_dict.items():
        ax1.plot(s.index, s.values, label=f"{city} (H1 Entropy)", color=colors.get(city, '#333333'), alpha=0.85)
    ax1.set_title("Longitudinal Topological Entropy (H1) Trajectory (May - Oct 2024 Monsoon)", fontweight='bold')
    ax1.set_ylabel("Persistence Entropy H1")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Dưới: Ma trận khoảng cách
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(dist_matrix_df.values, cmap='YlOrRd')
    ax2.set_xticks(range(len(dist_matrix_df.columns)))
    ax2.set_yticks(range(len(dist_matrix_df.index)))
    ax2.set_xticklabels(dist_matrix_df.columns)
    ax2.set_yticklabels(dist_matrix_df.index)
    
    for i in range(len(dist_matrix_df.index)):
        for j in range(len(dist_matrix_df.columns)):
            ax2.text(j, i, f"{dist_matrix_df.iloc[i, j]:.3f}", ha="center", va="center", color="black" if dist_matrix_df.iloc[i, j] < 5.0 else "white", fontweight='bold')
            
    ax2.set_title("Inter-City Topological Wasserstein Distance Matrix (H1 Space)", fontweight='bold')
    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_granger_cross_correlation(entropy_series: pd.Series, events_series: pd.Series, lags: np.ndarray, ccf_vals: np.ndarray, ci_95: float, output_path: str):
    """Xuất Figure 5: Causal Granger Co-movement và hàm tương quan chéo CCF."""
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
    
    # Trên: Co-movement
    ax1_twin = ax1.twinx()
    ax1.plot(entropy_series.index, entropy_series.values, color='#2980b9', linewidth=1.5, label='Atmospheric H1 Entropy (HCMC)')
    ax1_twin.plot(events_series.index, events_series.values, color='#c0392b', linestyle='--', linewidth=1.2, label='Neurological / Headache Events')
    ax1.set_ylabel("Persistence Entropy H1", color='#2980b9', fontweight='bold')
    ax1_twin.set_ylabel("Daily Clinical Events", color='#c0392b', fontweight='bold')
    ax1.set_title("Co-movement of Atmospheric Topological Entropy and Neurological Events", fontweight='bold')
    ax1.grid(True)
    
    # Dưới: Stem plot CCF
    markerline, stemlines, baseline = ax2.stem(lags, ccf_vals)
    plt.setp(markerline, color='#2c3e50', markersize=6)
    plt.setp(stemlines, color='#7f8c8d', linewidth=1.5)
    plt.setp(baseline, color='black', linewidth=1.0)
    
    ax2.axhline(ci_95, color='#e74c3c', linestyle=':', label=f'95% CI (±{ci_95:.2f})')
    ax2.axhline(-ci_95, color='#e74c3c', linestyle=':')
    ax2.set_xlabel("Time Delay (Lags - Days: Entropy H1 leads Clinical Events)", fontweight='bold')
    ax2.set_ylabel("Cross-Correlation (CCF)", fontweight='bold')
    ax2.set_title("Cross-Correlation Function (CCF): Lead-Lag Coupling Dynamics (Peak at Lag +2)", fontweight='bold')
    ax2.set_xticks(lags)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_cohort_eeg_distribution(df_cohort: pd.DataFrame, output_path: str):
    """Xuất Figure 6: Phân phối biến dạng hình học mạng não trên thuần tập N=10."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_cohort[["W1_H0", "W1_H1", "Bottleneck_H1"]].boxplot(
        ax=ax, patch_artist=True,
        boxprops=dict(facecolor="#e8f4f8", color="#2980b9", linewidth=1.5),
        medianprops=dict(color="#c0392b", linewidth=2.0)
    )
    ax.set_title(f"Figure 6: Cortical Network Topological Distances Across Cohort (N = {len(df_cohort)})", fontweight='bold')
    ax.set_ylabel("Metric Distance Value", fontweight='bold')
    ax.set_xticklabels(["Wasserstein W1(H0)", "Wasserstein W1(H1)", "Bottleneck dB(H1)"], fontweight='bold')
    ax.grid(True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
