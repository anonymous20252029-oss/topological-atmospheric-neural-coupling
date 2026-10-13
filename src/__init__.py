"""
Topological Atmospheric-Neural Coupling Core Module.
Provides data ingestion, dynamical reconstruction, persistent homology,
econometric causal inference, and visualization routines.
"""

from .data_loader import fetch_era5_weather, fetch_eeg_cohort
from .embedding import takens_delay_embedding, compute_angular_distance_matrix
from .tda_pipeline import (
    compute_vietoris_rips_homology,
    compute_persistence_entropy,
    compute_wasserstein_distance,
    compute_bottleneck_distance
)
from .statistical_tests import (
    run_adf_stationarity_test,
    run_granger_causality_test,
    compute_lead_lag_cross_correlation,
    run_ablation_benchmark
)
from .visualization import (
    plot_localized_drop_analysis,
    plot_atmospheric_persistence_diagram,
    plot_cortical_network_shift,
    plot_multicity_longitudinal_tda,
    plot_granger_cross_correlation,
    plot_cohort_eeg_distribution
)

__version__ = "1.0.0"
__all__ = [
    "fetch_era5_weather",
    "fetch_eeg_cohort",
    "takens_delay_embedding",
    "compute_angular_distance_matrix",
    "compute_vietoris_rips_homology",
    "compute_persistence_entropy",
    "compute_wasserstein_distance",
    "compute_bottleneck_distance",
    "run_adf_stationarity_test",
    "run_granger_causality_test",
    "compute_lead_lag_cross_correlation",
    "run_ablation_benchmark",
    "plot_localized_drop_analysis",
    "plot_atmospheric_persistence_diagram",
    "plot_cortical_network_shift",
    "plot_multicity_longitudinal_tda",
    "plot_granger_cross_correlation",
    "plot_cohort_eeg_distribution"
]
