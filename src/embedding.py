import numpy as np

def takens_delay_embedding(time_series: np.ndarray, dimension: int = 3, delay: int = 2) -> np.ndarray:
    """
    Tái cấu trúc đa tạp hút (attractor manifold) theo định lý nhúng Takens:
    X(t) = [P(t), P(t+tau), ..., P(t+(d-1)tau)]
    """
    ts = np.asarray(time_series, dtype=float)
    # Chuẩn hóa Z-score
    ts_norm = (ts - np.nanmean(ts)) / (np.nanstd(ts) + 1e-9)
    
    n_points = len(ts_norm) - (dimension - 1) * delay
    if n_points <= 0:
        raise ValueError("Series length is too short for the chosen dimension and delay.")
        
    embedded = np.empty((n_points, dimension), dtype=float)
    for d in range(dimension):
        embedded[:, d] = ts_norm[d * delay : d * delay + n_points]
        
    return embedded

def compute_angular_distance_matrix(signals: np.ndarray) -> np.ndarray:
    """
    Tính ma trận khoảng cách góc thỏa mãn bất đẳng thức tam giác:
    D_ij = sqrt(2 * (1 - R_ij)) từ ma trận tương quan Pearson R.
    """
    corr = np.corrcoef(signals)
    # Ràng buộc khoảng giá trị [-1, 1] để tránh số âm dưới căn bậc hai do sai số float
    corr = np.clip(corr, -1.0, 1.0)
    dist_matrix = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix
