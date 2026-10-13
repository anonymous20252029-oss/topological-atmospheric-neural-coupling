import numpy as np
from ripser import ripser
from persim import wasserstein, bottleneck

def compute_vietoris_rips_homology(data: np.ndarray, is_distance_matrix: bool = False, max_dim: int = 1) -> list:
    """
    Tính phức đơn Vietoris-Rips và trích xuất Persistence Diagrams (H0 và H1).
    """
    res = ripser(data, distance_matrix=is_distance_matrix, maxdim=max_dim)
    return res['dgms']

def compute_persistence_entropy(diagram: np.ndarray) -> float:
    """
    Tính toán Persistence Entropy (PE) cho một biểu đồ đồng điều H_p:
    PE = -sum(p_k * log2(p_k)), với p_k = (death - birth) / L.
    """
    if len(diagram) == 0:
        return 0.0
        
    # Loại bỏ các điểm có thời gian chết vô hạn (infinite death)
    finite_pts = diagram[np.isfinite(diagram[:, 1])]
    if len(finite_pts) == 0:
        return 0.0
        
    lifespans = finite_pts[:, 1] - finite_pts[:, 0]
    total_length = np.sum(lifespans)
    
    if total_length == 0.0:
        return 0.0
        
    probs = lifespans / total_length
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)

def compute_wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    Tính khoảng cách Wasserstein W_1 giữa hai biểu đồ persistence.
    Tự động xử lý và loại bỏ các điểm vô hạn để tránh lỗi tràn/cảnh báo.
    """
    d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))
    return float(wasserstein(d1, d2))

def compute_bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    Tính khoảng cách Bottleneck d_B giữa hai biểu đồ persistence.
    """
    d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))
    return float(bottleneck(d1, d2))
