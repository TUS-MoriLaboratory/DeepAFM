from dataclasses import dataclass

# =========================
# Structure Dependency
# =========================

@dataclass
class StructureDependencyConfig:
    morphing_start = "AF2Closed" # AF2Closed, 2ipc
    morphing_end = "2ipc"     # AF2Closed, 2ipc
    num_steps: int = 24


# -------------------------
# UMAP Configuration
# -------------------------
@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"    # euclidean or cosine
    spread: float = 1.0       # The effective scale of embedded points. In combination with min_dist, this will determine how clustered/clumped the embedded points are.
    use_gpu: bool = False     # Whether to use GPU (cuML) for UMAP