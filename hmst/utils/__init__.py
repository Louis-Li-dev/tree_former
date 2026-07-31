"""
hmst.utils
==========
Utility modules: configuration, evaluation metrics, and R-Tree mask builders.
"""

from .config  import MODEL_SIZES, TRAIN_CFG, LOOKBACK, NUM_GRIDS, BATCH_SIZE, K_ROOTS
from .config  import SMALL, BASE, LARGE
from .metrics import calculate_metrics, mae, rmse, wmape, fast_dtw_distance
from .masks   import build_forest_masks

__all__ = [
    # config
    "MODEL_SIZES", "TRAIN_CFG", "LOOKBACK", "NUM_GRIDS", "BATCH_SIZE", "K_ROOTS",
    "SMALL", "BASE", "LARGE",
    # metrics
    "calculate_metrics", "mae", "rmse", "wmape", "fast_dtw_distance",
    # masks
    "build_forest_masks",
]
