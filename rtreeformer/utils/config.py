"""
rtreeformer.utils.config
========================
Centralised hyperparameter registry - single source of truth.

All model sizes and training hyperparameters are defined here so that every
experiment (notebook cell or standalone script) references the same constants.

Usage::

    from rtreeformer.utils.config import MODEL_SIZES, TRAIN_CFG, LOOKBACK, NUM_GRIDS

    size    = MODEL_SIZES["large"]
    d_model = size["d_model"]          # 64
    cfg     = TRAIN_CFG               # lr, max_epochs, patience, clip
"""

# ---------------------------------------------------------------------------
# Data / sequence constants
# ---------------------------------------------------------------------------
LOOKBACK: int   = 12    # look-back window length (hours)
NUM_GRIDS: int  = 10   # number of grids randomly sampled from the clean data
BATCH_SIZE: int = 32    # mini-batch size for all DataLoaders
K_ROOTS: int    = 4     # number of root spatial partitions (AgglomerativeClustering)

# ---------------------------------------------------------------------------
# Fixed training hyperparameters - identical for every model
# ---------------------------------------------------------------------------
TRAIN_CFG: dict = {
    "lr":         1e-3,
    "max_epochs": 10,
    "patience":   10,
    "clip":       1.0,
}

# ---------------------------------------------------------------------------
# Model architecture sizes
#
#   Rule: the same size key -> the same hidden_dim / d_model regardless of
#         model type.  e.g. "base" always means d_model=32, hidden_dim=32.
#
#   Shared keys across architectures
#   ---------------------------------
#   hidden_dim  : LSTM hidden-state width
#   d_model     : Transformer token width (PatchTST, Reformer, R-Treeformer)
#   d_k         : attention key/query dim  (R-Treeformer only)
#   num_layers  : Transformer encoder depth (PatchTST, Reformer, R-Treeformer)
#   nhead       : number of attention heads  (d_model / 8)
#   d_ff        : FFN inner dim (Reformer)   (4 x d_model)
#   Canonical size for plots / tables
#   ----------------------------------
#   Always use MODEL_SIZES["large"] (LARGE alias) when generating result
#   tables, ablation plots, or any single-representative comparison.
# ---------------------------------------------------------------------------
MODEL_SIZES: dict = {
    "small": {
        "hidden_dim": 16,
        "d_model":    16,
        "d_k":         8,
        "num_layers":  1,
        "nhead":       2,
        "d_ff":       64,
    },
    "base": {
        "hidden_dim": 32,
        "d_model":    32,
        "d_k":        16,
        "num_layers":  2,
        "nhead":       4,
        "d_ff":      128,
    },
    "large": {
        "hidden_dim": 64,
        "d_model":    64,
        "d_k":        32,
        "num_layers":  4,
        "nhead":       8,
        "d_ff":      256,
    },
}

# Convenience aliases
SMALL = MODEL_SIZES["small"]
BASE  = MODEL_SIZES["base"]
LARGE = MODEL_SIZES["large"]

