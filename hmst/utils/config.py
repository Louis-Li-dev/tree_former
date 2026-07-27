"""
hmst.utils.config
=================
Centralised hyperparameter registry.

All model sizes and training hyperparameters are defined here so that every
experiment references the same constants — a single source of truth.

Usage::

    from hmst.utils.config import MODEL_SIZES, TRAIN_CFG, LOOKBACK

    size   = MODEL_SIZES["large"]
    d_model = size["d_model"]          # 64
    cfg     = TRAIN_CFG               # lr, max_epochs, patience, clip
"""

# ---------------------------------------------------------------------------
# Data / sequence constants
# ---------------------------------------------------------------------------
LOOKBACK: int = 12          # look-back window (hours)
NUM_GRIDS: int = 500        # total spatial grid cells (20 × 25)
GRID_H: int = 20
GRID_W: int = 25
BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Fixed training hyperparameters — identical for every model
# ---------------------------------------------------------------------------
TRAIN_CFG: dict = {
    "lr":         1e-3,
    "max_epochs": 100,
    "patience":   10,
    "clip":       1.0,
}

# ---------------------------------------------------------------------------
# Model architecture sizes
#
#   Rule: same SIZE KEY → same hidden_dim / d_model regardless of model type.
#         e.g. "base" always means d_model=32, hidden_dim=32.
#
#   Keys shared across architectures
#   ---------------------------------
#   hidden_dim  : LSTM / ConvLSTM hidden state width
#   d_model     : Transformer token width (PatchTST, HMST)
#   d_k         : Attention key/query dim  (HMST only)
#   num_layers  : Transformer depth (PatchTST, HMST)
#   nhead       : Transformer attention heads (PatchTST) — d_model / 8
# ---------------------------------------------------------------------------
MODEL_SIZES: dict = {
    "small": {
        "hidden_dim": 16,
        "d_model":    16,
        "d_k":         8,
        "num_layers":  1,
        "nhead":       2,
    },
    "base": {
        "hidden_dim": 32,
        "d_model":    32,
        "d_k":        16,
        "num_layers":  2,
        "nhead":       4,
    },
    "large": {
        "hidden_dim": 64,
        "d_model":    64,
        "d_k":        32,
        "num_layers":  4,
        "nhead":       8,
    },
}

# Convenience aliases
SMALL = MODEL_SIZES["small"]
BASE  = MODEL_SIZES["base"]
LARGE = MODEL_SIZES["large"]
