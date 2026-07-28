"""
hmst.model
==========
All model architectures used in the paper:

Proposed
--------
    HMSTv2          — Hierarchical Masked Spatial Transformer v2
    TimeEmbedding   — Hour-of-day + Day-of-week embedding
    HierarchicalBlock — One Transformer block with R-Tree masked attention

Baselines
---------
    RevIN           — Reversible Instance Normalization (shared utility)
    EachGridLSTM    — Per-grid independent LSTM (vectorised)
    AllGridLSTM     — Shared LSTM across all grids
    AllGridConvLSTM — Spatial ConvLSTM on 20 × 25 grid
    AllGridDLinear  — DLinear (AAAI 2023, LTSF-Linear)
    AllGridPatchTST — PatchTST (ICLR 2023, channel-independent Transformer)
"""

from .hmstv2    import TimeEmbedding, HierarchicalBlock, HMSTv2, RevIN
from .baselines import (
    EachGridLSTM,
    AllGridLSTM,
    AllGridConvLSTM,
    AllGridDLinear,
    AllGridPatchTST,
)

__all__ = [
    # proposed
    "TimeEmbedding", "HierarchicalBlock", "HMSTv2",
    # baselines
    "RevIN", "EachGridLSTM", "AllGridLSTM",
    "AllGridConvLSTM", "AllGridDLinear", "AllGridPatchTST",
]
