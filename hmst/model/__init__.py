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
    EachGridLSTM    — Per-grid independent LSTM (vectorised); internal RevIN
    AllGridLSTM     — Shared LSTM across all grids; internal RevIN
    AllGridConvLSTM — Spatial ConvLSTM (auto H×W from coords); NO internal RevIN
    AllGridDLinear  — DLinear (AAAI 2023, TSL series_decomp); internal RevIN
    AllGridPatchTST — PatchTST (ICLR 2023, TSL PatchEmbedding); internal RevIN
    AllGridReformer — Reformer (ICLR 2020, LSH attention); internal RevIN
"""

from .hmstv2 import TimeEmbedding, HierarchicalBlock, HMSTv2, RevIN
from .baselines import (
    EachGridLSTM,
    AllGridLSTM,
    AllGridConvLSTM,
    AllGridDLinear,
    AllGridPatchTST,
    AllGridReformer,
    AllGridInformer,  # backwards-compat alias for AllGridReformer
)

__all__ = [
    # proposed
    "TimeEmbedding", "HierarchicalBlock", "HMSTv2",
    # baselines
    "RevIN", "EachGridLSTM", "AllGridLSTM",
    "AllGridConvLSTM", "AllGridDLinear", "AllGridPatchTST",
    "AllGridReformer", "AllGridInformer",
]
