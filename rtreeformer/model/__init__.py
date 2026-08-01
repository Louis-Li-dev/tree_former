"""
rtreeformer.model
=================
All model architectures used in the paper:

Proposed
--------
    RTreeformer     - Hierarchical Masked Spatial Transformer
    TimeEmbedding   - Hour-of-day + Day-of-week embedding
    HierarchicalBlock - One Transformer block with R-Tree masked attention

Baselines
---------
    RevIN           - Reversible Instance Normalization (shared utility)
    AllGridDLinear  - DLinear (AAAI 2023, TSL series_decomp); internal RevIN
    AllGridPatchTST - PatchTST (ICLR 2023, TSL PatchEmbedding); internal RevIN
    AllGridReformer - Reformer (ICLR 2020, LSH attention); internal RevIN
"""

from .rtreeformer import TimeEmbedding, HierarchicalBlock, RTreeformer, RevIN
from .baselines import (
    EachGridLSTM,
    AllGridLSTM,
    AllGridDLinear,
    AllGridPatchTST,
    AllGridReformer,
    AllGridInformer,  # backwards-compat alias for AllGridReformer
)

__all__ = [
    # proposed
    "TimeEmbedding", "HierarchicalBlock", "RTreeformer",
    # baselines
    "RevIN", "EachGridLSTM", "AllGridLSTM",
    "AllGridDLinear", "AllGridPatchTST",
    "AllGridReformer", "AllGridInformer",
]
