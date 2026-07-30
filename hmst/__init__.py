"""
hmst
====
Hierarchical Masked Spatial Transformer v2 (HMST-v2)

Top-level package exposing all public APIs.

Quick start::

    from hmst.model   import HMSTv2, AllGridLSTM, AllGridInformer
    from hmst.train   import make_loaders, run_training
    from hmst.utils   import calculate_metrics, build_forest_masks
    from hmst.utils   import MODEL_SIZES, TRAIN_CFG
"""

from .model   import HMSTv2, TimeEmbedding, HierarchicalBlock
from .model   import RevIN, EachGridLSTM, AllGridLSTM, AllGridConvLSTM
from .model   import AllGridDLinear, AllGridPatchTST, AllGridReformer, AllGridInformer
from .train   import CHTDataset, make_loaders, run_training
from .utils   import MODEL_SIZES, TRAIN_CFG, SMALL, BASE, LARGE
from .utils   import calculate_metrics, build_forest_masks

__version__ = "0.2.0"
__author__  = "HMST Research Team"

__all__ = [
    # models — proposed
    "HMSTv2", "TimeEmbedding", "HierarchicalBlock",
    # models — baselines
    "RevIN", "EachGridLSTM", "AllGridLSTM", "AllGridConvLSTM",
    "AllGridDLinear", "AllGridPatchTST", "AllGridReformer", "AllGridInformer",
    # training
    "CHTDataset", "make_loaders", "run_training",
    # utilities
    "MODEL_SIZES", "TRAIN_CFG", "SMALL", "BASE", "LARGE",
    "calculate_metrics", "build_forest_masks",
]
