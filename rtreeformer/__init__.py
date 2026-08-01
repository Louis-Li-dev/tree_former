"""
rtreeformer
===========
R-Treeformer (Hierarchical Masked Spatial Transformer)

Top-level package exposing all public APIs.

Quick start::

    from rtreeformer.model   import RTreeformer, AllGridLSTM, AllGridReformer
    from rtreeformer.train   import make_loaders, run_training
    from rtreeformer.utils   import calculate_metrics, build_forest_masks
    from rtreeformer.utils   import MODEL_SIZES, TRAIN_CFG
"""

from .model   import RTreeformer, TimeEmbedding, HierarchicalBlock
from .model   import RevIN, EachGridLSTM, AllGridLSTM
from .model   import AllGridDLinear, AllGridPatchTST, AllGridReformer, AllGridInformer
from .train   import CHTDataset, make_loaders, run_training
from .utils   import MODEL_SIZES, TRAIN_CFG, SMALL, BASE, LARGE
from .utils   import calculate_metrics, build_forest_masks

__version__ = "0.2.0"
__author__  = "R-Treeformer Research Team"

__all__ = [
    # models - proposed
    "RTreeformer", "TimeEmbedding", "HierarchicalBlock",
    # models - baselines
    "RevIN", "EachGridLSTM", "AllGridLSTM",
    "AllGridDLinear", "AllGridPatchTST", "AllGridReformer", "AllGridInformer",
    # training
    "CHTDataset", "make_loaders", "run_training",
    # utilities
    "MODEL_SIZES", "TRAIN_CFG", "SMALL", "BASE", "LARGE",
    "calculate_metrics", "build_forest_masks",
]
