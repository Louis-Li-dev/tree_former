"""
rtreeformer.train
=================
Training utilities: Dataset, DataLoader factory, and the main training loop.
"""

from .dataset import CHTDataset, make_loaders
from .trainer import run_training

__all__ = ["CHTDataset", "make_loaders", "run_training"]
