"""
rtreeformer.train.dataset
=========================
PyTorch Dataset and DataLoader factory for CHT population-flow data.

The dataset returns **(x, y, hour, dow)** tuples where:
    x    - (N, L)  input look-back window
    y    - (N, 1)  next-step ground truth
    hour - scalar long  prediction hour-of-day (0-23)
    dow  - scalar long  prediction day-of-week (0-6)

Usage::

    from rtreeformer.train.dataset import CHTDataset, make_loaders

    train_l, val_l, test_l = make_loaders(
        selected_data,
        lookback=12, batch_size=32,
        train_frac=0.70, val_frac=0.85,
    )
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
class CHTDataset(Dataset):
    """
    Sliding-window dataset for multi-grid hourly population flow.

    Parameters
    ----------
    raw_data    : (N, T) float array — all grids × full time horizon
    lookback    : look-back window length L
    time_offset : absolute time index of raw_data[:, 0]; used to compute
                  the hour-of-day and day-of-week for each prediction step
    """

    def __init__(
        self,
        raw_data: np.ndarray,
        lookback: int = 12,
        time_offset: int = 0,
    ):
        self.data        = raw_data
        self.lookback    = lookback
        self.time_offset = time_offset

    def __len__(self) -> int:
        return self.data.shape[1] - self.lookback

    def __getitem__(self, idx: int):
        x = self.data[:, idx : idx + self.lookback]
        y = self.data[:, idx + self.lookback : idx + self.lookback + 1]

        pred_abs = self.time_offset + idx + self.lookback
        hour = pred_abs % 24
        dow  = (pred_abs // 24) % 7

        return (
            torch.tensor(x,    dtype=torch.float32),
            torch.tensor(y,    dtype=torch.float32),
            torch.tensor(hour, dtype=torch.long),
            torch.tensor(dow,  dtype=torch.long),
        )


# ---------------------------------------------------------------------------
def make_loaders(
    data: np.ndarray,
    lookback: int = 12,
    batch_size: int = 32,
    train_frac: float = 0.70,
    val_frac: float   = 0.85,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data and return (train_loader, val_loader, test_loader).

    Parameters
    ----------
    data        : (N, T) float array — full dataset (all grids)
    lookback    : look-back window length
    batch_size  : mini-batch size
    train_frac  : fraction of T used for training
    val_frac    : fraction of T used for train+validation (rest = test)

    Returns
    -------
    train_loader, val_loader, test_loader — PyTorch DataLoaders
    """
    T           = data.shape[1]
    split_train = int(T * train_frac)
    split_val   = int(T * val_frac)

    train_ds = CHTDataset(data[:, :split_train],        lookback, 0)
    val_ds   = CHTDataset(data[:, split_train:split_val], lookback, split_train)
    test_ds  = CHTDataset(data[:, split_val:],          lookback, split_val)

    train_l = DataLoader(train_ds, batch_size, shuffle=True,  drop_last=False)
    val_l   = DataLoader(val_ds,   batch_size, shuffle=False, drop_last=False)
    test_l  = DataLoader(test_ds,  batch_size, shuffle=False, drop_last=False)

    return train_l, val_l, test_l
