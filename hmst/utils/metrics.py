"""
hmst.utils.metrics
==================
Evaluation metrics for population-flow forecasting.

All functions operate on numpy arrays shaped **(T, N)** — time × grids.

Functions
---------
mae(y_true, y_pred)                → float
rmse(y_true, y_pred)               → float
wmape(y_true, y_pred)              → float  (percentage)
fast_dtw_distance(s1, s2, w)       → float
calculate_metrics(y_true, y_pred)  → (mae, rmse, wmape, daily_dtw, step_dtw)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error (%)."""
    return float(
        np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100
    )


# ---------------------------------------------------------------------------
def fast_dtw_distance(s1: np.ndarray, s2: np.ndarray, w: int = 24) -> float:
    """
    Windowed Dynamic Time Warping distance between two 1-D sequences.

    Parameters
    ----------
    s1, s2 : 1-D arrays of equal length T
    w      : Sakoe-Chiba band width (default 24 → ±1 day for hourly data)
    """
    T = len(s1)
    dtw_mat = np.full((T + 1, T + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, T + 1):
        for j in range(max(1, i - w), min(T + 1, i + w + 1)):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_mat[i, j] = cost + min(
                dtw_mat[i - 1, j], dtw_mat[i, j - 1], dtw_mat[i - 1, j - 1]
            )
    return float(dtw_mat[T, T])


# ---------------------------------------------------------------------------
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dtw_grids: int = 50,
) -> tuple[float, float, float, float, float]:
    """
    Compute all five benchmark metrics.

    Parameters
    ----------
    y_true, y_pred : (T, N)  raw-scale predictions and ground truth
    dtw_grids      : number of grids randomly sampled for DTW (expensive)

    Returns
    -------
    (mae, rmse, wmape_pct, daily_dtw, step_dtw)
        daily_dtw  — average DTW over 24-hour day segments
        step_dtw   — full-sequence DTW normalised by T (per-step cost)
    """
    _mae   = mae(y_true, y_pred)
    _rmse  = rmse(y_true, y_pred)
    _wmape = wmape(y_true, y_pred)

    T, N = y_true.shape
    sel  = np.random.choice(N, min(dtw_grids, N), replace=False)

    # Daily DTW (24-hour segments)
    n_days     = T // 24
    daily_dtws = [
        fast_dtw_distance(
            y_true[d * 24 : (d + 1) * 24, n],
            y_pred[d * 24 : (d + 1) * 24, n],
            w=6,
        )
        for d in range(n_days)
        for n in sel
    ]
    daily_dtw = float(np.mean(daily_dtws)) if daily_dtws else float("nan")

    # Step DTW (full sequence, normalised per step)
    step_dtw = float(
        np.mean(
            [fast_dtw_distance(y_true[:, n], y_pred[:, n], w=24) / T for n in sel]
        )
    )

    return _mae, _rmse, _wmape, daily_dtw, step_dtw
