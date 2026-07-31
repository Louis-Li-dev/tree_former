"""
scripts/_common.py
==================
Shared utilities for all run_*.py scripts.

Provides:
  - get_run_dir()        : resolve the latest run directory from runs/latest.txt
  - setup_logger()       : configure a file + console logger
  - gpu_cleanup()        : delete a model and flush GPU memory
  - save_result()        : write predictions + metadata to runs/{id}/results/
  - load_result()        : load a previously saved result
  - load_run_data()      : load all preprocessed arrays from runs/{id}/data/
  - eval_on_loader()     : collect predictions from a model on any DataLoader
  - KNOWN_PARAMS         : reference param-count table (from notebook output)
"""

from __future__ import annotations
import gc
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Canonical parameter counts (from notebook; used by run_results.py)
# ---------------------------------------------------------------------------
KNOWN_PARAMS: dict[str, float] = {
    "All-Grid ConvLSTM (Small)":  10.4,
    "All-Grid ConvLSTM (Base)":   39.3,
    "All-Grid ConvLSTM (Large)": 152.4,
    "Each-Grid LSTM (Small)":     12.3,
    "Each-Grid LSTM (Base)":      45.1,
    "Each-Grid LSTM (Large)":    172.2,
    "All-Grid LSTM (Small)":       1.2,
    "All-Grid LSTM (Base)":        4.5,
    "All-Grid LSTM (Large)":      17.2,
    "All-Grid DLinear (Small)":    0.1,
    "All-Grid DLinear (Base)":     0.1,
    "All-Grid DLinear (Large)":    0.1,
    "All-Grid PatchTST (Base)":   25.8,
    "All-Grid PatchTST (Large)": 200.8,
    "All-Grid Reformer (Base)":   23.4,
    "All-Grid Reformer (Large)": 183.2,
    "HMST-v2 (Base)":             31.1,
    "HMST-v2 (Large)":           235.3,
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent   # project root
RUNS_DIR  = REPO_ROOT / "runs"


def get_run_dir() -> Path:
    """Return the current run directory (from runs/latest.txt)."""
    pointer = RUNS_DIR / "latest.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            "runs/latest.txt not found. Run scripts/run_preprocess.py first."
        )
    run_id  = pointer.read_text(encoding="utf-8").strip()
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir


def get_or_create_run_dir(run_id: str) -> Path:
    """Create and register a new run directory."""
    run_dir = RUNS_DIR / run_id
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    # Write pointer
    pointer = RUNS_DIR / "latest.txt"
    pointer.write_text(run_id, encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both *log_file* and the console.
    Timestamps are included in every line.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# GPU cleanup
# ---------------------------------------------------------------------------
def gpu_cleanup(*models: nn.Module) -> None:
    """
    Delete model objects and flush the CUDA memory cache.
    Call between model training runs to prevent GPU memory accumulation.
    """
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------
def save_result(
    run_dir: Path,
    key: str,
    preds: np.ndarray,
    trues: np.ndarray,
    meta: dict,
) -> None:
    """
    Persist predictions and metadata for one model/variant.

    Files written
    -------------
    runs/{id}/results/{key}.npz       -- preds (T,N) and trues (T,N) arrays
    runs/{id}/results/{key}_meta.json -- metrics dict, n_params, train_time, etc.

    Parameters
    ----------
    key  : filename stem, e.g. "baseline_convlstm_small"
    meta : arbitrary dict (metrics, n_params, train_time_s, model_name, size ...)
    """
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez_compressed(results_dir / f"{key}.npz", preds=preds, trues=trues)
    with open(results_dir / f"{key}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_result(run_dir: Path, key: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a previously saved result. Returns (preds, trues, meta)."""
    npz  = np.load(run_dir / "results" / f"{key}.npz")
    with open(run_dir / "results" / f"{key}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return npz["preds"], npz["trues"], meta


def result_exists(run_dir: Path, key: str) -> bool:
    """Return True if the result files for *key* already exist."""
    return (run_dir / "results" / f"{key}.npz").exists()


# ---------------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------------
def load_run_data(run_dir: Path) -> dict:
    """
    Load all arrays saved by run_preprocess.py.

    Returns a dict with keys:
      selected_data, selected_coords, selected_grids, selected_indices,
      root_labels, hmst_masks, forest_levels, splits,
      adapted_data, adapt_bc_indices, adapt_region_info, meta
    """
    data_dir = run_dir / "data"

    def _npy(name):
        return np.load(data_dir / f"{name}.npy", allow_pickle=False)

    def _npy_obj(name):
        return np.load(data_dir / f"{name}.npy", allow_pickle=True)

    with open(data_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    with open(data_dir / "adapt_region_info.json", encoding="utf-8") as f:
        adapt_region_info = json.load(f)

    with open(data_dir / "forest_levels.pkl", "rb") as f:
        forest_levels = pickle.load(f)

    with open(data_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "selected_data":      _npy("selected_data"),
        "selected_coords":    _npy("selected_coords"),
        "selected_grids":     _npy_obj("selected_grids"),
        "selected_indices":   _npy("selected_indices"),
        "root_labels":        _npy("root_labels"),
        "hmst_masks":         [_npy(f"hmst_mask_{i}") for i in range(4)],
        "forest_levels":      forest_levels,
        "splits":             splits,
        "adapted_data":       _npy("adapted_data"),
        "adapt_bc_indices":   _npy("adapt_bc_indices"),
        "adapt_region_info":  adapt_region_info,
        "meta":               meta,
    }


# ---------------------------------------------------------------------------
# Zero-shot prediction helper
# ---------------------------------------------------------------------------
def eval_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    grid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model in eval mode over *loader* and return (preds, trues) at raw scale.
    Handles both RevIN and Z-score-normalised baselines.

    Parameters
    ----------
    scaler    : (mean_np, std_np) arrays — required for non-RevIN baselines
    grid_mask : optional index array to restrict output columns

    Returns
    -------
    preds, trues : (T, N) or (T, len(grid_mask)) numpy arrays
    """
    use_revin = getattr(model, "use_revin", False)
    mean_t, std_t = None, None

    if not use_revin and scaler is None:
        raw_data = getattr(loader.dataset, "data", None)
        if raw_data is not None:
            mean_np = raw_data.mean(axis=1, keepdims=True)
            std_np  = raw_data.std(axis=1, keepdims=True) + 1e-5
            scaler  = (mean_np, std_np)

    if not use_revin and scaler is not None:
        mean_np, std_np = scaler
        mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device).view(1, -1, 1)
        std_t  = torch.tensor(std_np,  dtype=torch.float32, device=device).view(1, -1, 1)

    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for xb, yb, hb, db in loader:
            xb, yb = xb.to(device), yb.to(device)
            hb, db = hb.to(device), db.to(device)
            if mean_t is not None:
                pred_raw = model((xb - mean_t) / std_t, hb, db) * std_t + mean_t
            else:
                pred_raw = model(xb, hb, db)
            pred_raw = torch.clamp(torch.round(pred_raw), min=0)
            ps.append(pred_raw.cpu().numpy())
            ts.append(yb.cpu().numpy())

    preds = np.clip(np.round(np.concatenate(ps, 0)[:, :, 0]), 0, None)
    trues = np.concatenate(ts, 0)[:, :, 0]

    if grid_mask is not None:
        return preds[:, grid_mask], trues[:, grid_mask]
    return preds, trues
