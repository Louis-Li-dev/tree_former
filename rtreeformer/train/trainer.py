"""
rtreeformer.train.trainer
=========================
Training loop and evaluation utilities for R-Treeformer and baseline models.

The ``run_training`` function is a self-contained train/validate/test cycle
with early stopping, gradient clipping, and automatic best-weight restoration.

Normalisation and Output Rules
-------------------------------
1. R-Treeformer (proposed model, ``use_revin=True``):
   Applies per-instance RevIN internally. Input dataloaders yield raw data.
2. Baselines (``use_revin=False``):
   Trained with dataset-level Z-score standardisation (per-grid mean and std
   computed from training split). Inputs are normalised before forward pass, and
   outputs are inverse-transformed back to raw population scale during evaluation.
3. Integer Output Alignment:
   All final predictions (and test outputs) are rounded to non-negative integers
   to represent actual headcount values.

Usage::

    from rtreeformer.train.trainer import run_training
    from rtreeformer.utils.config  import TRAIN_CFG

    model, preds, trues, preds_eval, trues_eval, train_time_s, n_params = run_training(
        name         = "All-Grid LSTM (Large)",
        model        = my_model,
        cfg          = TRAIN_CFG,
        train_loader = train_l,
        val_loader   = val_l,
        test_loader  = test_l,
        device       = device,
    )
"""

from __future__ import annotations
import copy
import gc
import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
def run_training(
    name: str,
    model: nn.Module,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eval_grid_mask: Optional[np.ndarray] = None,
    integer_outputs: bool = True,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """
    Train *model* and evaluate on the test set.

    Parameters
    ----------
    name           : human-readable model name (used only for logging)
    model          : nn.Module -- will be moved to *device* in-place
    cfg            : dict with keys ``lr``, ``max_epochs``, ``patience``, ``clip``
    train_loader   : DataLoader yielding (x, y, hour, dow) batches
    val_loader     : validation DataLoader
    test_loader    : test DataLoader
    device         : torch.device
    scaler         : optional tuple (mean, std) for dataset Z-score normalisation;
                     if None and model does NOT use internal RevIN, scaler is
                     automatically computed from train_loader.dataset.data
    eval_grid_mask : optional (M,) index array -- restrict test metrics to these grids
    integer_outputs: if True (default), round final predictions to non-negative ints
    verbose        : print per-epoch progress to stdout
    logger         : optional Python logger; epoch lines are sent to both stdout
                     (when verbose=True) and the log file

    Returns
    -------
    (model, preds, trues, preds_eval, trues_eval, train_time_s, n_params)
        preds       : (T_test, N) full test predictions
        trues       : (T_test, N) full ground truth
        preds_eval  : predictions restricted to eval_grid_mask (or full if None)
        trues_eval  : ground truth restricted to eval_grid_mask (or full if None)
        train_time_s: wall-clock seconds for the entire training run
        n_params    : trainable parameter count
    """
    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)

    model = model.to(device)
    use_internal_revin = getattr(model, "use_revin", False)

    # Prepare Z-score normalisation tensors for baselines (no internal RevIN)
    mean_t, std_t = None, None
    if not use_internal_revin:
        if scaler is not None:
            mean_np, std_np = scaler
        else:
            raw_train = getattr(train_loader.dataset, "data", None)
            if raw_train is not None:
                mean_np = raw_train.mean(axis=1, keepdims=True)
                std_np  = raw_train.std(axis=1, keepdims=True) + 1e-5
            else:
                mean_np, std_np = None, None

        if mean_np is not None and std_np is not None:
            mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device).view(1, -1, 1)
            std_t  = torch.tensor(std_np,  dtype=torch.float32, device=device).view(1, -1, 1)

    opt     = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit    = nn.MSELoss()
    best_val, best_weights, no_improve = float("inf"), None, 0
    max_ep  = cfg.get("max_epochs", 10)
    t_start = time.time()

    for epoch in range(max_ep):
        # -- Training ---------------------------------------------------------
        model.train()
        t_loss = 0.0
        for xb, yb, hb, db in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            hb, db = hb.to(device), db.to(device)

            if mean_t is not None:
                xb_in = (xb - mean_t) / std_t
                yb_in = (yb - mean_t) / std_t
            else:
                xb_in, yb_in = xb, yb

            opt.zero_grad()
            pred  = model(xb_in, hb, db)
            loss  = crit(pred, yb_in)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            t_loss += loss.item() * xb.size(0)
        t_loss /= len(train_loader.dataset)

        # -- Validation -------------------------------------------------------
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb, hb, db in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                hb, db = hb.to(device), db.to(device)

                if mean_t is not None:
                    xb_in = (xb - mean_t) / std_t
                    yb_in = (yb - mean_t) / std_t
                else:
                    xb_in, yb_in = xb, yb

                v_loss += crit(model(xb_in, hb, db), yb_in).item() * xb.size(0)
        v_loss /= len(val_loader.dataset)

        # -- Early stopping ---------------------------------------------------
        tag = ""
        if v_loss < best_val:
            best_val, best_weights, no_improve = (
                v_loss, copy.deepcopy(model.state_dict()), 0
            )
            tag = " <- best"
        else:
            no_improve += 1

        # GPU memory info (only when CUDA is active)
        gpu_info = ""
        if device.type == "cuda":
            used_mb  = torch.cuda.memory_allocated(device) / 1024 ** 2
            total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
            gpu_info = f" | GPU {used_mb:.0f}/{total_mb:.0f}MB"

        elapsed = time.time() - t_start
        line = (
            f"  [{name}] Epoch {epoch+1:3d}/{max_ep:3d} | "
            f"Train: {t_loss:.5f} | Val: {v_loss:.5f}{tag} | "
            f"elapsed {elapsed:.1f}s{gpu_info}"
        )
        _log(line)

        if no_improve >= cfg["patience"]:
            _log(f"  [Early Stop] {name} epoch {epoch + 1}")
            break

    train_time = time.time() - t_start
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # -- Test inference -------------------------------------------------------
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb, hb, db in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            hb, db = hb.to(device), db.to(device)

            if mean_t is not None:
                xb_in    = (xb - mean_t) / std_t
                pred_out = model(xb_in, hb, db)
                pred_raw = pred_out * std_t + mean_t
            else:
                pred_raw = model(xb, hb, db)

            if integer_outputs:
                pred_raw = torch.clamp(torch.round(pred_raw), min=0)

            preds_list.append(pred_raw.cpu().numpy())
            trues_list.append(yb.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)[:, :, 0]   # (T_test, N)
    trues = np.concatenate(trues_list, axis=0)[:, :, 0]   # (T_test, N)

    if integer_outputs:
        preds = np.clip(np.round(preds), 0, None)

    if eval_grid_mask is not None:
        preds_eval = preds[:, eval_grid_mask]
        trues_eval = trues[:, eval_grid_mask]
    else:
        preds_eval, trues_eval = preds, trues

    return model, preds, trues, preds_eval, trues_eval, train_time, n_params
