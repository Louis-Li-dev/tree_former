"""
hmst.train.trainer
==================
Training loop and evaluation utilities for HMST-v2 and baseline models.

The ``run_training`` function is a self-contained train/validate/test cycle
with early stopping, gradient clipping, and automatic best-weight restoration.

Usage::

    from hmst.train.trainer import run_training
    from hmst.utils.config  import TRAIN_CFG

    model, preds, trues, metrics = run_training(
        name       = "HMST-v2 (Large)",
        model      = my_model,
        cfg        = TRAIN_CFG,
        train_loader = train_l,
        val_loader   = val_l,
        test_loader  = test_l,
        device       = device,
    )
"""

from __future__ import annotations
import copy
import time
from typing import Optional

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
    eval_grid_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> tuple:
    """
    Train *model* and evaluate on the test set.

    Parameters
    ----------
    name           : human-readable model name (used only for logging)
    model          : nn.Module — will be moved to *device* in-place
    cfg            : dict with keys ``lr``, ``max_epochs``, ``patience``, ``clip``
    train_loader   : DataLoader yielding (x, y, hour, dow) batches
    val_loader     : validation DataLoader
    test_loader    : test DataLoader
    device         : torch.device
    eval_grid_mask : optional (M,) index array — restrict test metrics to these grids
    verbose        : print per-epoch progress

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
    model   = model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit    = nn.MSELoss()
    best_val, best_weights, no_improve = float("inf"), None, 0
    t_start = time.time()

    for epoch in range(cfg["max_epochs"]):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for xb, yb, hb, db in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            hb, db = hb.to(device), db.to(device)
            opt.zero_grad()
            pred  = model(xb, hb, db)
            loss  = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            t_loss += loss.item() * xb.size(0)
        t_loss /= len(train_loader.dataset)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb, hb, db in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                hb, db = hb.to(device), db.to(device)
                v_loss += crit(model(xb, hb, db), yb).item() * xb.size(0)
        v_loss /= len(val_loader.dataset)

        # ── Early stopping ──────────────────────────────────────────────────
        tag = ""
        if v_loss < best_val:
            best_val, best_weights, no_improve = (
                v_loss, copy.deepcopy(model.state_dict()), 0
            )
            tag = " <- best"
        else:
            no_improve += 1

        if verbose and ((epoch + 1) % 5 == 0 or tag):
            print(
                f"  [{name}] Epoch {epoch+1:3d} | "
                f"Train: {t_loss:.5f}  Val: {v_loss:.5f}{tag}"
            )
        if no_improve >= cfg["patience"]:
            if verbose:
                print(f"  [Early Stop] epoch {epoch + 1}")
            break

    train_time = time.time() - t_start
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Restore best weights ─────────────────────────────────────────────────
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ── Test inference ───────────────────────────────────────────────────────
    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb, hb, db in test_loader:
            preds_list.append(
                model(xb.to(device), hb.to(device), db.to(device)).cpu().numpy()
            )
            trues_list.append(yb.numpy())

    preds = np.concatenate(preds_list, axis=0)[:, :, 0]   # (T_test, N)
    trues = np.concatenate(trues_list, axis=0)[:, :, 0]

    if eval_grid_mask is not None:
        preds_eval = preds[:, eval_grid_mask]
        trues_eval = trues[:, eval_grid_mask]
    else:
        preds_eval, trues_eval = preds, trues

    return model, preds, trues, preds_eval, trues_eval, train_time, n_params
