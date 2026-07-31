"""
scripts/run_hmstv2.py
=====================
Train proposed HMST-v2 model (Base / Large).

This script also:
  - Evaluates HMST-v2 Large zero-shot on the swapped adaptation test set
  - Runs HMST-v2 partial fine-tuning on the swapped adaptation data
    (unfreezes blocks[0] and out_proj; trains with masked MSE on B+C grids)

Usage
-----
    python scripts/run_hmstv2.py
    python scripts/run_hmstv2.py --sizes base
    python scripts/run_hmstv2.py --no-finetune  # skip adaptation
    python scripts/run_hmstv2.py --force
"""

from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import (
        get_run_dir, setup_logger, gpu_cleanup,
        save_result, result_exists, load_run_data, eval_on_loader,
    )
except ModuleNotFoundError:
    from _common import (
        get_run_dir, setup_logger, gpu_cleanup,
        save_result, result_exists, load_run_data, eval_on_loader,
    )

from hmst.model   import HMSTv2
from hmst.train   import CHTDataset, run_training
from hmst.utils   import MODEL_SIZES, TRAIN_CFG, LOOKBACK, BATCH_SIZE, calculate_metrics

SCRIPT = "hmstv2"
SIZES  = ("base", "large")

# Fine-tune config (shorter, smaller lr)
FT_CFG = {"lr": 5e-4, "max_epochs": 20, "patience": 5, "clip": 1.0}


def make_loaders(data, splits):
    st, sv = splits["split_train"], splits["split_val"]
    kw = dict(batch_size=BATCH_SIZE, drop_last=False)
    tl = DataLoader(CHTDataset(data[:, :st],   LOOKBACK, 0),  shuffle=True,  **kw)
    vl = DataLoader(CHTDataset(data[:, st:sv], LOOKBACK, st), shuffle=False, **kw)
    el = DataLoader(CHTDataset(data[:, sv:],   LOOKBACK, sv), shuffle=False, **kw)
    return tl, vl, el


def finetune_hmst(model, ad_train_l, ad_val_l, bc_tensor, device, logger):
    """Partial fine-tune: freeze all layers except blocks[0] and out_proj."""
    ft_model = copy.deepcopy(model)
    for p in ft_model.parameters():
        p.requires_grad = False
    for p in ft_model.blocks[0].parameters():
        p.requires_grad = True
    for p in ft_model.out_proj.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(
        [p for p in ft_model.parameters() if p.requires_grad],
        lr=FT_CFG["lr"],
    )
    best_val, best_w, no_imp = float("inf"), None, 0
    t_start = time.time()

    for epoch in range(FT_CFG["max_epochs"]):
        ft_model.train()
        for xb, yb, hb, db in ad_train_l:
            xb, yb, hb, db = xb.to(device), yb.to(device), hb.to(device), db.to(device)
            opt.zero_grad()
            pred = ft_model(xb, hb, db)
            loss = nn.functional.mse_loss(pred[:, bc_tensor, :], yb[:, bc_tensor, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in ft_model.parameters() if p.requires_grad],
                FT_CFG["clip"],
            )
            opt.step()

        ft_model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb, hb, db in ad_val_l:
                xb, yb, hb, db = xb.to(device), yb.to(device), hb.to(device), db.to(device)
                v = nn.functional.mse_loss(
                    ft_model(xb, hb, db)[:, bc_tensor, :], yb[:, bc_tensor, :]
                ).item() * xb.size(0)
                v_loss += v
        v_loss /= len(ad_val_l.dataset)

        tag = ""
        if v_loss < best_val:
            best_val, best_w, no_imp = v_loss, copy.deepcopy(ft_model.state_dict()), 0
            tag = " <- best"
        else:
            no_imp += 1
        if (epoch + 1) % 5 == 0 or tag:
            logger.info(f"  FT Epoch {epoch+1:3d} | Val: {v_loss:.5f}{tag}")
        if no_imp >= FT_CFG["patience"]:
            logger.info(f"  [Early Stop] FT epoch {epoch+1}")
            break

    ft_model.load_state_dict(best_w)
    return ft_model, time.time() - t_start


def run(sizes=SIZES, force=False, do_finetune=True):
    run_dir = get_run_dir()
    logger  = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    logger.info(f"=== {SCRIPT}  run_dir={run_dir} ===")

    d      = load_run_data(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    hmst_masks = [torch.tensor(m, dtype=torch.float32) for m in d["hmst_masks"]]
    N          = d["selected_data"].shape[0]
    train_l, val_l, test_l = make_loaders(d["selected_data"], d["splits"])

    sv = d["splits"]["split_val"]
    st = d["splits"]["split_train"]
    ad_train_l = DataLoader(
        CHTDataset(d["adapted_data"][:, :st],   LOOKBACK, 0),  batch_size=BATCH_SIZE, shuffle=True)
    ad_val_l   = DataLoader(
        CHTDataset(d["adapted_data"][:, st:sv], LOOKBACK, st), batch_size=BATCH_SIZE, shuffle=False)
    ad_test_l  = DataLoader(
        CHTDataset(d["adapted_data"][:, sv:],   LOOKBACK, sv), batch_size=BATCH_SIZE, shuffle=False)

    bc_mask   = d["adapt_bc_indices"]
    bc_tensor = torch.tensor(bc_mask, dtype=torch.long)

    large_model = None  # keep Large model for fine-tuning

    for sz in sizes:
        key    = f"{SCRIPT}_{sz}"
        zs_key = "adapt_zs_hmstv2_large"
        ft_key = "adapt_ft_hmstv2"
        if result_exists(run_dir, key) and not force:
            logger.info(f"  [{sz}] Exists, skipping.")
            continue

        s = MODEL_SIZES[sz]
        logger.info(f"\n{'='*60}\n  HMST-v2 ({sz.capitalize()})\n{'='*60}")
        model = HMSTv2(
            num_grids=N, lookback=LOOKBACK,
            d_model=s["d_model"], d_k=s["d_k"],
            num_layers=s["num_layers"], dropout=0.1,
            forest_masks=hmst_masks,
            use_revin=True, use_rtree_mask=True, random_mask=False,
            use_spatial_emb=True, use_time_emb=True,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Params: {n_params/1000:.1f}K")

        _, preds, trues, pe, te, t_secs, _ = run_training(
            f"HMST-v2 ({sz.capitalize()})",
            model, TRAIN_CFG, train_l, val_l, test_l, device, logger=logger,
        )
        mae, rmse, wmape, ddtw, sdtw = calculate_metrics(te, pe)
        save_result(run_dir, key, preds, trues, {
            "model_name": f"HMST-v2 ({sz.capitalize()})",
            "size": sz, "n_params_K": round(n_params/1000, 1),
            "train_time_s": round(t_secs, 1),
            "MAE": mae, "RMSE": rmse, "wMAPE": wmape,
            "Daily_DTW": ddtw, "Step_DTW": sdtw,
        })
        logger.info(f"  TEST MAE={mae:.4f}  wMAPE={wmape:.2f}%  Time={t_secs:.1f}s")

        if sz == "large":
            # Zero-shot on adapted test
            logger.info("  Zero-shot eval on adapted test set ...")
            zs_p, zs_t = eval_on_loader(model, ad_test_l, device, grid_mask=bc_mask)
            zm, *_ = calculate_metrics(zs_t, zs_p)
            save_result(run_dir, zs_key, zs_p, zs_t,
                        {"model_name": "HMST-v2 (Large) Zero-Shot", "MAE": zm})
            logger.info(f"  Zero-Shot MAE on B+C = {zm:.4f}")

            # Partial fine-tune
            if do_finetune and (not result_exists(run_dir, ft_key) or force):
                logger.info("  Starting HMST partial fine-tune on adapted data ...")
                ft_model, ft_time = finetune_hmst(
                    model, ad_train_l, ad_val_l, bc_tensor, device, logger
                )
                ft_p, ft_t = eval_on_loader(ft_model, ad_test_l, device, grid_mask=bc_mask)
                fm, *_ = calculate_metrics(ft_t, ft_p)
                save_result(run_dir, ft_key, ft_p, ft_t, {
                    "model_name": "HMST-v2 Fine-Tune",
                    "train_time_s": round(ft_time, 1), "MAE": fm,
                })
                logger.info(f"  Fine-Tune MAE on B+C = {fm:.4f}  Time={ft_time:.1f}s")
                gpu_cleanup(ft_model)

        gpu_cleanup(model)

    logger.info(f"=== {SCRIPT} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-finetune", dest="finetune", action="store_false")
    args = parser.parse_args()
    run(sizes=args.sizes, force=args.force, do_finetune=args.finetune)
