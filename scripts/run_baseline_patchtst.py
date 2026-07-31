"""
scripts/run_baseline_patchtst.py
=====================================
Train All-Grid PatchTST (("base", "large")).
Also runs zero-shot and retrain adaptation experiments for the Large variant.

Usage
-----
    python scripts/run_baseline_patchtst.py
    python scripts/run_baseline_patchtst.py --sizes base
    python scripts/run_baseline_patchtst.py --no-adapt
    python scripts/run_baseline_patchtst.py --force
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
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
from hmst.model   import AllGridPatchTST
from hmst.train   import CHTDataset, run_training
from hmst.utils   import MODEL_SIZES, TRAIN_CFG, LOOKBACK, BATCH_SIZE, calculate_metrics

SCRIPT = "baseline_patchtst"
SIZES  = ("base", "large")


def make_loaders(data, splits):
    st, sv = splits["split_train"], splits["split_val"]
    kw = dict(batch_size=BATCH_SIZE, drop_last=False)
    tl = DataLoader(CHTDataset(data[:, :st],   LOOKBACK, 0),  shuffle=True,  **kw)
    vl = DataLoader(CHTDataset(data[:, st:sv], LOOKBACK, st), shuffle=False, **kw)
    el = DataLoader(CHTDataset(data[:, sv:],   LOOKBACK, sv), shuffle=False, **kw)
    return tl, vl, el


def build_model(sz):
    s = MODEL_SIZES[sz]
    return AllGridPatchTST(lookback=LOOKBACK, patch_len=4, stride=4,
                           d_model=s["d_model"], nhead=s["nhead"], num_layers=s["num_layers"])


def run(sizes=SIZES, force=False, do_adapt=True):
    run_dir = get_run_dir()
    logger  = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    logger.info(f"=== {SCRIPT}  run_dir={run_dir} ===")

    d      = load_run_data(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_l, val_l, test_l = make_loaders(d["selected_data"], d["splits"])
    sv, st = d["splits"]["split_val"], d["splits"]["split_train"]
    bc_mask = d["adapt_bc_indices"]

    ad_train_l = DataLoader(CHTDataset(d["adapted_data"][:, :st],   LOOKBACK, 0),
                            batch_size=BATCH_SIZE, shuffle=True)
    ad_val_l   = DataLoader(CHTDataset(d["adapted_data"][:, st:sv], LOOKBACK, st),
                            batch_size=BATCH_SIZE, shuffle=False)
    ad_test_l  = DataLoader(CHTDataset(d["adapted_data"][:, sv:],   LOOKBACK, sv),
                            batch_size=BATCH_SIZE, shuffle=False)

    for sz in sizes:
        key = f"baseline_patchtst_{sz}"
        if result_exists(run_dir, key) and not force:
            logger.info(f"  [{sz}] Exists, skipping.")
            continue

        logger.info(f"\n{'='*60}\n  All-Grid PatchTST ({sz.capitalize()})\n{'='*60}")
        model    = build_model(sz)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Params: {n_params/1000:.1f}K")

        _, preds, trues, pe, te, t_secs, _ = run_training(
            f"All-Grid PatchTST ({sz.capitalize()})",
            model, TRAIN_CFG, train_l, val_l, test_l, device, logger=logger,
        )
        mae, rmse, wmape, ddtw, sdtw = calculate_metrics(te, pe)
        save_result(run_dir, key, preds, trues, {
            "model_name": f"All-Grid PatchTST ({sz.capitalize()})",
            "size": sz, "n_params_K": round(n_params/1000, 1),
            "train_time_s": round(t_secs, 1),
            "MAE": mae, "RMSE": rmse, "wMAPE": wmape,
            "Daily_DTW": ddtw, "Step_DTW": sdtw,
        })
        logger.info(f"  TEST MAE={mae:.4f}  wMAPE={wmape:.2f}%  Time={t_secs:.1f}s")

        if sz == "large" and do_adapt:
            zs_key  = "adapt_zs_patchtst"
            ret_key = "adapt_retrain_patchtst"

            if not result_exists(run_dir, zs_key) or force:
                zs_p, zs_t = eval_on_loader(model, ad_test_l, device, grid_mask=bc_mask)
                zm, *_ = calculate_metrics(zs_t, zs_p)
                save_result(run_dir, zs_key, zs_p, zs_t,
                            {"model_name": "All-Grid PatchTST (Large) Zero-Shot", "MAE": zm})
                logger.info(f"  [Adapt] Zero-Shot MAE on B+C = {zm:.4f}")

            if not result_exists(run_dir, ret_key) or force:
                logger.info("  [Adapt] Retraining on adapted data ...")
                gpu_cleanup(model)
                model = build_model("large")
                _, rp, rt, rpe, rte, rt_s, _ = run_training(
                    "All-Grid PatchTST (Large) Retrain",
                    model, TRAIN_CFG, ad_train_l, ad_val_l, ad_test_l,
                    device, eval_grid_mask=bc_mask, logger=logger,
                )
                rm, *_ = calculate_metrics(rte, rpe)
                save_result(run_dir, ret_key, rp, rt,
                            {"model_name": "All-Grid PatchTST (Large) Retrain",
                             "train_time_s": round(rt_s, 1), "MAE": rm})
                logger.info(f"  [Adapt] Retrain MAE on B+C = {rm:.4f}")

        gpu_cleanup(model)

    logger.info(f"=== {SCRIPT} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",    nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--no-adapt", dest="adapt", action="store_false")
    args = parser.parse_args()
    run(sizes=args.sizes, force=args.force, do_adapt=args.adapt)
