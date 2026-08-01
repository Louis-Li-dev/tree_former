"""
scripts/run_ablation.py
=======================
Ablation study: train six R-Treeformer variants (all LARGE size).

Variants:
  Full               - complete model
  w/o RevIN          - no instance normalisation
  w/o Time Embedding - no hour/DoW embedding
  w/o Spatial Pos    - no spatial position embedding
  w/o R-Tree Mask    - uniform (non-hierarchical) attention
  Random Mask        - random hierarchical attention (control)

Usage
-----
    python scripts/run_ablation.py
    python scripts/run_ablation.py --variants Full "w/o RevIN"
    python scripts/run_ablation.py --force
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
        save_result, result_exists, load_run_data,
    )
except ModuleNotFoundError:
    from _common import (
        get_run_dir, setup_logger, gpu_cleanup,
        save_result, result_exists, load_run_data,
    )
from rtreeformer.model   import RTreeformer
from rtreeformer.train   import CHTDataset, run_training
from rtreeformer.utils   import LARGE, TRAIN_CFG, LOOKBACK, BATCH_SIZE, NUM_GRIDS, calculate_metrics

SCRIPT = "ablation"

VARIANT_KWARGS = {
    "Full": dict(
        use_revin=True,  use_rtree_mask=True,  random_mask=False,
        use_spatial_emb=True,  use_time_emb=True,
    ),
    "w/o RevIN": dict(
        use_revin=False, use_rtree_mask=True,  random_mask=False,
        use_spatial_emb=True,  use_time_emb=True,
    ),
    "w/o Time Embedding": dict(
        use_revin=True,  use_rtree_mask=True,  random_mask=False,
        use_spatial_emb=True,  use_time_emb=False,
    ),
    "w/o Spatial Position": dict(
        use_revin=True,  use_rtree_mask=True,  random_mask=False,
        use_spatial_emb=False, use_time_emb=True,
    ),
    "w/o R-Tree Mask": dict(
        use_revin=True,  use_rtree_mask=False, random_mask=False,
        use_spatial_emb=True,  use_time_emb=True,
    ),
    "Random Mask": dict(
        use_revin=True,  use_rtree_mask=True,  random_mask=True,
        use_spatial_emb=True,  use_time_emb=True,
    ),
}

ALL_VARIANTS = list(VARIANT_KWARGS.keys())


def variant_key(name: str) -> str:
    """Convert variant name to a filesystem-safe key."""
    return "ablation_" + name.lower().replace("/", "").replace(" ", "_").replace("-", "")


def make_loaders(data, splits):
    st, sv = splits["split_train"], splits["split_val"]
    kw = dict(batch_size=BATCH_SIZE, drop_last=False)
    tl = DataLoader(CHTDataset(data[:, :st],   LOOKBACK, 0),  shuffle=True,  **kw)
    vl = DataLoader(CHTDataset(data[:, st:sv], LOOKBACK, st), shuffle=False, **kw)
    el = DataLoader(CHTDataset(data[:, sv:],   LOOKBACK, sv), shuffle=False, **kw)
    return tl, vl, el


def run(variants=ALL_VARIANTS, force=False):
    run_dir = get_run_dir()
    logger  = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    logger.info(f"=== {SCRIPT}  run_dir={run_dir} ===")

    d          = load_run_data(run_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hmst_masks = [torch.tensor(m, dtype=torch.float32) for m in d["hmst_masks"]]
    N          = d["selected_data"].shape[0]
    logger.info(f"Device: {device}  N={N}")

    train_l, val_l, test_l = make_loaders(d["selected_data"], d["splits"])

    for var_name in variants:
        key = variant_key(var_name)
        if result_exists(run_dir, key) and not force:
            logger.info(f"  [{var_name}] Exists, skipping.")
            continue

        logger.info(f"\n{'='*50}\n  Ablation: {var_name}\n{'='*50}")
        kwargs = VARIANT_KWARGS[var_name]
        model  = RTreeformer(
            num_grids=N, lookback=LOOKBACK,
            d_model=LARGE["d_model"], d_k=LARGE["d_k"],
            num_layers=LARGE["num_layers"], dropout=0.1,
            forest_masks=hmst_masks,
            **kwargs,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Params: {n_params/1000:.1f}K")

        _, preds, trues, pe, te, t_secs, _ = run_training(
            var_name, model, TRAIN_CFG, train_l, val_l, test_l, device, logger=logger,
        )
        mae, rmse, wmape, ddtw, sdtw = calculate_metrics(te, pe)
        save_result(run_dir, key, preds, trues, {
            "model_name": var_name, "variant": var_name,
            "size": "large", "n_params_K": round(n_params/1000, 1),
            "train_time_s": round(t_secs, 1),
            "MAE": mae, "RMSE": rmse, "wMAPE": wmape,
            "Daily_DTW": ddtw, "Step_DTW": sdtw,
        })
        logger.info(f"  MAE={mae:.4f}  wMAPE={wmape:.2f}%  Time={t_secs:.1f}s")
        gpu_cleanup(model)

    logger.info(f"=== {SCRIPT} complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS, metavar="VARIANT")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(variants=args.variants, force=args.force)
