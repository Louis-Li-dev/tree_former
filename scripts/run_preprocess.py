"""
scripts/run_preprocess.py
=========================
Step 1 of the pipeline: load raw data, clean, select grids, build the
spatiotemporal R-Tree hierarchy, compute HMST attention masks, generate
the region-swap adaptation dataset, and save everything to runs/{run_id}/data/.

Run this once (with --fresh to clear an old run) before any training scripts.

Usage
-----
    python scripts/run_preprocess.py            # re-uses existing run if present
    python scripts/run_preprocess.py --fresh    # deletes old run, starts new
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import get_or_create_run_dir, setup_logger, RUNS_DIR
except ModuleNotFoundError:
    from _common import get_or_create_run_dir, setup_logger, RUNS_DIR

from hmst.utils import (
    NUM_GRIDS, LOOKBACK, BATCH_SIZE, K_ROOTS,
    build_forest_masks,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "2019_hourly_pop_byCHT.parquet"


# ─────────────────────────────────────────────────────────────────────────────
class SubTreeBuilder:
    """Bottom-up R-Tree hierarchy builder for a single root region."""

    def __init__(self, region_coords: np.ndarray, region_indices: np.ndarray):
        self.coords        = region_coords
        self.global_indices = region_indices

    def build_hierarchy(self) -> list:
        """
        Build a bottom-up level list.
        Level 0: each point is its own group [[0], [1], ..., [n-1]]
        Each subsequent level merges nearest-centroid pairs.
        Returns list of levels (each level is a list of index groups).
        """
        n_points     = len(self.coords)
        curr_groups  = [[i] for i in range(n_points)]
        levels       = [curr_groups]

        while len(curr_groups) > 1:
            next_groups = []
            used        = set()
            centers     = np.array([self.coords[g].mean(axis=0) for g in curr_groups])

            for i in range(len(curr_groups)):
                if i in used:
                    continue
                dists = np.linalg.norm(centers - centers[i], axis=1)
                dists[i] = np.inf
                for u in used:
                    dists[u] = np.inf
                nearest = int(np.argmin(dists))
                if dists[nearest] < np.inf:
                    next_groups.append(curr_groups[i] + curr_groups[nearest])
                    used.add(i)
                    used.add(nearest)
                else:
                    next_groups.append(curr_groups[i])
                    used.add(i)

            if len(next_groups) == len(curr_groups):
                break
            curr_groups = next_groups
            levels.append(curr_groups)

        return levels


# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(logger) -> tuple:
    """Load parquet, extract coordinates, apply the four cleaning filters."""
    logger.info(f"Loading data from {DATA_FILE} ...")
    df_raw    = pd.read_parquet(DATA_FILE)
    grids_raw = df_raw["Grid"].values
    data_raw  = df_raw.iloc[:, 1:].values.astype(np.float32)
    N_raw, T_raw = data_raw.shape
    logger.info(f"Loaded: {N_raw} grids, {T_raw} hours.")

    # Extract (lon, lat) from "lon_lat" grid labels
    coords = np.array([[float(g.split("_")[0]), float(g.split("_")[1])]
                       for g in grids_raw])

    # Filter 1: last 24 hours all-zero while earlier mean > 1 (data dropout)
    k_hours = 24
    type1 = [i for i in range(N_raw)
              if np.all(data_raw[i, -k_hours:] == 0.0) and data_raw[i, :-k_hours].mean() > 1.0]

    # Filter 2: large sliding-window level shift (W=168h, threshold=30, ratio>5)
    W = 168
    type2 = []
    type1_set = set(type1)
    for i in range(N_raw):
        if i in type1_set:
            continue
        s      = data_raw[i]
        cumsum = np.concatenate([[0.0], np.cumsum(s)])
        max_gap, best_ml, best_mr = -1.0, 0.0, 0.0
        for t in range(W, T_raw - W, 24):
            ml = (cumsum[t] - cumsum[t - W]) / W
            mr = (cumsum[t + W] - cumsum[t]) / W
            gap = abs(ml - mr)
            if gap > max_gap:
                max_gap, best_ml, best_mr = gap, ml, mr
        if max_gap > 30.0:
            rl = best_ml / (best_mr + 1e-5)
            rr = best_mr / (best_ml + 1e-5)
            if rl > 5.0 or rr > 5.0:
                type2.append(i)

    # Filter 3: zero ratio > 30% (sparse pulse grid)
    zero_ratios = (data_raw == 0.0).mean(axis=1)
    type3 = [i for i in range(N_raw) if zero_ratios[i] > 0.30]

    # Filter 4: any single zero hour (zero-intolerant grids not caught above)
    bad = set(type1) | set(type2) | set(type3)
    type4 = [i for i in range(N_raw) if i not in bad and (data_raw[i] == 0.0).any()]

    anomaly = set(type1) | set(type2) | set(type3) | set(type4)
    logger.info(
        f"Cleaning: filter1={len(type1)} filter2={len(type2)} "
        f"filter3={len(type3)} filter4={len(type4)} "
        f"total_removed={len(anomaly)} remaining={N_raw - len(anomaly)}"
    )

    clean_mask   = np.array([i not in anomaly for i in range(N_raw)])
    data_clean   = data_raw[clean_mask]
    coords_clean = coords[clean_mask]
    grids_clean  = grids_raw[clean_mask]

    del df_raw, data_raw
    gc.collect()

    return data_clean, coords_clean, grids_clean


# ─────────────────────────────────────────────────────────────────────────────
def select_grids(data_clean, coords_clean, grids_clean, logger) -> tuple:
    """Randomly sample NUM_GRIDS grids and cluster them into K_ROOTS regions."""
    N_clean = data_clean.shape[0]
    np.random.seed(42)
    sel_idx      = np.random.choice(N_clean, NUM_GRIDS, replace=False)
    sel_data     = data_clean[sel_idx].astype(np.float32)
    sel_coords   = coords_clean[sel_idx]
    sel_grids    = grids_clean[sel_idx]

    logger.info(f"Selected {NUM_GRIDS} grids from {N_clean} clean grids.")

    # Daily profile features for clustering
    daily = sel_data.reshape(NUM_GRIDS, -1, 24).mean(axis=1)       # (N, 24)
    daily_norm = (daily - daily.mean(axis=1, keepdims=True)) / (
        daily.std(axis=1, keepdims=True) + 1e-5
    )

    # Spatial KNN graph + agglomerative clustering
    knn   = kneighbors_graph(sel_coords, n_neighbors=8, include_self=False)
    clust = AgglomerativeClustering(n_clusters=K_ROOTS, connectivity=knn, linkage="ward")
    root_labels = clust.fit_predict(daily_norm)

    for r in range(K_ROOTS):
        cnt = int((root_labels == r).sum())
        logger.info(f"  Region {r}: {cnt} grids ({cnt/NUM_GRIDS*100:.1f}%)")

    return sel_data, sel_coords, sel_grids, sel_idx, root_labels


# ─────────────────────────────────────────────────────────────────────────────
def build_rtree(sel_coords, root_labels, logger) -> dict:
    """Build per-region R-Tree hierarchies."""
    forest_levels = {}
    for r in range(K_ROOTS):
        mask   = root_labels == r
        coords = sel_coords[mask]
        idx    = np.where(mask)[0]
        levels = SubTreeBuilder(coords, idx).build_hierarchy()
        forest_levels[r] = levels
        logger.info(f"  Region {r}: {len(levels)} R-Tree levels, "
                    f"root has {len(levels[-1])} groups.")
    return forest_levels


# ─────────────────────────────────────────────────────────────────────────────
def build_adaptation_data(sel_data, root_labels, logger) -> tuple:
    """
    Region-cascade swap (adaptation experiment):
      - Sort regions by size (largest first): A, B, C, ...
      - C grids take B-region time-series patterns
      - B grids take A-region time-series patterns
    Returns adapted_data, bc_indices, region_info dict.
    """
    np.random.seed(42)

    # Identify grids per region (by index in sel_data)
    region_sel = {r: [si for si in range(NUM_GRIDS) if root_labels[si] == r]
                  for r in range(K_ROOTS)}

    sorted_r = sorted(region_sel, key=lambda r: -len(region_sel[r]))
    A_idx    = sorted_r[0]
    B_idx    = sorted_r[1] if len(sorted_r) > 1 else sorted_r[0]
    C_idx    = sorted_r[2] if len(sorted_r) > 2 else B_idx

    grids_A, grids_B, grids_C = region_sel[A_idx], region_sel[B_idx], region_sel[C_idx]

    adapted_data = sel_data.copy()

    # Step 1: C <- B pattern
    b_samp = np.random.choice(len(grids_B), len(grids_C),
                               replace=len(grids_B) < len(grids_C))
    for i, c in enumerate(grids_C):
        adapted_data[c] = sel_data[grids_B[b_samp[i]]]

    # Step 2: B <- A pattern
    a_samp = np.random.choice(len(grids_A), len(grids_B),
                               replace=len(grids_A) < len(grids_B))
    for i, b in enumerate(grids_B):
        adapted_data[b] = sel_data[grids_A[a_samp[i]]]

    bc_indices = np.array(grids_B + grids_C, dtype=np.int64)
    region_info = {
        "A_region": A_idx, "B_region": B_idx, "C_region": C_idx,
        "grids_A": grids_A, "grids_B": grids_B, "grids_C": grids_C,
        "n_changed": len(grids_B) + len(grids_C),
        "pct_changed": (len(grids_B) + len(grids_C)) / NUM_GRIDS * 100,
    }
    logger.info(
        f"Adaptation swap: A={A_idx}({len(grids_A)}) -> "
        f"B={B_idx}({len(grids_B)}) -> C={C_idx}({len(grids_C)}); "
        f"{region_info['pct_changed']:.1f}% of grids changed."
    )

    return adapted_data, bc_indices, region_info


# ─────────────────────────────────────────────────────────────────────────────
def main(fresh: bool = False) -> None:
    t0 = time.time()

    # Handle --fresh: remove old run
    pointer = RUNS_DIR / "latest.txt"
    if fresh and pointer.exists():
        old_id  = pointer.read_text(encoding="utf-8").strip()
        old_dir = RUNS_DIR / old_id
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"[preprocess] Removed old run: {old_dir}")
        pointer.unlink()

    # Create new run directory
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_or_create_run_dir(run_id)
    logger  = setup_logger("preprocess", run_dir / "logs" / "preprocess.log")

    logger.info(f"=== run_preprocess.py  run_id={run_id} ===")
    logger.info(f"Config: NUM_GRIDS={NUM_GRIDS}, LOOKBACK={LOOKBACK}, "
                f"BATCH_SIZE={BATCH_SIZE}, K_ROOTS={K_ROOTS}")

    # 1. Load & clean
    data_clean, coords_clean, grids_clean = load_and_clean(logger)

    # 2. Select grids & cluster
    sel_data, sel_coords, sel_grids, sel_idx, root_labels = select_grids(
        data_clean, coords_clean, grids_clean, logger
    )
    del data_clean, coords_clean, grids_clean
    gc.collect()

    # 3. Build R-Tree hierarchies
    logger.info("Building per-region R-Tree hierarchies ...")
    forest_levels = build_rtree(sel_coords, root_labels, logger)

    # 4. Build HMST attention masks
    logger.info("Building HMST attention masks ...")
    hmst_masks = build_forest_masks(
        num_grids=NUM_GRIDS,
        K_roots=K_ROOTS,
        root_labels=root_labels,
        forest_levels=forest_levels,
    )
    logger.info(f"  {len(hmst_masks)} masks built, shape: {hmst_masks[0].shape}")

    # 5. Train/val/test splits (indices only)
    T_total     = sel_data.shape[1]
    split_train = int(T_total * 0.70)
    split_val   = int(T_total * 0.85)
    splits = {
        "T_total": T_total,
        "split_train": split_train,
        "split_val": split_val,
        "train_hours": split_train,
        "val_hours": split_val - split_train,
        "test_hours": T_total - split_val,
    }
    logger.info(f"Splits: train={splits['train_hours']}h  "
                f"val={splits['val_hours']}h  test={splits['test_hours']}h")

    # 6. Region-swap adaptation data
    logger.info("Generating region-swap adaptation dataset ...")
    adapted_data, bc_indices, region_info = build_adaptation_data(
        sel_data, root_labels, logger
    )

    # 7. Save everything
    data_dir = run_dir / "data"
    logger.info(f"Saving data to {data_dir} ...")

    np.save(data_dir / "selected_data.npy",    sel_data)
    np.save(data_dir / "selected_coords.npy",  sel_coords)
    np.save(data_dir / "selected_grids.npy",   sel_grids)
    np.save(data_dir / "selected_indices.npy", sel_idx)
    np.save(data_dir / "root_labels.npy",      root_labels)
    np.save(data_dir / "adapted_data.npy",     adapted_data)
    np.save(data_dir / "adapt_bc_indices.npy", bc_indices)
    for i, m in enumerate(hmst_masks):
        np.save(data_dir / f"hmst_mask_{i}.npy", m)

    with open(data_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    with open(data_dir / "adapt_region_info.json", "w", encoding="utf-8") as f:
        json.dump({k: (v if isinstance(v, (int, float, str)) else list(v))
                   for k, v in region_info.items()}, f, indent=2)

    with open(data_dir / "forest_levels.pkl", "wb") as f:
        pickle.dump(forest_levels, f)

    meta = {
        "run_id":    run_id,
        "NUM_GRIDS": NUM_GRIDS,
        "LOOKBACK":  LOOKBACK,
        "BATCH_SIZE": BATCH_SIZE,
        "K_ROOTS":   K_ROOTS,
        "T_total":   T_total,
        "created":   datetime.now().isoformat(),
    }
    with open(data_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Preprocessing complete in {elapsed:.1f}s. Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CHT data for training.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the existing run and start a new one.")
    args = parser.parse_args()
    main(fresh=args.fresh)
