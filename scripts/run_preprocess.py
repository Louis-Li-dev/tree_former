"""
scripts/run_preprocess.py
=========================
Step 1 of the pipeline: load raw data, clean, select grids, build the
spatiotemporal R-Tree hierarchy, compute HMST attention masks, generate
the region-swap adaptation dataset, save everything to runs/{run_id}/data/,
and generate publication-quality preprocessing & adaptation setup figures.

Usage
-----
    python scripts/run_preprocess.py            # re-uses existing run if present
    python scripts/run_preprocess.py --fresh    # deletes old run, starts new
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Global plot style: Times New Roman, publication-quality
plt.rcParams.update({
    "font.family":        "Times New Roman",
    "font.size":          14,
    "axes.titlesize":     15,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "figure.dpi":         120,
    "axes.unicode_minus": False,
})

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import get_or_create_run_dir, setup_logger, RUNS_DIR
except ModuleNotFoundError:
    from _common import get_or_create_run_dir, setup_logger, RUNS_DIR

from rtreeformer.utils import (
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

    coords = np.array([[float(g.split("_")[0]), float(g.split("_")[1])]
                       for g in grids_raw])

    # Filter 1: last 24 hours all-zero while earlier mean > 1
    k_hours = 24
    type1 = [i for i in range(N_raw)
              if np.all(data_raw[i, -k_hours:] == 0.0) and data_raw[i, :-k_hours].mean() > 1.0]

    # Filter 2: large sliding-window level shift
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

    # Filter 3: zero ratio > 30%
    zero_ratios = (data_raw == 0.0).mean(axis=1)
    type3 = [i for i in range(N_raw) if zero_ratios[i] > 0.30]

    # Filter 4: any zero value
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

    daily = sel_data.reshape(NUM_GRIDS, -1, 24).mean(axis=1)
    daily_norm = (daily - daily.mean(axis=1, keepdims=True)) / (
        daily.std(axis=1, keepdims=True) + 1e-5
    )

    knn   = kneighbors_graph(sel_coords, n_neighbors=8, include_self=False)
    clust = AgglomerativeClustering(n_clusters=K_ROOTS, connectivity=knn, linkage="ward")
    root_labels = clust.fit_predict(daily_norm)

    for r in range(K_ROOTS):
        cnt = int((root_labels == r).sum())
        logger.info(f"  Region {r}: {cnt} grids ({cnt/NUM_GRIDS*100:.1f}%)")

    return sel_data, sel_coords, sel_grids, sel_idx, root_labels, daily_norm


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
    """Region-cascade swap dataset generation."""
    np.random.seed(42)

    region_sel = {r: [si for si in range(NUM_GRIDS) if root_labels[si] == r]
                  for r in range(K_ROOTS)}

    sorted_r = sorted(region_sel, key=lambda r: -len(region_sel[r]))
    A_idx    = sorted_r[0]
    B_idx    = sorted_r[1] if len(sorted_r) > 1 else sorted_r[0]
    C_idx    = sorted_r[2] if len(sorted_r) > 2 else B_idx

    grids_A, grids_B, grids_C = region_sel[A_idx], region_sel[B_idx], region_sel[C_idx]

    adapted_data = sel_data.copy()
    orig_B_data  = sel_data[grids_B].copy()
    orig_C_data  = sel_data[grids_C].copy()

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

    return adapted_data, bc_indices, region_info, orig_B_data, orig_C_data


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & Setup Plot Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_preprocessing_plots(
    sel_coords, root_labels, daily_norm, forest_levels,
    sel_data, adapted_data, region_info, orig_B_data, orig_C_data, plots_dir, logger
):
    plots_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Generating preprocessing & adaptation setup figures ...")

    # Figure 1: Spatial Root Partition Map & 24h Daily Profiles
    colors = [plt.cm.tab10(i % 10) for i in range(K_ROOTS)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for r in range(K_ROOTS):
        mask = (root_labels == r)
        ax.scatter(sel_coords[mask, 0], sel_coords[mask, 1],
                   color=colors[r], label=f'Region {r}', s=15, alpha=0.8)
    ax.set_title(f"Taipei Spatial Root Partition Map ({K_ROOTS} Clusters)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(markerscale=3); ax.grid(True, alpha=0.3)

    ax = axes[1]
    hours = np.arange(24)
    for r in range(K_ROOTS):
        mask = (root_labels == r)
        profile_mean = daily_norm[mask].mean(axis=0)
        profile_std  = daily_norm[mask].std(axis=0)
        ax.plot(hours, profile_mean, color=colors[r], label=f'Region {r} Profile', linewidth=2.5)
        ax.fill_between(hours, profile_mean - 0.2*profile_std, profile_mean + 0.2*profile_std,
                        color=colors[r], alpha=0.15)
    ax.set_title("Normalized 24-Hour Population Profiles by Region", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day (0-23)"); ax.set_ylabel("Normalized Population Flow (Z-score)")
    ax.set_xticks(np.arange(0, 25, 4)); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "preprocessing_spatial_clusters.png", dpi=150)
    plt.close(fig)

    # Figure 2: R-Tree Dendrograms
    def draw_tree_structure(levels, ax, title, colors, r_color_idx):
        tree_depth = len(levels)
        plot_depth = min(4, tree_depth)
        nodes_to_plot = []

        root_group = levels[-1][0]
        nodes_to_plot.append({
            'level_idx': tree_depth - 1, 'group': root_group,
            'x': 0.5, 'y': plot_depth,
            'parent_x': None, 'parent_y': None,
            'label': f"Root\n({len(root_group)} Grids)"
        })

        for current_plot_y in range(plot_depth - 1, 0, -1):
            parent_nodes = [n for n in nodes_to_plot if n['y'] == current_plot_y + 1]
            if not parent_nodes: break
            current_level_idx = parent_nodes[0]['level_idx'] - 1
            if current_level_idx < 0: break
            current_groups = levels[current_level_idx]

            for parent_node in parent_nodes:
                parent_group = parent_node['group']
                children = [grp for grp in current_groups
                            if set(grp).issubset(set(parent_group)) and len(grp) < len(parent_group)]
                if not children: continue
                children = sorted(children, key=lambda g: len(g), reverse=True)
                dx = 0.5 * (0.5 ** (plot_depth - current_plot_y))

                if len(children) == 1:
                    child = children[0]
                    nodes_to_plot.append({
                        'level_idx': current_level_idx, 'group': child,
                        'x': parent_node['x'], 'y': current_plot_y,
                        'parent_x': parent_node['x'], 'parent_y': parent_node['y'],
                        'label': f"({len(child)} Grids)"
                    })
                elif len(children) >= 2:
                    left_child, right_child = children[0], children[1]
                    nodes_to_plot.append({
                        'level_idx': current_level_idx, 'group': left_child,
                        'x': parent_node['x'] - dx, 'y': current_plot_y,
                        'parent_x': parent_node['x'], 'parent_y': parent_node['y'],
                        'label': f"({len(left_child)} Grids)"
                    })
                    nodes_to_plot.append({
                        'level_idx': current_level_idx, 'group': right_child,
                        'x': parent_node['x'] + dx, 'y': current_plot_y,
                        'parent_x': parent_node['x'], 'parent_y': parent_node['y'],
                        'label': f"({len(right_child)} Grids)"
                    })

        for node in nodes_to_plot:
            if node['parent_x'] is not None:
                ax.plot([node['x'], node['parent_x']], [node['y'], node['parent_y']],
                        color='#999999', linestyle='-', linewidth=3.0, zorder=1)
            is_leaf    = (node['y'] == 1)
            edge_color = colors[r_color_idx]
            fs = 14 if not is_leaf else 12
            ax.text(node['x'], node['y'], node['label'],
                    ha='center', va='center', fontsize=fs, fontweight='bold',
                    color='#111111', zorder=2,
                    bbox=dict(boxstyle="round,pad=0.5",
                              fc="white" if not is_leaf else "#F9F9F9",
                              ec=edge_color, lw=2.5, alpha=1.0))

        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.5, plot_depth + 0.5)
        ax.set_title(title, fontsize=16, fontweight='bold', color='black', pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#BBBBBB')
            spine.set_linestyle('--'); spine.set_linewidth(2.0)
        ax.set_facecolor('#FCFCFC')

    n_cols = 2 if K_ROOTS == 4 else min(K_ROOTS, 3)
    n_rows = math.ceil(K_ROOTS / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
    axes_flat = [axes] if n_rows == 1 and n_cols == 1 else np.array(axes).flatten()
    tree_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for r in range(K_ROOTS):
        ax = axes_flat[r]
        draw_tree_structure(forest_levels[r], ax, f"Region {r} Dendrogram",
                            tree_colors, r % len(tree_colors))

    for r in range(K_ROOTS, len(axes_flat)):
        fig.delaxes(axes_flat[r])

    plt.suptitle("Hierarchical R-Tree Structures for All Regions",
                 fontsize=22, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=3.0, h_pad=3.0)
    fig.savefig(plots_dir / "preprocessing_rtree_dendrograms.png", dpi=150)
    plt.close(fig)

    # Figure F1: Cascade Swap Map
    A_idx, B_idx, C_idx = region_info["A_region"], region_info["B_region"], region_info["C_region"]
    grids_A, grids_B, grids_C = region_info["grids_A"], region_info["grids_B"], region_info["grids_C"]
    pct_changed = region_info["pct_changed"]

    fig, ax = plt.subplots(figsize=(9, 7))
    region_color = {A_idx: '#3498db', B_idx: '#e74c3c', C_idx: '#f39c12'}
    for r in range(K_ROOTS):
        mask_r = (root_labels == r)
        color  = region_color.get(r, '#bdc3c7')
        label  = {A_idx: f'A (unchanged, {len(grids_A)} grids)',
                   B_idx: f'B <- A pattern ({len(grids_B)} grids)',
                   C_idx: f'C <- B pattern ({len(grids_C)} grids)'}.get(r, f'Region {r} (unchanged)')
        ax.scatter(sel_coords[mask_r, 0], sel_coords[mask_r, 1],
                   color=color, label=label, s=15, alpha=0.8)

    ax.set_title(f"F1: Cascade Swap Map ({pct_changed:.1f}% of grids changed)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(markerscale=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "adapt_f1_swap_map.png", dpi=150)
    plt.close(fig)

    # Figure F2: Before/After time series
    n_ex = min(3, len(grids_B), len(grids_C))
    fig, axes = plt.subplots(2, n_ex, figsize=(18, 8))
    for ax_row, (g_list, orig, region_label, color) in enumerate([
            (grids_B[:n_ex], orig_B_data, f'Region B ({A_idx}->{B_idx})', '#e74c3c'),
            (grids_C[:n_ex], orig_C_data, f'Region C ({B_idx}->{C_idx})', '#f39c12')]):
        for j in range(n_ex):
            ax = axes[ax_row, j]
            T_show = min(168, sel_data.shape[1])
            ax.plot(orig[j, :T_show], color='gray', ls='--', lw=1.5, label='Before', alpha=0.8)
            ax.plot(adapted_data[g_list[j], :T_show], color=color, lw=1.5, label='After')
            ax.set_title(f"{region_label}\nGrid {g_list[j]}", fontsize=10)
            ax.set_xlabel("Hours"); ax.set_ylabel("Population")
            if j == 0: ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    plt.suptitle("F2: Before vs After Swap (first 168h)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(plots_dir / "adapt_f2_series_swap.png", dpi=150)
    plt.close(fig)

    # Figure F3: 24h Daily Profile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hours_x = np.arange(24)
    for ax, (g_list, orig, label, color) in zip(axes, [
            (grids_B, orig_B_data, f'Region B (A->B swap)', '#e74c3c'),
            (grids_C, orig_C_data, f'Region C (B->C swap)', '#f39c12')]):
        before_prof = orig.reshape(len(g_list), -1, 24).mean(axis=(0, 1))
        after_prof  = adapted_data[g_list].reshape(len(g_list), -1, 24).mean(axis=(0, 1))
        ax.plot(hours_x, before_prof, color='gray', ls='--', lw=2, label='Before')
        ax.plot(hours_x, after_prof,  color=color,          lw=2, label='After')
        ax.fill_between(hours_x, before_prof, after_prof, alpha=0.15, color=color)
        ax.set_title(f"F3: 24h Daily Profile -- {label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Population")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "adapt_f3_daily_profile.png", dpi=150)
    plt.close(fig)

    # Figure F4: Difference Coefficient Violin plot
    def diff_coeff(before, after):
        bp = before.reshape(-1, 24).mean(axis=0)
        ap = after.reshape(-1, 24).mean(axis=0)
        c  = np.corrcoef(bp, ap)[0, 1]
        return 1 - c

    dc_B = [diff_coeff(sel_data[b], adapted_data[b]) for b in grids_B]
    dc_C = [diff_coeff(sel_data[c], adapted_data[c]) for c in grids_C]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.violinplot([dc_B, dc_C], positions=[1, 2], showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Region B\n(A->B, n={len(dc_B)})', f'Region C\n(B->C, n={len(dc_C)})'])
    ax.axhline(np.mean(dc_B), color='#e74c3c', ls='--', alpha=0.7, label=f'B mean={np.mean(dc_B):.3f}')
    ax.axhline(np.mean(dc_C), color='#f39c12', ls='--', alpha=0.7, label=f'C mean={np.mean(dc_C):.3f}')
    ax.set_title("F4: Distribution of Difference Coefficients\n(1-Pearson, 0=identical, 2=opposite)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Diff Coeff (1 - Pearson Corr)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "adapt_f4_diff_coeff_violin.png", dpi=150)
    plt.close(fig)

    logger.info("Preprocessing plots successfully generated and saved to plots/.")


# ─────────────────────────────────────────────────────────────────────────────
def main(fresh: bool = False) -> None:
    t0 = time.time()

    pointer = RUNS_DIR / "latest.txt"
    if fresh and pointer.exists():
        old_id  = pointer.read_text(encoding="utf-8").strip()
        old_dir = RUNS_DIR / old_id
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"[preprocess] Removed old run: {old_dir}")
        pointer.unlink()

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_or_create_run_dir(run_id)
    logger  = setup_logger("preprocess", run_dir / "logs" / "preprocess.log")

    logger.info(f"=== run_preprocess.py  run_id={run_id} ===")
    logger.info(f"Config: NUM_GRIDS={NUM_GRIDS}, LOOKBACK={LOOKBACK}, "
                f"BATCH_SIZE={BATCH_SIZE}, K_ROOTS={K_ROOTS}")

    # 1. Load & clean
    data_clean, coords_clean, grids_clean = load_and_clean(logger)

    # 2. Select grids & cluster
    sel_data, sel_coords, sel_grids, sel_idx, root_labels, daily_norm = select_grids(
        data_clean, coords_clean, grids_clean, logger
    )
    del data_clean, coords_clean, grids_clean
    gc.collect()

    # 3. Build R-Tree hierarchies
    logger.info("Building per-region R-Tree hierarchies ...")
    forest_levels = build_rtree(sel_coords, root_labels, logger)

    # 4. Build R-Treeformer attention masks
    logger.info("Building R-Treeformer attention masks ...")
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
    adapted_data, bc_indices, region_info, orig_B_data, orig_C_data = build_adaptation_data(
        sel_data, root_labels, logger
    )

    # 7. Generate Preprocessing Plots
    plots_dir = run_dir / "plots"
    generate_preprocessing_plots(
        sel_coords, root_labels, daily_norm, forest_levels,
        sel_data, adapted_data, region_info, orig_B_data, orig_C_data, plots_dir, logger
    )

    # 8. Save everything
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
