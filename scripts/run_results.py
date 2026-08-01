"""
scripts/run_results.py
======================
Auto-detects all available model results in runs/latest/results/ and generates
publication-quality tables and figures matching new_tree_building.ipynb.

Usage
-----
    python scripts/run_results.py               # use latest run
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

# Global plot style: Times New Roman, larger publication-quality font sizes
plt.rcParams.update({
    "font.family":        "Times New Roman",
    "font.size":          16,
    "axes.titlesize":     16,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "figure.dpi":         150,
    "axes.unicode_minus": False,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS
except ModuleNotFoundError:
    from _common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS

from rtreeformer.utils import calculate_metrics

SCRIPT = "results"

# ─────────────────────────────────────────────────────────────────────────────
# Key Mapping & Family Style Definitions
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_KEYS = {
    "Each-Grid LSTM (Small)":      "baseline_each_grid_lstm_small",
    "Each-Grid LSTM (Base)":       "baseline_each_grid_lstm_base",
    "Each-Grid LSTM (Large)":      "baseline_each_grid_lstm_large",
    "All-Grid LSTM (Small)":       "baseline_all_grid_lstm_small",
    "All-Grid LSTM (Base)":        "baseline_all_grid_lstm_base",
    "All-Grid LSTM (Large)":       "baseline_all_grid_lstm_large",
    "All-Grid DLinear (Small)":    "baseline_dlinear_small",
    "All-Grid DLinear (Base)":     "baseline_dlinear_base",
    "All-Grid DLinear (Large)":    "baseline_dlinear_large",
    "All-Grid PatchTST (Base)":    "baseline_patchtst_base",
    "All-Grid PatchTST (Large)":   "baseline_patchtst_large",
    "All-Grid Reformer (Base)":    "baseline_reformer_base",
    "All-Grid Reformer (Large)":   "baseline_reformer_large",
    "R-Treeformer (Base)":         ["rtreeformer_base", "hmstv2_base", "baseline_hmstv2_base"],
    "R-Treeformer (Large)":        ["rtreeformer_large", "hmstv2_large", "baseline_hmstv2_large"],
}

ABLATION_KEYS = {
    "Full":                   "ablation_full",
    "w/o RevIN":              "ablation_wo_revin",
    "w/o Time Embedding":     "ablation_wo_time_embedding",
    "w/o Spatial Position":   "ablation_wo_spatial_position",
    # "w/o R-Tree Mask":        "ablation_wo_rtree_mask",
    "Random Mask":            "ablation_random_mask",
}

ADAPT_KEYS = {
    "R-Treeformer Zero-Shot":       ["adapt_zs_rtreeformer_large", "adapt_zs_hmstv2", "adapt_zs_hmstv2_large"],
    "R-Treeformer Fine-Tune":       ["adapt_ft_rtreeformer", "adapt_ft_hmstv2"],
    "Each-Grid LSTM Zero-Shot":     ["adapt_zs_each_grid_lstm", "adapt_zs_each_grid_lstm_large"],
    "Each-Grid LSTM Retrain":       "adapt_retrain_each_grid_lstm",
    "All-Grid LSTM Zero-Shot":      ["adapt_zs_all_grid_lstm", "adapt_zs_all_grid_lstm_large"],
    "All-Grid LSTM Retrain":        ["adapt_retrain_all_grid_lstm", "adapt_retrain_lstm"],
    "All-Grid DLinear Zero-Shot":   ["adapt_zs_dlinear", "adapt_zs_dlinear_large"],
    "All-Grid DLinear Retrain":     ["adapt_retrain_dlinear", "adapt_retrain_dlinear_large"],
    "All-Grid PatchTST Zero-Shot":  ["adapt_zs_patchtst", "adapt_zs_patchtst_large"],
    "All-Grid PatchTST Retrain":    ["adapt_retrain_patchtst", "adapt_retrain_patchtst_large"],
    "All-Grid Reformer Zero-Shot":  ["adapt_zs_reformer", "adapt_zs_reformer_large"],
    "All-Grid Reformer Retrain":    ["adapt_retrain_reformer", "adapt_retrain_reformer_large"],
}

FAMILY_COLORS = {
    "R-Treeformer":    "#c0392b", 
    "Reformer":        "#8e44ad", 
    "PatchTST":        "#27ae60", 
    "Each-Grid LSTM":  "#3498db", 
    "All-Grid LSTM":   "#2980b9", 
    "DLinear":         "#636e72",
}

SIZE_SHAPES = {"Small": "^", "Base": "s", "Large": "o", "": "o"}
ADAPT_SHAPES = {"Zero-Shot": "o", "Retrain": "*", "Fine-Tune": "*"}
SIZE_ORDER = {"Small": 0, "Base": 1, "Large": 2, "": 0}

# Hardcoded offsets to avoid text overlapping
FAM_OFFSETS_COMPLEXITY = {
    "DLinear": (10, -14), "R-Treeformer": (10, -4), "Each-Grid LSTM": (10, -5), 
    "All-Grid LSTM": (10, -14), "PatchTST": (10, 8), "Reformer": (10, 8)
}
FAM_OFFSETS_SPEED = {
    "DLinear": (10, -14), "R-Treeformer": (10, -4), "Each-Grid LSTM": (10, -12), 
    "All-Grid LSTM": (10, -14), "PatchTST": (10, 8), "Reformer": (10, -4)
}
FAM_OFFSETS_SCATTER = {
    "DLinear": (10, -14), "R-Treeformer": (10, -4), "Each-Grid LSTM": (10, -14), 
    "All-Grid LSTM": (10, 6), "PatchTST": (10, 8), "Reformer": (10, -4)
}

def _family(name):
    if "HMST" in name or "R-Treeformer" in name: return "R-Treeformer"
    if "Reformer" in name: return "Reformer"
    if "PatchTST" in name: return "PatchTST"
    if "Each-Grid" in name: return "Each-Grid LSTM"
    if "All-Grid LSTM" in name: return "All-Grid LSTM"
    return "DLinear"

def _size(name):
    for tok in ("Small", "Base", "Large"):
        if tok in name: return tok
    return ""

def _adapt_type(name):
    if "Zero-Shot" in name: return "Zero-Shot"
    if "Retrain" in name: return "Retrain"
    if "Fine-Tune" in name: return "Fine-Tune"
    return "Retrain"

# ─────────────────────────────────────────────────────────────────────────────
def load_all(run_dir, key_map, logger):
    """Load available results and (re)compute metrics (DTW disabled for speed)."""
    rows = {}
    for display, keys in key_map.items():
        candidate_keys = [keys] if isinstance(keys, str) else keys
        found_key = next((k for k in candidate_keys if (run_dir / "results" / f"{k}.npz").exists()), None)
        if not found_key: continue

        preds, trues, meta = load_result(run_dir, found_key)
        metrics = calculate_metrics(trues, preds, dtw_grids=0) 
        mae, rmse, wmape = metrics[0], metrics[1], metrics[2]
        
        params_k = meta.get("n_params_K") or KNOWN_PARAMS.get(display, float("nan"))
        rows[display] = {
            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "wMAPE (%)": round(wmape, 2),
            "Params (K)": params_k, "Train (s)": meta.get("train_time_s", float("nan")),
            "_preds": preds, "_trues": trues,
        }
        logger.info(f"  Loaded: {display}  MAE={mae:.4f}")
    return rows

def save_table(rows: dict, path: Path, title: str) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    display_cols = ["MAE", "RMSE", "wMAPE (%)", "Params (K)", "Train (s)"]
    df = pd.DataFrame({k: {c: v[c] for c in display_cols if c in v} for k, v in rows.items()}).T
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{df.to_markdown(floatfmt='.4f')}\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
def plot_benchmark_tradeoffs(df_bench, plots_dir, logger):
    """Complexity vs MAE and Speed vs wMAPE with clean broken axes and bottom legends."""
    if df_bench.empty: return

    fig = plt.figure(figsize=(16, 6.8))
    # Reduced broken axis gap size (width ratio 0.15)
    gs = fig.add_gridspec(1, 5, width_ratios=[3.8, 1, 0.15, 3.8, 1])
    
    ax_cl = fig.add_subplot(gs[0, 0])
    ax_cr = fig.add_subplot(gs[0, 1], sharey=ax_cl)
    ax_sl = fig.add_subplot(gs[0, 3])
    ax_sr = fig.add_subplot(gs[0, 4], sharey=ax_sl)

    def draw_broken_axis(ax_l, ax_r, ratio=3.8):
        ax_l.spines['right'].set_visible(False)
        ax_r.spines['left'].set_visible(False)
        ax_l.tick_params(labelright=False)
        ax_r.tick_params(left=False, labelleft=False, right=False, labelright=False)
        d = 0.015
        kwargs = dict(transform=ax_l.transAxes, color='k', clip_on=False, lw=1.2)
        ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax_r.transAxes)
        ax_r.plot((-d * ratio, +d * ratio), (-d, +d), **kwargs)
        ax_r.plot((-d * ratio, +d * ratio), (1 - d, 1 + d), **kwargs)

    draw_broken_axis(ax_cl, ax_cr, 3.8)
    draw_broken_axis(ax_sl, ax_sr, 3.8)

    families = {}
    for name, row in df_bench.iterrows():
        fam = _family(name)
        families.setdefault(fam, []).append((SIZE_ORDER[_size(name)], float(row["Params (K)"]), float(row["MAE"]), 
                                             float(row["Train (s)"]), float(row["wMAPE (%)"]), _size(name)))

    for fam, pts in families.items():
        pts_sorted = sorted(pts, key=lambda t: t[0])
        color = FAMILY_COLORS.get(fam, "gray")
        
        # Plot Complexity (Params vs MAE)
        cx, cy = [p[1] for p in pts_sorted], [p[2] for p in pts_sorted]
        ax_cl.plot(cx, cy, ls="--", lw=1.5, color=color, alpha=0.4)
        ax_cr.plot(cx, cy, ls="--", lw=1.5, color=color, alpha=0.4)
        
        # Plot Speed (Train Time vs wMAPE)
        sx, sy = [p[3] for p in pts_sorted], [p[4] for p in pts_sorted]
        ax_sl.plot(sx, sy, ls="--", lw=1.5, color=color, alpha=0.4)
        ax_sr.plot(sx, sy, ls="--", lw=1.5, color=color, alpha=0.4)

        for p in pts_sorted:
            marker = SIZE_SHAPES.get(p[5], "o")
            # Complexity scatter (Half transparent: alpha=0.55)
            ax_cl.scatter(p[1], p[2], c=color, marker=marker, s=140, edgecolors="black", alpha=0.55, zorder=4)
            ax_cr.scatter(p[1], p[2], c=color, marker=marker, s=140, edgecolors="black", alpha=0.55, zorder=4)
            # Speed scatter (Half transparent: alpha=0.55)
            ax_sl.scatter(p[3], p[4], c=color, marker=marker, s=140, edgecolors="black", alpha=0.55, zorder=4)
            ax_sr.scatter(p[3], p[4], c=color, marker=marker, s=140, edgecolors="black", alpha=0.55, zorder=4)

        # Annotate model names (preventing overflow & point overlap)
        if pts_sorted:
            p_max = pts_sorted[-1]
            c_off = FAM_OFFSETS_COMPLEXITY.get(fam, (10, 0))
            s_off = FAM_OFFSETS_SPEED.get(fam, (10, 0))
            
            # Complexity Annotations
            if fam == "Each-Grid LSTM":
                ax_cr.annotate(fam, (p_max[1], p_max[2]), xytext=(-115, -4), textcoords="offset points", fontsize=11, zorder=5)
            else:
                ax_cl.annotate(fam, (p_max[1], p_max[2]), xytext=c_off, textcoords="offset points", fontsize=11, zorder=5)
                ax_cr.annotate(fam, (p_max[1], p_max[2]), xytext=c_off, textcoords="offset points", fontsize=11, zorder=5)
            
            # Speed Annotations
            ax_sl.annotate(fam, (p_max[3], p_max[4]), xytext=s_off, textcoords="offset points", fontsize=11, zorder=5)
            ax_sr.annotate(fam, (p_max[3], p_max[4]), xytext=s_off, textcoords="offset points", fontsize=11, zorder=5)

    # Set appropriate limits to prevent clipping & overflow
    ax_cl.set_xlim(-100, 1800)
    ax_cr.set_xlim(4900, 5350)
    ax_cl.set_ylim(37, 62)

    ax_sl.set_xlim(-20, 260)
    ax_sr.set_xlim(850, 1380)
    ax_sl.set_ylim(2.7, 4.6)

    # Styling and Grid
    for ax in [ax_cl, ax_cr, ax_sl, ax_sr]:
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#FCFCFC")
        
    ax_cl.set_ylabel("MAE", fontsize=14)
    ax_sl.set_ylabel("wMAPE (%)", fontsize=14)
    
    # Titles & X Labels
    ax_cl.set_title("Complexity vs. Accuracy", fontsize=15, fontweight="bold", x=0.6)
    ax_sl.set_title("Speed vs. Accuracy", fontsize=15, fontweight="bold", x=0.6)
    ax_cl.set_xlabel("Model Complexity (Params in K)", fontsize=14, x=0.6, labelpad=10)
    ax_sl.set_xlabel("Training Time (seconds)", fontsize=14, x=0.6, labelpad=10)
    
    fig.suptitle("Model Performance Trade-offs (Lower-Left is Better)", fontsize=17, fontweight="bold", y=0.97)

    # Side-by-side horizontal legends at the bottom
    fam_handles = [mpatches.Patch(color=c, label=f, alpha=0.65) for f, c in FAMILY_COLORS.items() if f in [_family(n) for n in df_bench.index]]
    size_handles = [Line2D([0], [0], marker=m, color='w', label=s, markerfacecolor='gray', markersize=10, markeredgecolor='k', alpha=0.65) 
                    for s, m in SIZE_SHAPES.items() if s != ""]
    
    fig.legend(handles=fam_handles, title="Model Family", loc="lower center", bbox_to_anchor=(0.33, 0.01), ncol=3, frameon=True, fontsize=11, title_fontsize=12)
    fig.legend(handles=size_handles, title="Model Size", loc="lower center", bbox_to_anchor=(0.78, 0.01), ncol=len(size_handles), frameon=True, fontsize=11, title_fontsize=12)

    plt.subplots_adjust(bottom=0.22, top=0.88)
    fig.savefig(plots_dir / "benchmark_tradeoffs.png", dpi=150)
    plt.close(fig)


def filter_largest_per_family(preds_store: dict) -> dict:
    family_models = {}
    for name, data in preds_store.items():
        fam = _family(name)
        val = SIZE_ORDER.get(_size(name), 0)
        if fam not in family_models or val > family_models[fam][0]:
            family_models[fam] = (val, name, data)
    return {name: data for _, name, data in family_models.values()}


def plot_benchmark_trajectories(bench_rows, plots_dir, logger):
    """Plot Benchmark Trajectories (Best & Worst, bottom legend, transparent lines if overlapping)."""
    if not bench_rows: return
    preds_store = filter_largest_per_family({k: (v["_preds"], v["_trues"]) for k, v in bench_rows.items()})

    trues, N_min = next(iter(preds_store.values()))[1], min(t.shape[1] for p, t in preds_store.values())
    grid_maes = {k: np.mean(np.abs(p[:, :N_min] - t[:, :N_min]), axis=0) for k, (p, t) in preds_store.items()}

    avg_mae = np.mean(np.array(list(grid_maes.values())), axis=0)
    best_grid, worst_grid = int(np.argmin(avg_mae)), int(np.argmax(avg_mae))

    T_SHOW = min(24, trues.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    
    for ax, gid, title in zip(axes, [best_grid, worst_grid], ["Best Case Grid", "Worst Case Grid"]):
        ax.plot(trues[:T_SHOW, gid], color="black", lw=2.5, label="Ground Truth", alpha=0.9, zorder=10)
        
        for name, (preds, _) in preds_store.items():
            if gid >= preds.shape[1]: continue
            fam = _family(name)
            col = FAMILY_COLORS.get(fam, "gray")
            mae_val = np.mean(np.abs(preds[:, gid] - trues[:, gid]))
            
            is_rtree = "R-Treeformer" in fam or "HMST" in name
            lw, ls, zo, al = (2.0, "-", 5, 0.9) if is_rtree else (1.2, "--", 3, 0.7)
            
            label_text = f"{fam} (MAE: {mae_val:.2f})"
            ax.plot(preds[:T_SHOW, gid], color=col, lw=lw, ls=ls, label=label_text, alpha=al, zorder=zo)

        ax.set_title(f"Benchmark: Grid {gid} ({title} - Next 24h)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Population Flow", fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hours", fontsize=13)
    
    # Extract legends and move them to bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True, fontsize=11)
    
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(plots_dir / "benchmark_trajectories.png", dpi=150)
    plt.close(fig)


def plot_ablation_bars(abl_rows, plots_dir, logger):
    if not abl_rows: return
    sorted_items = sorted(abl_rows.items(), key=lambda x: x[1]["MAE"])
    names, vals = [x[0] for x in sorted_items], [x[1]["MAE"] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color='#3498db', edgecolor='black', lw=1.0, width=0.6, alpha=0.65)
    ax.set_title("Ablation Study: MAE", fontsize=14, fontweight='bold')
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(min(vals) * 0.9, max(vals) * 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + (max(vals) - min(vals))*0.015,
                f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "ablation_bars.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_adaptation_trajectories(adapt_rows, plots_dir, logger):
    """Plot Adaptation Trajectories with alpha transparency for Retrain models & bottom legend."""
    if not adapt_rows: return
    preds_store = {k: (v["_preds"], v["_trues"]) for k, v in adapt_rows.items()}
    trues, N_min = next(iter(preds_store.values()))[1], min(t.shape[1] for p, t in preds_store.values())
    grid_maes = {k: np.mean(np.abs(p[:, :N_min] - t[:, :N_min]), axis=0) for k, (p, t) in preds_store.items()}

    avg_mae = np.mean(np.array(list(grid_maes.values())), axis=0)
    best_grid, worst_grid = int(np.argmin(avg_mae)), int(np.argmax(avg_mae))

    T_SHOW = min(24, trues.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    
    for ax, gid, title in zip(axes, [best_grid, worst_grid], ["Best Case Grid", "Worst Case Grid"]):
        ax.plot(trues[:T_SHOW, gid], color="black", lw=2.5, label="Ground Truth", alpha=0.9, zorder=10)
        
        for name, (preds, _) in preds_store.items():
            if gid >= preds.shape[1]: continue
            fam = _family(name)
            col = FAMILY_COLORS.get(fam, "gray")
            mae_val = np.mean(np.abs(preds[:, gid] - trues[:, gid]))
            
            # Use semi-transparency (0.4) and thinner lines for Retrain to avoid blocking Zero-shot
            is_retrain = _adapt_type(name) in ["Retrain", "Fine-Tune"]
            lw = 1.2 if is_retrain else 2.0
            ls = "--" if is_retrain else "-"
            al = 0.4 if is_retrain else 0.85
            zo = 4 if is_retrain else 5
            
            label_text = f"{name} (MAE: {mae_val:.2f})"
            ax.plot(preds[:T_SHOW, gid], color=col, lw=lw, ls=ls, label=label_text, alpha=al, zorder=zo)

        ax.set_title(f"Adaptation: Grid {gid} ({title} - Next 24h)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Population Flow", fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hours", fontsize=13)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True, fontsize=10)

    plt.subplots_adjust(bottom=0.25)
    fig.savefig(plots_dir / "adapt_trajectories.png", dpi=150)
    plt.close(fig)


def plot_adaptation_scatter(adapt_rows, plots_dir, logger):
    """Adaptation Scatter with clean broken X-Axis, staggered labels, transparency, and full expanded side-by-side legend."""
    if not adapt_rows: return

    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.08)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    fam_pts = {}
    for key, vals in adapt_rows.items():
        fam = _family(key)
        x, y = vals.get("Train (s)", 0.0), vals["MAE"]
        if np.isnan(x): x = 0.0
        fam_pts.setdefault(fam, []).append((x, y, _adapt_type(key)))

    for fam, pts in fam_pts.items():
        color = FAMILY_COLORS.get(fam, "gray")
        if len(pts) > 1:
            pts.sort(key=lambda t: t[0]) 
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            for ax in (ax1, ax2):
                ax.plot(xs, ys, ls="--", color=color, alpha=0.4, zorder=2)
        
        for p in pts:
            marker = ADAPT_SHAPES.get(p[2], "o")
            al = 0.55
            
            for ax in (ax1, ax2):
                ax.scatter(p[0], p[1], color=color, marker=marker, s=150, edgecolors="black", lw=1.2, alpha=al, zorder=5)

            if p[2] in ["Retrain", "Fine-Tune"]:
                off = FAM_OFFSETS_SCATTER.get(fam, (10, 0))
                ax1.annotate(fam, (p[0], p[1]), textcoords="offset points", xytext=off, fontsize=11, zorder=6)
                ax2.annotate(fam, (p[0], p[1]), textcoords="offset points", xytext=off, fontsize=11, zorder=6)

    # Broken Axis config with break marks
    ax1.set_xlim(-10, 200)
    ax2.set_xlim(1150, 1300)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1.2)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs), ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d * 4, +d * 4), (-d, +d), **kwargs), ax2.plot((-d * 4, +d * 4), (1 - d, 1 + d), **kwargs)

    ax1.set_xlabel("Adaptation Time (seconds)", fontsize=14, labelpad=10)
    ax1.xaxis.set_label_coords(0.6, -0.1)
    ax1.set_ylabel("MAE on Changed Grids (B+C)", fontsize=14)
    fig.suptitle("Adaptation Speed vs. Accuracy (lower-left = better)", fontsize=16, fontweight='bold', y=0.96)
    
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    fam_handles = [mpatches.Patch(color=c, label=f, alpha=0.65) for f, c in FAMILY_COLORS.items() if f in fam_pts]
    type_handles = [Line2D([0], [0], marker=m, color='w', label=s, markerfacecolor='gray', markersize=10, markeredgecolor='k', alpha=0.65) 
                    for s, m in ADAPT_SHAPES.items() if s != "Fine-Tune"]
    
    fig.legend(handles=fam_handles, title="Model Family", loc="lower center", bbox_to_anchor=(0.33, 0.01), ncol=3, frameon=True, fontsize=11, title_fontsize=12)
    fig.legend(handles=type_handles, title="Method", loc="lower center", bbox_to_anchor=(0.80, 0.01), ncol=len(type_handles), frameon=True, fontsize=11, title_fontsize=12)

    plt.subplots_adjust(bottom=0.22, top=0.9)
    fig.savefig(plots_dir / "adapt_scatter.png", dpi=150)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    run_dir   = get_run_dir()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    logger    = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    
    table_path = plots_dir / "results_tables.md"
    table_path.write_text(f"# Results -- run: {run_dir.name}\n", encoding="utf-8")

    bench_rows = load_all(run_dir, BENCHMARK_KEYS, logger)
    if bench_rows:
        df_bench = save_table(bench_rows, table_path, "Benchmark Results")
        plot_benchmark_tradeoffs(df_bench, plots_dir, logger)
        plot_benchmark_trajectories(bench_rows, plots_dir, logger)

    abl_rows = load_all(run_dir, ABLATION_KEYS, logger)
    if abl_rows:
        save_table(abl_rows, table_path, "Ablation Study")
        plot_ablation_bars(abl_rows, plots_dir, logger)

    adapt_rows = load_all(run_dir, ADAPT_KEYS, logger)
    if adapt_rows:
        save_table(adapt_rows, table_path, "Adaptation Results")
        plot_adaptation_trajectories(adapt_rows, plots_dir, logger)
        plot_adaptation_scatter(adapt_rows, plots_dir, logger)

    print(f"\nDone. Saved tables and plots to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()