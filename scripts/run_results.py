"""
scripts/run_results.py
======================
Auto-detects all available model results in runs/latest/results/ and generates
publication-quality tables and figures matching new_tree_building.ipynb:

Tables:
  1. Benchmark Results Table
  2. Ablation Results Table
  3. Adaptation Results Table

Plots:
  1. Benchmark Performance Trade-offs (Complexity vs MAE, Speed vs wMAPE with size connections)
  2. Case A Trajectory: Best Case Grid (Ground Truth + All Available Trained Benchmark Models)
  3. Case B Trajectory: Worst Case Grid (Ground Truth + All Available Trained Benchmark Models)
  4. Ablation Bars (MAE and Training Time)
  5. Adaptation Best-Case Trajectory (Ground Truth + All Available Adaptation Models)
  6. Adaptation Worst-Case Trajectory (Ground Truth + All Available Adaptation Models)
  7. Adaptation Speed vs Accuracy Scatter (Lower-left is faster and more accurate)

Usage
-----
    python scripts/run_results.py               # use latest run
    python scripts/run_results.py --no-dtw      # skip DTW recomputation
    python scripts/run_results.py --dtw-grids 20
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Global plot style: Times New Roman, larger publication-quality font sizes
plt.rcParams.update({
    "font.family":        "Times New Roman",
    "font.size":          16,
    "axes.titlesize":     18,
    "axes.labelsize":     16,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "legend.fontsize":    13,
    "figure.dpi":         150,
    "axes.unicode_minus": False,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS
except ModuleNotFoundError:
    from _common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS

from hmst.utils import calculate_metrics

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
    "HMST-v2 (Base)":              ["hmstv2_base", "baseline_hmstv2_base"],
    "HMST-v2 (Large)":             ["hmstv2_large", "baseline_hmstv2_large"],
}

ABLATION_KEYS = {
    "Full":                   "ablation_full",
    "w/o RevIN":              "ablation_wo_revin",
    "w/o Time Embedding":     "ablation_wo_time_embedding",
    "w/o Spatial Position":   "ablation_wo_spatial_position",
    "w/o R-Tree Mask":        "ablation_wo_rtree_mask",
    "Random Mask":            "ablation_random_mask",
}

ADAPT_KEYS = {
    "HMST Zero-Shot":               ["adapt_zs_hmstv2", "adapt_zs_hmstv2_large"],
    "HMST Fine-Tune":               "adapt_ft_hmstv2",
    "Each-Grid LSTM Zero-Shot":     ["adapt_zs_each_grid_lstm", "adapt_zs_each_grid_lstm_large"],
    "Each-Grid LSTM Retrain":      "adapt_retrain_each_grid_lstm",
    "All-Grid LSTM Zero-Shot":      ["adapt_zs_all_grid_lstm", "adapt_zs_all_grid_lstm_large"],
    "All-Grid LSTM Retrain":        ["adapt_retrain_all_grid_lstm", "adapt_retrain_lstm"],
    "All-Grid DLinear Zero-Shot":   ["adapt_zs_dlinear", "adapt_zs_dlinear_large"],
    "All-Grid DLinear Retrain":     ["adapt_retrain_dlinear", "adapt_retrain_dlinear_large"],
    "All-Grid PatchTST Zero-Shot":  ["adapt_zs_patchtst", "adapt_zs_patchtst_large"],
    "All-Grid PatchTST Retrain":    ["adapt_retrain_patchtst", "adapt_retrain_patchtst_large"],
    "All-Grid Reformer Zero-Shot":  ["adapt_zs_reformer", "adapt_zs_reformer_large"],
    "All-Grid Reformer Retrain":    ["adapt_retrain_reformer", "adapt_retrain_reformer_large"],
}

FAMILY_STYLE = {
    "HMST":            {"color": "#c0392b", "marker": "*", "s": 220, "zorder": 5},
    "Reformer":        {"color": "#8e44ad", "marker": "s", "s": 140, "zorder": 4},
    "PatchTST":        {"color": "#27ae60", "marker": "D", "s": 140, "zorder": 4},
    "Each-Grid LSTM":  {"color": "#3498db", "marker": "^", "s": 140, "zorder": 3},
    "All-Grid LSTM":   {"color": "#2980b9", "marker": "<", "s": 140, "zorder": 3},
    "DLinear":         {"color": "#636e72", "marker": "o", "s": 140, "zorder": 3},
}

SIZE_ORDER = {"Small": 0, "Base": 1, "Large": 2, "": 0}

LABEL_OFFSETS = {
    "Each-Grid LSTM (Small)":  ( 10,  14),
    "Each-Grid LSTM (Base)":   ( 10,   0),
    "Each-Grid LSTM (Large)":  ( 10, -14),
    "All-Grid LSTM (Small)":   (-10,  14),
    "All-Grid LSTM (Base)":    (-10,   0),
    "All-Grid LSTM (Large)":   (-10, -14),
    "DLinear (Small)":         ( 10,  14),
    "DLinear (Base)":          (-10,   6),
    "DLinear (Large)":         (-10,  -6),
    "PatchTST (Base)":         ( 10,   8),
    "PatchTST (Large)":        (-10,  -8),
    "Reformer (Base)":         ( 10,   8),
    "Reformer (Large)":        (-10,  -8),
    "HMST-v2 (Base)":          ( 10,   8),
    "HMST-v2 (Large)":         ( 10,  -8),
}


def _family(name):
    if "HMST"          in name: return "HMST"
    if "Reformer"      in name: return "Reformer"
    if "PatchTST"      in name: return "PatchTST"
    if "Each-Grid"     in name: return "Each-Grid LSTM"
    if "All-Grid LSTM" in name: return "All-LSTM"
    return "DLinear"


def _size(name):
    for tok in ("Small", "Base", "Large"):
        if tok in name:
            return tok
    return ""


def _offset(name):
    for key, off in LABEL_OFFSETS.items():
        if key in name:
            return off
    return (10, 5)


def _ha(name):
    ox, _ = _offset(name)
    return "left" if ox >= 0 else "right"


def _display(name):
    return (name
            .replace("All-Grid ",  "All-Grid")
            .replace("Each-Grid ", "Each-Grid")
            .replace("HMST-v2 ",   "HMST-v2"))


# ─────────────────────────────────────────────────────────────────────────────
def load_all(run_dir, key_map, logger, dtw_grids=50):
    """Load available results and (re)compute metrics."""
    rows = {}
    for display, keys in key_map.items():
        if isinstance(keys, str):
            candidate_keys = [keys]
        else:
            candidate_keys = keys

        found_key = None
        for k in candidate_keys:
            if (run_dir / "results" / f"{k}.npz").exists():
                found_key = k
                break

        if not found_key:
            continue

        preds, trues, meta = load_result(run_dir, found_key)
        mae, rmse, wmape, ddtw, sdtw = calculate_metrics(trues, preds, dtw_grids=dtw_grids)
        params_k = meta.get("n_params_K") or KNOWN_PARAMS.get(display, float("nan"))
        rows[display] = {
            "MAE":        round(mae,   4),
            "RMSE":       round(rmse,  4),
            "wMAPE (%)":  round(wmape, 2),
            "Daily DTW":  round(ddtw,  4),
            "Step DTW":   round(sdtw,  4),
            "Params (K)": params_k,
            "Train (s)":  meta.get("train_time_s", float("nan")),
            "_preds":     preds,
            "_trues":     trues,
        }
        logger.info(f"  Loaded: {display}  MAE={mae:.4f}")
    return rows


def save_table(rows: dict, path: Path, title: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    display_cols = ["MAE", "RMSE", "wMAPE (%)", "Daily DTW", "Step DTW",
                    "Params (K)", "Train (s)"]
    df = pd.DataFrame({k: {c: v[c] for c in display_cols if c in v}
                       for k, v in rows.items()}).T
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write(df.to_markdown(floatfmt=".4f"))
        f.write("\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions (Times New Roman for ALL elements)
# ─────────────────────────────────────────────────────────────────────────────
def plot_benchmark_tradeoffs(df_bench, plots_dir, logger):
    """Complexity vs MAE and Speed vs wMAPE 2-subplot figure with connected size lines."""
    if df_bench.empty:
        return

    def plot_tradeoff(ax, x_metric, y_metric, x_label, y_label, title):
        families = {}
        for name, row in df_bench.iterrows():
            fam = _family(name)
            sz  = _size(name)
            x, y = float(row[x_metric]), float(row[y_metric])
            families.setdefault(fam, []).append((SIZE_ORDER[sz], x, y, name))

        for fam, pts in families.items():
            if len(pts) < 2:
                continue
            pts_sorted = sorted(pts, key=lambda t: t[0])
            xs = [p[1] for p in pts_sorted]
            ys = [p[2] for p in pts_sorted]
            sty = FAMILY_STYLE.get(fam, {"color": "gray", "zorder": 3})
            ax.plot(xs, ys, ls="--", lw=1.3, color=sty["color"],
                    alpha=0.55, zorder=sty["zorder"] - 1)

        for name, row in df_bench.iterrows():
            x, y = float(row[x_metric]), float(row[y_metric])
            fam  = _family(name)
            sty  = FAMILY_STYLE.get(fam, {"color": "gray", "s": 120, "marker": "o", "zorder": 3})
            ox, oy = _offset(name)

            ax.scatter(x, y,
                       c=sty["color"], s=sty["s"], marker=sty["marker"],
                       edgecolors="black", linewidths=0.8,
                       alpha=0.90, zorder=sty["zorder"])

            ax.annotate(
                _display(name),
                (x, y),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=11,
                fontfamily="Times New Roman",
                fontweight="normal",
                color="black",
                zorder=5,
                va="center",
                ha=_ha(name),
            )

        ax.set_title(title, fontsize=16, fontfamily="Times New Roman", color="black")
        ax.set_xlabel(x_label, fontsize=14, fontfamily="Times New Roman", color="black")
        ax.set_ylabel(y_label, fontsize=14, fontfamily="Times New Roman", color="black")
        ax.tick_params(colors="black", labelsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        ax.grid(True, alpha=0.25, color="gray")
        ax.set_facecolor("#FCFCFC")
        ax.margins(x=0.18, y=0.18)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.suptitle("Model Performance Trade-offs (Lower-Left is Better)",
                 fontsize=18, fontfamily="Times New Roman", color="black")

    legend_handles = [
        mpatches.Patch(color=v["color"], label=k)
        for k, v in FAMILY_STYLE.items()
        if k in [_family(n) for n in df_bench.index]
    ]
    legend_handles.append(
        plt.Line2D([0], [0], ls="--", color="gray", lw=1.3,
                   label="Size: Small -> Base -> Large")
    )

    plot_tradeoff(axes[0],
                  x_metric="Params (K)", y_metric="MAE",
                  x_label="Model Complexity (Params in K)",
                  y_label="MAE (Accuracy)",
                  title="Complexity vs. Accuracy (MAE)")
    axes[0].legend(handles=legend_handles, fontsize=10,
                   framealpha=0.85, edgecolor="black",
                   prop={"family": "Times New Roman"})

    plot_tradeoff(axes[1],
                  x_metric="Train (s)", y_metric="wMAPE (%)",
                  x_label="Training Efficiency (Seconds)",
                  y_label="wMAPE (%)",
                  title="Speed vs. Accuracy (wMAPE)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = plots_dir / "benchmark_tradeoffs.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


def filter_largest_per_family(preds_store: dict) -> dict:
    """Filter preds_store to include only the largest available size for each model family."""
    family_models = {}
    for name, data in preds_store.items():
        fam = _family(name)
        sz  = _size(name)
        val = SIZE_ORDER.get(sz, 0)
        if fam not in family_models or val > family_models[fam][0]:
            family_models[fam] = (val, name, data)
    return {name: data for _, name, data in family_models.values()}


def plot_benchmark_trajectories(bench_rows, plots_dir, logger):
    """Plot Case A (Best Grid) and Case B (Worst Grid) for the largest available model of each family."""
    if not bench_rows:
        return

    full_preds_store = {k: (v["_preds"], v["_trues"]) for k, v in bench_rows.items()}
    preds_store      = filter_largest_per_family(full_preds_store)

    first_name  = next(iter(preds_store))
    trues       = preds_store[first_name][1]
    T_total     = trues.shape[0]

    # Minimum grid count across loaded benchmark models
    N_min = min(t.shape[1] for p, t in preds_store.values())

    # Per-grid MAE for all loaded models (restricted to N_min)
    grid_maes = {k: np.mean(np.abs(p[:, :N_min] - t[:, :N_min]), axis=0) for k, (p, t) in preds_store.items()}

    # Select best & worst grids
    if any("HMST" in k for k in preds_store):
        hmst_k = next(k for k in preds_store if "HMST" in k)
        bl_maes = [m for k, m in grid_maes.items() if "HMST" not in k]
        if bl_maes:
            best_bl = np.min(np.array(bl_maes), axis=0)
            diff    = grid_maes[hmst_k] - best_bl
            case_a_grid = int(np.argmin(diff))
            case_b_grid = int(np.argmax(diff))
        else:
            case_a_grid = int(np.argmin(grid_maes[hmst_k]))
            case_b_grid = int(np.argmax(grid_maes[hmst_k]))
    else:
        avg_mae     = np.mean(np.array(list(grid_maes.values())), axis=0)
        case_a_grid = int(np.argmin(avg_mae))
        case_b_grid = int(np.argmax(avg_mae))

    T_SHOW = min(168, T_total)

    model_colors = {
        "HMST-v2 (Large)":        "#c0392b",
        "HMST-v2 (Base)":         "#e74c3c",
        "Each-Grid LSTM (Large)": "#3498db",
        "All-Grid LSTM (Large)":  "#2980b9",
        "All-Grid DLinear (Large)":"#e67e22",
        "All-Grid PatchTST (Large)":"#27ae60",
        "All-Grid Reformer (Large)":"#8e44ad",
        "All-Grid ConvLSTM (Large)":"#6c5ce7",
    }
    fallback_colors = ["#3498db", "#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#6c5ce7", "#1abc9c", "#f39c12"]

    for gid, label_title, filename in [
        (case_a_grid, "Case A -- Best Performing Grid",  "trajectory_case_a_best.png"),
        (case_b_grid, "Case B -- Worst Performing Grid", "trajectory_case_b_worst.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(trues[:T_SHOW, gid], color="black", lw=2.5, label="Ground Truth", alpha=0.9, zorder=2)

        for i, (name, (preds, _)) in enumerate(preds_store.items()):
            if gid >= preds.shape[1]:
                continue
            col = model_colors.get(name, fallback_colors[i % len(fallback_colors)])
            is_hmst = "HMST" in name
            lw  = 2.2 if is_hmst else 1.4
            ls  = "-" if is_hmst else "--"
            zo  = 5 if is_hmst else 3
            ax.plot(preds[:T_SHOW, gid], color=col, lw=lw, ls=ls, label=_display(name), alpha=0.85, zorder=zo)

        ax.set_title(f"Benchmark Forecast Trajectory: Grid {gid} ({label_title})",
                     fontsize=15, fontfamily="Times New Roman", fontweight="bold")
        ax.set_xlabel("Hours", fontsize=13, fontfamily="Times New Roman")
        ax.set_ylabel("Population Flow", fontsize=13, fontfamily="Times New Roman")
        ax.legend(loc="upper right", framealpha=0.85, prop={"family": "Times New Roman", "size": 11})
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: {filename}")


def plot_ablation_bars(abl_rows, plots_dir, logger):
    """Ablation MAE and Training Time bar chart."""
    if not abl_rows:
        return
    palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    names   = list(abl_rows.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric in zip(axes, ["MAE", "Train (s)"]):
        vals = [abl_rows[v][metric] for v in names]
        bars = ax.bar(names, vals, color=palette[:len(vals)], edgecolor='black', lw=0.8)
        ax.set_title(f"Ablation: {metric}", fontsize=15, fontfamily="Times New Roman", fontweight='bold')
        ax.set_ylabel(metric, fontsize=13, fontfamily="Times New Roman")
        ax.tick_params(colors='black', labelsize=12)
        ax.set_xticklabels(names, rotation=20, ha="right", fontfamily="Times New Roman")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontfamily="Times New Roman")
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "ablation_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


def plot_adaptation_trajectories(adapt_rows, plots_dir, logger):
    """Plot Adaptation Best Case and Worst Case grid trajectories for ANY trained adaptation models."""
    if not adapt_rows:
        return

    preds_store = {k: (v["_preds"], v["_trues"]) for k, v in adapt_rows.items()}
    first_name  = next(iter(preds_store))
    trues       = preds_store[first_name][1]
    T_total     = trues.shape[0]

    # Restrict to minimum common grid count
    N_min = min(t.shape[1] for p, t in preds_store.values())

    grid_maes = {k: np.mean(np.abs(p[:, :N_min] - t[:, :N_min]), axis=0) for k, (p, t) in preds_store.items()}

    if "HMST Fine-Tune" in grid_maes:
        ref_mae = grid_maes["HMST Fine-Tune"]
    else:
        ref_mae = np.mean(np.array(list(grid_maes.values())), axis=0)

    adapt_best_grid  = int(np.argmin(ref_mae))
    adapt_worst_grid = int(np.argmax(ref_mae))

    T_SHOW = min(168, T_total)

    color_map = {
        "HMST Zero-Shot":               "#c0392b",
        "HMST Fine-Tune":               "#e74c3c",
        "All-Grid LSTM Zero-Shot":      "#3498db",
        "All-Grid LSTM Retrain":        "#2980b9",
        "All-Grid DLinear Zero-Shot":   "#2ecc71",
        "All-Grid DLinear Retrain":     "#27ae60",
        "All-Grid PatchTST Zero-Shot":  "#9b59b6",
        "All-Grid PatchTST Retrain":    "#8e44ad",
        "All-Grid Reformer Zero-Shot":  "#f39c12",
        "All-Grid Reformer Retrain":    "#d35400",
        "All-Grid ConvLSTM Zero-Shot":  "#a29bfe",
        "All-Grid ConvLSTM Retrain":    "#6c5ce7",
    }
    fallback_colors = ["#e74c3c", "#3498db", "#2980b9", "#2ecc71", "#27ae60", "#8e44ad", "#f39c12"]

    for gid, label_tag, filename in [
        (adapt_best_grid,  "Best Case",  "adapt_trajectory_best.png"),
        (adapt_worst_grid, "Worst Case", "adapt_trajectory_worst.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(trues[:T_SHOW, gid], color="black", lw=2.5, label="Ground Truth (new pattern)", alpha=0.9, zorder=2)

        for i, (name, (preds, _)) in enumerate(preds_store.items()):
            if gid >= preds.shape[1]:
                continue
            col = color_map.get(name, fallback_colors[i % len(fallback_colors)])
            is_ft = "Fine-Tune" in name or "Retrain" in name
            lw    = 2.0 if "HMST" in name else 1.4
            ls    = "-" if is_ft else (":" if "Zero-Shot" in name else "-.")
            zo    = 5 if "HMST" in name else 3
            ax.plot(preds[:T_SHOW, gid], color=col, lw=lw, ls=ls, label=name, alpha=0.85, zorder=zo)

        ax.set_title(f"Adaptation Forecast Trajectory: Grid {gid} ({label_tag})",
                     fontsize=15, fontfamily="Times New Roman", fontweight="bold")
        ax.set_xlabel("Hours", fontsize=13, fontfamily="Times New Roman")
        ax.set_ylabel("Population Flow", fontsize=13, fontfamily="Times New Roman")
        ax.legend(loc="upper right", framealpha=0.85, prop={"family": "Times New Roman", "size": 10})
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: {filename}")


def plot_adaptation_scatter(adapt_rows, plots_dir, logger):
    """Adaptation Speed vs Accuracy Scatter Plot (matching notebook cell 31)."""
    if not adapt_rows:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    color_map = {
        "HMST Zero-Shot":                  "#e74c3c",
        "Each-Grid LSTM Zero-Shot":        "#95a5a6",
        "All-Grid LSTM Zero-Shot":         "#7f8c8d",
        "All-Grid ConvLSTM Zero-Shot":     "#bdc3c7",
        "All-Grid DLinear Zero-Shot":      "#34495e",
        "All-Grid PatchTST Zero-Shot":     "#2c3e50",
        "All-Grid Reformer Zero-Shot":     "#e67e22",
        "HMST Fine-Tune":                  "#c0392b",
        "All-Grid LSTM Retrain":          "#3498db",
        "All-Grid DLinear Retrain":       "#27ae60",
        "All-Grid PatchTST Retrain":      "#8e44ad",
        "All-Grid Reformer Retrain":      "#d35400",
        "All-Grid ConvLSTM Retrain":      "#6c5ce7",
    }

    for key, vals in adapt_rows.items():
        t_sec = vals.get("Train (s)", 0.0)
        if np.isnan(t_sec):
            t_sec = 0.0
        mae_val = vals["MAE"]
        col = color_map.get(key, "#95a5a6")

        ax.scatter(t_sec, mae_val, color=col, s=160, zorder=5, edgecolors="black", lw=0.8)
        ax.annotate(key.replace(" ", "\n"), (t_sec, mae_val),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=10, fontfamily="Times New Roman")

    ax.set_xlabel("Adaptation Time (seconds)", fontsize=13, fontfamily="Times New Roman")
    ax.set_ylabel("MAE on Changed Grids (B+C)", fontsize=13, fontfamily="Times New Roman")
    ax.set_title("Adaptation Speed vs. Accuracy (lower-left = faster and more accurate)",
                 fontsize=15, fontfamily="Times New Roman", fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "adapt_speed_accuracy_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
def main(dtw_grids=50):
    run_dir   = get_run_dir()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    logger    = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    logger.info(f"=== {SCRIPT}  run_dir={run_dir} ===")

    table_path = plots_dir / "results_tables.md"
    table_path.write_text(f"# Results -- run: {run_dir.name}\n", encoding="utf-8")

    # 1. Benchmark
    logger.info("\n--- Loading benchmark results ---")
    bench_rows = load_all(run_dir, BENCHMARK_KEYS, logger, dtw_grids)
    if bench_rows:
        df_bench = save_table(bench_rows, table_path, "Benchmark Results")
        print("\n" + "="*50 + " BENCHMARK RESULTS " + "="*50)
        print(df_bench.to_markdown(floatfmt=".4f"))
        plot_benchmark_tradeoffs(df_bench, plots_dir, logger)
        plot_benchmark_trajectories(bench_rows, plots_dir, logger)

    # 2. Ablation
    logger.info("\n--- Loading ablation results ---")
    abl_rows = load_all(run_dir, ABLATION_KEYS, logger, dtw_grids)
    if abl_rows:
        df_abl = save_table(abl_rows, table_path, "Ablation Study")
        print("\n" + "="*50 + " ABLATION RESULTS " + "="*50)
        print(df_abl.to_markdown(floatfmt=".4f"))
        plot_ablation_bars(abl_rows, plots_dir, logger)

    # 3. Adaptation
    logger.info("\n--- Loading adaptation results ---")
    adapt_rows = load_all(run_dir, ADAPT_KEYS, logger, dtw_grids)
    if adapt_rows:
        df_adapt = save_table(adapt_rows, table_path, "Adaptation Results")
        print("\n" + "="*50 + " ADAPTATION RESULTS " + "="*50)
        print(df_adapt.to_markdown(floatfmt=".4f"))
        plot_adaptation_trajectories(adapt_rows, plots_dir, logger)
        plot_adaptation_scatter(adapt_rows, plots_dir, logger)

    n_bench = len(bench_rows)
    n_abl   = len(abl_rows)
    n_adapt = len(adapt_rows)
    logger.info(
        f"\nSummary: {n_bench} benchmark | {n_abl} ablation | {n_adapt} adaptation results. "
        f"Tables -> {table_path}  Plots -> {plots_dir}"
    )
    print(f"\nDone. {n_bench} benchmark, {n_abl} ablation, {n_adapt} adaptation results.")
    print(f"Tables: {table_path}")
    print(f"Plots:  {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtw-grids", type=int, default=50,
                        help="Number of grids to sample for DTW computation (default 50).")
    parser.add_argument("--no-dtw", action="store_true",
                        help="Skip DTW; set dtw_grids=0.")
    args = parser.parse_args()
    dtw = 0 if args.no_dtw else args.dtw_grids
    main(dtw_grids=dtw)
