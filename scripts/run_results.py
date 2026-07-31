"""
scripts/run_results.py
======================
Auto-detects all available model results in runs/latest/results/ and generates:
  1. Benchmark results table  (MAE, RMSE, wMAPE, Daily DTW, Step DTW, Params, Time)
  2. Ablation results table
  3. Adaptation results table
  4. Benchmark scatter plot   (Params vs MAE, colour-coded by family)
  5. Ablation bar chart       (MAE, Train time)
  6. Adaptation scatter plot  (Time vs MAE)
  7. Trajectory plots         (best/worst HMST grid, best/worst adaptation grid)

Plots are saved to runs/latest/plots/.
Tables are printed to stdout and saved as runs/latest/plots/results_tables.md.

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
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts._common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS
except ModuleNotFoundError:
    from _common import get_run_dir, setup_logger, load_result, KNOWN_PARAMS

from hmst.utils import calculate_metrics

SCRIPT = "results"

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_COLOR = {
    "ConvLSTM":  "#6c5ce7",
    "Each-LSTM": "#3498db",
    "All-LSTM":  "#2980b9",
    "DLinear":   "#e67e22",
    "PatchTST":  "#27ae60",
    "Reformer":  "#8e44ad",
    "HMST-v2":   "#e74c3c",
    "Ablation":  "#2ecc71",
    "Adapt":     "#f39c12",
}
FAMILY_MARKER = {k: "o" for k in FAMILY_COLOR}
FAMILY_MARKER["HMST-v2"] = "*"

SIZE_LS = {"small": ":", "base": "--", "large": "-"}

# Key patterns for auto-detection (supports single string or candidate list)
BENCHMARK_KEYS = {
    "All-Grid ConvLSTM (Small)":   "baseline_convlstm_small",
    "All-Grid ConvLSTM (Base)":    "baseline_convlstm_base",
    "All-Grid ConvLSTM (Large)":   "baseline_convlstm_large",
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
    "All-Grid ConvLSTM Zero-Shot":  ["adapt_zs_convlstm", "adapt_zs_convlstm_large"],
    "All-Grid ConvLSTM Retrain":   "adapt_retrain_convlstm",
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


def family_of(name: str) -> str:
    if "ConvLSTM"  in name: return "ConvLSTM"
    if "Each-Grid" in name: return "Each-LSTM"
    if "All-Grid LSTM" in name: return "All-LSTM"
    if "DLinear"   in name: return "DLinear"
    if "PatchTST"  in name: return "PatchTST"
    if "Reformer"  in name: return "Reformer"
    if "HMST"      in name: return "HMST-v2"
    return "Other"


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
        # Recompute from arrays to be consistent
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


# ─────────────────────────────────────────────────────────────────────────────
def save_table(rows: dict, path: Path, title: str) -> pd.DataFrame:
    """Save a metrics table as markdown and return the DataFrame."""
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
def plot_benchmark_scatter(bench_rows, plots_dir, logger):
    if not bench_rows:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    used_families = set()
    for name, row in bench_rows.items():
        fam = family_of(name)
        col = FAMILY_COLOR.get(fam, "#95a5a6")
        mkr = FAMILY_MARKER.get(fam, "o")
        sz  = 180 if fam == "HMST-v2" else 100
        ax.scatter(row["Params (K)"], row["MAE"], c=col, marker=mkr,
                   s=sz, zorder=3, alpha=0.85)
        used_families.add(fam)

    # Legend by family
    for fam in used_families:
        ax.scatter([], [], c=FAMILY_COLOR.get(fam, "#95a5a6"),
                   marker=FAMILY_MARKER.get(fam, "o"),
                   s=80, label=fam)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (K)", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("Benchmark: Parameters vs MAE (all sizes)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    plt.tight_layout()
    out = plots_dir / "benchmark_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


def plot_ablation_bars(abl_rows, plots_dir, logger):
    if not abl_rows:
        return
    palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    names   = list(abl_rows.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["MAE", "Train (s)"]):
        vals  = [abl_rows[n].get(metric, 0) for n in names]
        bars  = ax.bar(names, vals, color=palette[:len(names)], edgecolor="black", lw=0.8)
        ax.set_title(f"Ablation: {metric}", fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "ablation_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


def plot_adaptation_scatter(adapt_rows, plots_dir, logger):
    if not adapt_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, row in adapt_rows.items():
        is_zs  = "Zero-Shot" in name or "Zero" in name
        is_ft  = "Fine-Tune" in name
        col    = "#e74c3c" if "HMST" in name else "#95a5a6"
        marker = "^" if is_ft else ("o" if is_zs else "s")
        ax.scatter(row.get("Train (s)", 0), row["MAE"],
                   c=col, marker=marker, s=120, zorder=3, alpha=0.85)
        ax.annotate(name, (row.get("Train (s)", 0), row["MAE"]),
                    fontsize=7.5, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Adaptation Time (s)", fontsize=12)
    ax.set_ylabel("MAE on B+C grids", fontsize=12)
    ax.set_title("Adaptation Speed vs Accuracy", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = plots_dir / "adaptation_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


def plot_trajectory(name, preds, trues, grid_idx, T_show, plots_dir, logger, suffix=""):
    """Plot a single 168-hour prediction vs ground truth for one grid."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(trues[:T_show, grid_idx], color="black", lw=2.5, label="Ground Truth")
    ax.plot(preds[:T_show, grid_idx], color="#e74c3c", lw=2, ls="-", label=name)
    ax.set_title(f"Trajectory — {name} | Grid {grid_idx}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("Population")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("/", "")[:30]
    out  = plots_dir / f"trajectory_{safe}{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
def main(dtw_grids=50):
    run_dir   = get_run_dir()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    logger    = setup_logger(SCRIPT, run_dir / "logs" / f"{SCRIPT}.log")
    logger.info(f"=== {SCRIPT}  run_dir={run_dir} ===")

    table_path = plots_dir / "results_tables.md"
    table_path.write_text(f"# Results — run: {run_dir.name}\n", encoding="utf-8")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    logger.info("\n--- Loading benchmark results ---")
    bench_rows = load_all(run_dir, BENCHMARK_KEYS, logger, dtw_grids)
    if bench_rows:
        df_bench = save_table(bench_rows, table_path, "Benchmark Results")
        print("\n" + "="*50 + " BENCHMARK RESULTS " + "="*50)
        print(df_bench.to_markdown(floatfmt=".4f"))
        plot_benchmark_scatter(bench_rows, plots_dir, logger)

        # Trajectory for HMST-v2 Large best/worst grid
        if "HMST-v2 (Large)" in bench_rows:
            row   = bench_rows["HMST-v2 (Large)"]
            p, t  = row["_preds"], row["_trues"]
            per_g = np.mean(np.abs(p - t), axis=0)
            best  = int(np.argmin(per_g))
            worst = int(np.argmax(per_g))
            plot_trajectory("HMST-v2 (Large)", p, t, best,  168, plots_dir, logger, "_best")
            plot_trajectory("HMST-v2 (Large)", p, t, worst, 168, plots_dir, logger, "_worst")

    # ── Ablation ──────────────────────────────────────────────────────────────
    logger.info("\n--- Loading ablation results ---")
    abl_rows = load_all(run_dir, ABLATION_KEYS, logger, dtw_grids)
    if abl_rows:
        df_abl = save_table(abl_rows, table_path, "Ablation Study")
        print("\n" + "="*50 + " ABLATION RESULTS " + "="*50)
        print(df_abl.to_markdown(floatfmt=".4f"))
        plot_ablation_bars(abl_rows, plots_dir, logger)

    # ── Adaptation ────────────────────────────────────────────────────────────
    logger.info("\n--- Loading adaptation results ---")
    adapt_rows = load_all(run_dir, ADAPT_KEYS, logger, dtw_grids)
    if adapt_rows:
        df_adapt = save_table(adapt_rows, table_path, "Adaptation Results")
        print("\n" + "="*50 + " ADAPTATION RESULTS " + "="*50)
        print(df_adapt.to_markdown(floatfmt=".4f"))
        plot_adaptation_scatter(adapt_rows, plots_dir, logger)

        # HMST fine-tune trajectory
        if "HMST Fine-Tune" in adapt_rows:
            row   = adapt_rows["HMST Fine-Tune"]
            p, t  = row["_preds"], row["_trues"]
            per_g = np.mean(np.abs(p - t), axis=0)
            best  = int(np.argmin(per_g))
            worst = int(np.argmax(per_g))
            plot_trajectory("HMST Fine-Tune", p, t, best,  168, plots_dir, logger, "_best")
            plot_trajectory("HMST Fine-Tune", p, t, worst, 168, plots_dir, logger, "_worst")

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
