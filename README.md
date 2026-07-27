# HMST-v2: Hierarchical Masked Spatial Transformer v2

> Urban population-flow forecasting with R-Tree hierarchical masked spatial attention.

---

## Installation

```bash
pip install git+https://github.com/<your-org>/hmst.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/<your-org>/hmst.git
cd hmst
pip install -e ".[dev]"
```

---

## Package Structure

```
hmst/
├── model/
│   ├── hmstv2.py       ← HMSTv2 (proposed), TimeEmbedding, HierarchicalBlock
│   └── baselines.py    ← RevIN, EachGridLSTM, AllGridLSTM, AllGridConvLSTM,
│                          AllGridDLinear, AllGridPatchTST
├── train/
│   ├── dataset.py      ← CHTDataset, make_loaders
│   └── trainer.py      ← run_training (early stopping, gradient clipping)
└── utils/
    ├── config.py        ← MODEL_SIZES, TRAIN_CFG (single source of truth)
    ├── metrics.py       ← calculate_metrics (MAE, RMSE, wMAPE, DTW)
    └── masks.py         ← build_forest_masks (R-Tree attention masks)
```

---

## Quick Start

```python
import torch
from hmst.model  import HMSTv2, AllGridLSTM
from hmst.train  import make_loaders, run_training
from hmst.utils  import MODEL_SIZES, TRAIN_CFG, calculate_metrics, build_forest_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build data loaders
train_l, val_l, test_l = make_loaders(data, lookback=12, batch_size=32)

# 2. Instantiate a model using a named size
size  = MODEL_SIZES["large"]
model = HMSTv2(
    num_grids   = 500,
    lookback    = 12,
    d_model     = size["d_model"],
    d_k         = size["d_k"],
    num_layers  = size["num_layers"],
    forest_masks = forest_masks,
)

# 3. Train
model, preds, trues, pe, te, t_secs, n_params = run_training(
    "HMST-v2 (Large)", model, TRAIN_CFG, train_l, val_l, test_l, device
)

# 4. Evaluate
mae, rmse, wmape, daily_dtw, step_dtw = calculate_metrics(te, pe)
print(f"MAE={mae:.4f}  wMAPE={wmape:.2f}%  Params={n_params/1e3:.1f}K")
```

---

## Model Sizes

All model sizes are defined in `hmst.utils.config.MODEL_SIZES`:

| Size   | `hidden_dim` / `d_model` | `d_k` | `num_layers` | `nhead` |
|--------|--------------------------|-------|--------------|---------|
| small  | 16                       | 8     | 1            | 2       |
| base   | 32                       | 16    | 2            | 4       |
| large  | 64                       | 32    | 4            | 8       |

> **Rule**: the same size key applies uniformly across model types — LSTM, ConvLSTM, PatchTST, and HMST all use the same `hidden_dim`/`d_model` for a given size.

---

## Training Config

All models share `TRAIN_CFG` (no per-model overrides):

```python
TRAIN_CFG = {"lr": 1e-3, "max_epochs": 100, "patience": 10, "clip": 1.0}
```

---

## Experiments

The main research notebook (`new_tree_building.ipynb`) is organised as:

| Section | Description |
|---------|-------------|
| §1–3    | Data loading and R-Tree construction |
| §4      | Benchmark (Small / Base / Large × all models) |
| §5      | Ablation study (Large config) |
| §6      | Region-swap adaptation experiment (Large config) |

See `example.ipynb` for a minimal runnable walkthrough.

---

## Citation

```bibtex
@article{hmst2025,
  title  = {Hierarchical Masked Spatial Transformer for Urban Population Flow Forecasting},
  year   = {2025},
}
```
