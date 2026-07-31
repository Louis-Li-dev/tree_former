"""
hmst.model.baselines
====================
Baseline models for the HMST-v2 paper.

None of the baseline models apply internal RevIN — they accept standardized or
raw inputs as provided by the caller/trainer, and output predictions on the same
scale. Dataset-level Z-score standardization (train set mean/std) and inverse
transformation are handled by the trainer/runner.

Only the proposed HMST-v2 model retains internal RevIN normalization.

Architecture overview
---------------------
EachGridLSTM    : N independent LSTMs, vectorised via einsum
AllGridLSTM     : shared LSTM treating all grids as batch elements
AllGridConvLSTM : spatial ConvLSTM with auto coordinate-detection
AllGridDLinear  : DLinear (AAAI 2023) with TSL series_decomp
AllGridPatchTST : PatchTST (ICLR 2023) with TSL PatchEmbedding
AllGridReformer : Reformer (ICLR 2020) encoder-only, O(L log L) LSH attention
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TSL layers (vendored into hmst/layers/)
from hmst.layers.Autoformer_EncDec import series_decomp
from hmst.layers.Embed import DataEmbedding, PatchEmbedding
from hmst.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    ReformerLayer,
)
from hmst.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)


# ─────────────────────────────────────────────────────────────────────────────
class EachGridLSTM(nn.Module):
    """
    N 個獨立的 LSTM 共用權重 (透過 einsum 向量化加速)。
    """

    def __init__(self, num_grids: int, lookback: int = 12, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_ih = nn.Parameter(torch.Tensor(num_grids, 4 * hidden_dim, 1))
        self.W_hh = nn.Parameter(torch.Tensor(num_grids, 4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.zeros(num_grids, 4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.zeros(num_grids, 4 * hidden_dim))
        self.out_proj = nn.Parameter(torch.Tensor(num_grids, 1, hidden_dim))
        self.out_bias = nn.Parameter(torch.zeros(num_grids, 1))

        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.out_proj)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape
        h = torch.zeros(B, N, self.hidden_dim, device=x.device)
        c = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for t in range(L):
            xt = x[:, :, t].unsqueeze(-1)
            gates = (
                torch.einsum("bni,noi->bno", xt, self.W_ih)
                + self.b_ih
                + torch.einsum("bnh,noh->bno", h, self.W_hh)
                + self.b_hh
            )
            i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=-1)
            c = torch.sigmoid(f_g) * c + torch.sigmoid(i_g) * torch.tanh(g_g)
            h = torch.sigmoid(o_g) * torch.tanh(c)
        out = torch.einsum("bnh,noh->bno", h, self.out_proj) + self.out_bias.unsqueeze(0)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridLSTM(nn.Module):
    """單一 LSTM 權重，將所有網格視為獨立 Batch 進行預測。"""

    def __init__(self, hidden_dim: int = 32, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape
        out_flat, _ = self.lstm(x.view(B * N, L, 1))
        out = self.fc(out_flat[:, -1, :]).view(B, N, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
def _coord_to_grid_indices(vals: np.ndarray) -> np.ndarray:
    """
    將浮點座標轉換為從 0 開始的整數格位索引。

    原理
    ----
    找出所有唯一座標值之間的最小非零間距 (min_step)，以其倒數
    作為 scale factor，使得 min_step * scale ≈ 1（即相鄰格點
    距離恰好 1 格）。再減去最小值歸零，得到緊湊的整數索引。

    範例
    ----
    vals = [121.23, 121.24, 121.26]
      unique diffs = [0.01, 0.02] → min_step = 0.01 → scale = 100
      scaled = [12123, 12124, 12126]
      norm   = [0, 1, 3]          → W = 4 (range + 1)
    """
    unique_sorted = np.unique(vals)
    if len(unique_sorted) < 2:
        # 只有一個唯一值，所有格點疊在同一列/行
        return np.zeros(len(vals), dtype=int)

    diffs = np.diff(unique_sorted)
    positive_diffs = diffs[diffs > 1e-9]
    if len(positive_diffs) == 0 or positive_diffs.min() >= 1 - 1e-9:
        # 座標本身已是整數間距（或間距 ≥ 1），直接 round
        scale = 1.0
    else:
        min_step = positive_diffs.min()
        scale = round(1.0 / min_step)

    scaled = np.round(vals * scale).astype(int)
    return (scaled - scaled.min()).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
class AllGridConvLSTM(nn.Module):
    """
    Spatial ConvLSTM: auto-detects coordinate range and maps N grid sequences
    to an H x W spatial canvas.

    Coordinate normalisation:
        x_norm = x - x.min()   (range 0 ... x.max()-x.min())
        y_norm = y - y.min()   (range 0 ... y.max()-y.min())
        W = x.max() - x.min() + 1
        H = y.max() - y.min() + 1

    Valid-mask channel:
        A static 0/1 mask channel is concatenated to the Conv2d input at every
        time step to indicate which H x W cells contain real data (1) versus
        empty padding (0).  This prevents Z-score-normalised empty cells
        (value 0 ~= mean flow) from polluting adjacent cells via convolution.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        coords: np.ndarray | None = None,
        grid_h: int | None = None,
        grid_w: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if coords is not None:
            # coords: (N, 2) — [x_coord, y_coord]
            x = coords[:, 0]
            y = coords[:, 1]

            # 自動偵測 scale，讓浮點座標正確對齊到整數格位
            # （例如 0.01° 精度的經緯度 → scale=100，避免 round 後撞格）
            x_norm = _coord_to_grid_indices(x)
            y_norm = _coord_to_grid_indices(y)

            self.W = int(x_norm.max() + 1)   # width  = x range + 1
            self.H = int(y_norm.max() + 1)   # height = y range + 1

            # 1-D 展平索引：row-major (y * W + x)
            valid_indices = y_norm * self.W + x_norm
            self.register_buffer(
                "valid_indices",
                torch.from_numpy(valid_indices).long(),
                persistent=False,
            )

            # valid_mask：(1, 1, H, W)，有資料的格子為 1，空白為 0
            mask_flat = torch.zeros(self.H * self.W, dtype=torch.float32)
            mask_flat[torch.from_numpy(valid_indices).long()] = 1.0
            self.register_buffer(
                "valid_mask",
                mask_flat.view(1, 1, self.H, self.W),
                persistent=False,
            )
        else:
            # Fallback：呼叫端明確傳入 grid_h / grid_w
            assert grid_h is not None and grid_w is not None, (
                "AllGridConvLSTM requires either `coords` or both `grid_h` and `grid_w`."
            )
            self.H = grid_h
            self.W = grid_w
            # Fallback 時所有格子都有效
            self.register_buffer(
                "valid_mask",
                torch.ones(1, 1, self.H, self.W, dtype=torch.float32),
                persistent=False,
            )

        # Input channels: data (1) + valid_mask (1) + hidden state (hidden_dim)
        self.conv = nn.Conv2d(2 + hidden_dim, 4 * hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape

        # ── Scatter：N 個網格 → 完整 H × W 空間畫布 ─────────────────────────
        if hasattr(self, "valid_indices"):
            xg_dense = torch.zeros(B, self.H * self.W, L, device=x.device)
            idx = self.valid_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, L)
            xg_dense.scatter_(1, idx, x)
            xg = xg_dense.view(B, self.H, self.W, L)
        else:
            xg = x.view(B, self.H, self.W, L)

        # valid_mask expanded to batch: (B, 1, H, W), shared across all time steps
        mask = self.valid_mask.expand(B, -1, -1, -1)

        h = torch.zeros(B, self.hidden_dim, self.H, self.W, device=x.device)
        c = torch.zeros(B, self.hidden_dim, self.H, self.W, device=x.device)

        # -- ConvLSTM time loop -----------------------------------------------
        for t in range(L):
            # Input = [data_t, valid_mask, h_prev] so the conv kernel knows
            # which cells contain real data vs empty padding.
            gates = self.conv(torch.cat([xg[:, :, :, t].unsqueeze(1), mask, h], dim=1))
            i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=1)
            c = torch.sigmoid(f_g) * c + torch.sigmoid(i_g) * torch.tanh(g_g)
            h = torch.sigmoid(o_g) * torch.tanh(c)
            # Zero out empty cells to prevent phantom states from leaking into
            # adjacent valid cells across time steps.
            # mask: (B, 1, H, W) broadcasts to (B, hidden_dim, H, W)
            h = h * mask
            c = c * mask

        # ── Gather：H × W → 抽回原本的 N 個網格 ─────────────────────────────
        h_flat = h.view(B, self.hidden_dim, self.H * self.W)
        if hasattr(self, "valid_indices"):
            idx = self.valid_indices.unsqueeze(0).unsqueeze(1).expand(B, self.hidden_dim, N)
            h_valid = torch.gather(h_flat, 2, idx)  # (B, hidden_dim, N)
        else:
            h_valid = h_flat

        out = self.fc(h_valid.transpose(1, 2))  # (B, N, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridDLinear(nn.Module):
    """
    DLinear — AAAI 2023 (LTSF-Linear).
    通道獨立 (Channel-Independent) + TSL series_decomp.
    """

    def __init__(self, lookback: int = 12, kernel_size: int = 3, d_model: int = 32, **kwargs):
        super().__init__()
        self.decomp = series_decomp(kernel_size)
        # Linear heads：lookback → 1（1-step 預測）
        self.linear_trend = nn.Linear(lookback, 1)
        self.linear_seasonal = nn.Linear(lookback, 1)

        # 均勻初始化（對齊 TSL 原始實作）
        nn.init.constant_(self.linear_trend.weight, 1.0 / lookback)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / lookback)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape

        # (B, N, L) → (B*N, L, 1)
        x_flat = x.view(B * N, L, 1)
        seasonal, trend = self.decomp(x_flat)          # both (B*N, L, 1)
        seasonal = seasonal.squeeze(-1)                 # (B*N, L)
        trend = trend.squeeze(-1)                       # (B*N, L)

        out = self.linear_trend(trend) + self.linear_seasonal(seasonal)  # (B*N, 1)
        out = out.view(B, N, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridPatchTST(nn.Module):
    """
    PatchTST — ICLR 2023.
    使用 TSL PatchEmbedding + FullAttention Transformer Encoder.
    """

    def __init__(
        self,
        lookback: int = 12,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        padding = stride  # TSL 的 padding = stride

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        # n_patches 計算（含 padding）
        n_patches = int((lookback - patch_len) / stride + 2)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout),
                        d_model,
                        nhead,
                    ),
                    d_model,
                    d_ff=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ],
            norm_layer=nn.Sequential(
                _Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                _Transpose(1, 2),
            ),
        )

        # Flatten head
        head_nf = d_model * n_patches
        self.head_dropout = nn.Dropout(dropout)
        self.head_linear = nn.Linear(head_nf, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape

        x_perm = x.reshape(B * N, 1, L)  # treat each grid as a separate "variate"

        enc_out, n_vars = self.patch_embedding(x_perm)  # (B*N, n_patches, d_model)
        enc_out, _ = self.encoder(enc_out)               # (B*N, n_patches, d_model)

        # Reshape back: (B, N, d_model, n_patches)
        enc_out = enc_out.reshape(B, N, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)  # (B, N, d_model, n_patches)

        # Flatten head
        h = self.head_dropout(enc_out.reshape(B * N, -1))
        out = self.head_linear(h).view(B, N, 1)
        return out


class _Transpose(nn.Module):
    """Helper for BatchNorm inside Sequential."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


# ─────────────────────────────────────────────────────────────────────────────
class AllGridReformer(nn.Module):
    """
    Reformer — ICLR 2020 (Kitaev et al.).
    Channel-independent wrapper: 每個 grid 單獨跟Reformer encoder。

    Uses LSH self-attention with O(L log L) complexity instead of the O(L²)
    full attention in standard Transformers — substantially faster than
    Informer (which also needed a decoder), especially at the Large tier.
    """

    def __init__(
        self,
        lookback: int = 12,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        bucket_size: int = 4,
        n_hashes: int = 4,
        **kwargs,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # Channel-independent: c_in = 1
        self.enc_embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(
                        None,           # attention arg unused
                        d_model,
                        nhead,
                        bucket_size=bucket_size,
                        n_hashes=n_hashes,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Encoder-only (like PatchTST): project last token to prediction
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        B, N, L = x.shape

        # (B, N, L) → (B*N, L, 1)
        x_flat = x.reshape(B * N, L, 1)

        # Encoder
        enc_out = self.enc_embedding(x_flat)         # (B*N, L, d_model)
        enc_out, _ = self.encoder(enc_out)            # (B*N, L, d_model)

        # Take last time-step and project to 1-step prediction
        out = self.projection(enc_out[:, -1, :])     # (B*N, 1)
        out = out.view(B, N, 1)
        return out


# Backwards-compatibility alias
AllGridInformer = AllGridReformer