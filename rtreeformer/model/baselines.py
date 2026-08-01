"""
rtreeformer.model.baselines
===========================
Baseline models for the R-Treeformer paper.

None of the baseline models apply internal RevIN - they accept standardized or
raw inputs as provided by the caller/trainer, and output predictions on the same
scale. Dataset-level Z-score standardization (train set mean/std) and inverse
transformation are handled by the trainer/runner.

Only the proposed R-Treeformer model retains internal RevIN normalization.

Architecture overview
---------------------
EachGridLSTM    : N independent LSTMs, vectorised via einsum
AllGridLSTM     : shared LSTM treating all grids as batch elements
AllGridDLinear  : DLinear (AAAI 2023) with TSL series_decomp
AllGridPatchTST : PatchTST (ICLR 2023) with TSL PatchEmbedding
AllGridReformer : Reformer (ICLR 2020) encoder-only, O(L log L) LSH attention
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TSL layers (vendored into rtreeformer/layers/)
from rtreeformer.layers.Autoformer_EncDec import series_decomp
from rtreeformer.layers.Embed import DataEmbedding, PatchEmbedding
from rtreeformer.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    ReformerLayer,
)
from rtreeformer.layers.Transformer_EncDec import (
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