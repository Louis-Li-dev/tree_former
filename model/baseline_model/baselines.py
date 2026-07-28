"""
Baseline models for urban population flow forecasting.

Models:
  RevIN             - Reversible Instance Normalization (shared utility)
  EachGridLSTM      - Per-grid independent LSTM
  AllGridLSTM       - Shared LSTM across all grids (channel-independent)
  AllGridConvLSTM   - Spatial ConvLSTM on 20x25 grid
  AllGridDLinear    - DLinear (AAAI 2023, LTSF-Linear)
  AllGridPatchTST   - PatchTST (ICLR 2023, channel-independent)

All models:
  - Input:  (B, N, L)  raw population flow
  - Output: (B, N, 1)  next-step prediction
  - Use RevIN for fair distribution-shift handling
  - Accept (hour, dow) kwargs but ignore them (no time embedding = our contribution)
"""

import numpy as np
import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
class RevIN(nn.Module):
    """Per-instance, per-grid reversible normalization."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._mu    = None
        self._sigma = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, L) → normalised"""
        self._mu    = x.mean(dim=-1, keepdim=True).detach()
        self._sigma = x.std(dim=-1,  keepdim=True).clamp(min=self.eps).detach()
        return (x - self._mu) / self._sigma

    def denorm(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, N, *) → raw scale"""
        return y * self._sigma + self._mu


# ─────────────────────────────────────────────────────────────────────────────
class EachGridLSTM(nn.Module):
    """
    500 independent per-grid LSTMs sharing the same weight structure.
    (Vectorised via einsum — equivalent to N separate nn.LSTM(1, H).)
    """

    def __init__(self, num_grids: int, lookback: int = 12,
                 hidden_dim: int = 16, use_revin: bool = True):
        super().__init__()
        self.use_revin  = use_revin
        self.revin      = RevIN()
        self.hidden_dim = hidden_dim

        self.W_ih     = nn.Parameter(torch.Tensor(num_grids, 4 * hidden_dim, 1))
        self.W_hh     = nn.Parameter(torch.Tensor(num_grids, 4 * hidden_dim, hidden_dim))
        self.b_ih     = nn.Parameter(torch.zeros(num_grids, 4 * hidden_dim))
        self.b_hh     = nn.Parameter(torch.zeros(num_grids, 4 * hidden_dim))
        self.out_proj = nn.Parameter(torch.Tensor(num_grids, 1, hidden_dim))
        self.out_bias = nn.Parameter(torch.zeros(num_grids, 1))

        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.out_proj)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        if self.use_revin:
            x = self.revin.norm(x)
        B, N, L = x.shape
        h = torch.zeros(B, N, self.hidden_dim, device=x.device)
        c = torch.zeros(B, N, self.hidden_dim, device=x.device)
        for t in range(L):
            xt    = x[:, :, t].unsqueeze(-1)
            gates = (torch.einsum("bni,noi->bno", xt, self.W_ih) + self.b_ih
                   + torch.einsum("bnh,noh->bno",  h, self.W_hh) + self.b_hh)
            i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=-1)
            c = torch.sigmoid(f_g) * c + torch.sigmoid(i_g) * torch.tanh(g_g)
            h = torch.sigmoid(o_g) * torch.tanh(c)
        out = (torch.einsum("bnh,noh->bno", h, self.out_proj)
               + self.out_bias.unsqueeze(0))
        if self.use_revin:
            out = self.revin.denorm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridLSTM(nn.Module):
    """Single shared LSTM applied independently to every grid."""

    def __init__(self, hidden_dim: int = 32, use_revin: bool = True):
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN()
        self.lstm  = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        if self.use_revin:
            x = self.revin.norm(x)
        B, N, L  = x.shape
        out_flat, _ = self.lstm(x.view(B * N, L, 1))
        out = self.fc(out_flat[:, -1, :]).view(B, N, 1)
        if self.use_revin:
            out = self.revin.denorm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridConvLSTM(nn.Module):
    """ConvLSTM treating the N grids as a H×W spatial map (default 20×25)."""

    def __init__(self, hidden_dim: int = 16, grid_h: int = 20, grid_w: int = 25,
                 use_revin: bool = True):
        super().__init__()
        self.use_revin  = use_revin
        self.revin      = RevIN()
        self.hidden_dim = hidden_dim
        self.H, self.W  = grid_h, grid_w
        self.conv = nn.Conv2d(1 + hidden_dim, 4 * hidden_dim,
                              kernel_size=3, padding=1)
        self.fc   = nn.Linear(hidden_dim, 1)

    def _resolve_grid_shape(self, num_grids: int) -> tuple[int, int]:
        if self.H is None and self.W is None:
            side = int(math.sqrt(num_grids))
            if side * side != num_grids:
                raise ValueError(
                    f"AllGridConvLSTM requires a full spatial grid, but got "
                    f"{num_grids} grids. Provide grid_h/grid_w or use a "
                    f"non-spatial baseline like AllGridLSTM."
                )
            return side, side

        if self.H is None:
            if num_grids % self.W != 0:
                raise ValueError(
                    f"Cannot infer grid_h from {num_grids} grids and grid_w={self.W}."
                )
            return num_grids // self.W, self.W

        if self.W is None:
            if num_grids % self.H != 0:
                raise ValueError(
                    f"Cannot infer grid_w from {num_grids} grids and grid_h={self.H}."
                )
            return self.H, num_grids // self.H

        if self.H * self.W != num_grids:
            raise ValueError(
                f"Grid count mismatch: AllGridConvLSTM was configured for "
                f"{self.H}x{self.W}={self.H * self.W} grids, but input has "
                f"{num_grids}."
            )
        return self.H, self.W

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        if self.use_revin:
            x = self.revin.norm(x)
        B, N, L = x.shape
        H, W = self._resolve_grid_shape(N)
        xg = x.view(B, H, W, L)
        h  = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c  = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        for t in range(L):
            gates  = self.conv(torch.cat([xg[:, :, :, t].unsqueeze(1), h], dim=1))
            i_g, f_g, g_g, o_g = torch.chunk(gates, 4, dim=1)
            c = torch.sigmoid(f_g) * c + torch.sigmoid(i_g) * torch.tanh(g_g)
            h = torch.sigmoid(o_g) * torch.tanh(c)
        out = (self.fc(h.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim))
               .view(B, N, 1))
        if self.use_revin:
            out = self.revin.denorm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridDLinear(nn.Module):
    """
    DLinear — AAAI 2023 (LTSF-Linear).
    Decomposes input into trend (moving average) + seasonal (residual).
    Applies an independent linear layer to each component.
    Channel-independent: weights shared across grids.

    Reference: Zeng et al., "Are Transformers Effective for Time Series
    Forecasting?", AAAI 2023.
    """

    def __init__(self, lookback: int = 12, kernel_size: int = 3,
                 use_revin: bool = True):
        super().__init__()
        self.use_revin = use_revin
        self.revin     = RevIN()

        # Moving-average decomposition kernel (causal padding keeps length)
        pad = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=pad)

        # One linear per component; channel-independent (shared across N)
        self.linear_trend    = nn.Linear(lookback, 1)
        self.linear_seasonal = nn.Linear(lookback, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        """x: (B, N, L) → (B, N, 1)"""
        if self.use_revin:
            x = self.revin.norm(x)
        B, N, L = x.shape
        # Decompose (operate on the flattened B*N stream)
        x_flat   = x.view(B * N, 1, L)
        trend    = self.avg_pool(x_flat).view(B, N, L)    # (B, N, L)
        seasonal = x - trend
        # Predict
        out = (self.linear_trend(trend)
               + self.linear_seasonal(seasonal))           # (B, N, 1)
        if self.use_revin:
            out = self.revin.denorm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class AllGridPatchTST(nn.Module):
    """
    PatchTST — ICLR 2023.
    Channel-independent patch-based Transformer.
    Splits the input time series into non-overlapping patches,
    embeds each patch as a token, and applies standard Transformer encoder.

    For L=12, patch_len=4, stride=4 → 3 patches.

    Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers", ICLR 2023.
    """

    def __init__(self, lookback: int = 12, patch_len: int = 4, stride: int = 4,
                 d_model: int = 32, nhead: int = 4, num_layers: int = 2,
                 use_revin: bool = True):
        super().__init__()
        self.use_revin  = use_revin
        self.revin      = RevIN()
        self.patch_len  = patch_len
        self.stride     = stride

        # Number of patches (no padding — choose patch_len/stride to divide L evenly)
        self.n_patches = (lookback - patch_len) // stride + 1

        # Patch value embedding
        self.patch_emb = nn.Linear(patch_len, d_model)
        # Learnable positional embedding
        self.pos_emb   = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Flatten all patch representations → predict 1 step
        self.out_proj = nn.Linear(self.n_patches * d_model, 1)

    def forward(self, x: torch.Tensor, hour=None, dow=None) -> torch.Tensor:
        """x: (B, N, L) → (B, N, 1)"""
        if self.use_revin:
            x = self.revin.norm(x)
        B, N, L = x.shape

        # Extract patches: (B, N, n_patches, patch_len)
        patches = x.unfold(-1, self.patch_len, self.stride)

        # Process channel-independently: (B*N, n_patches, patch_len)
        p_flat = patches.reshape(B * N, self.n_patches, self.patch_len)

        # Embed + positional
        h = self.patch_emb(p_flat) + self.pos_emb          # (B*N, P, d_model)

        # Transformer
        h = self.transformer(h)                              # (B*N, P, d_model)

        # Predict: flatten patches → 1 value
        out = self.out_proj(h.reshape(B * N, -1)).view(B, N, 1)

        if self.use_revin:
            out = self.revin.denorm(out)
        return out
