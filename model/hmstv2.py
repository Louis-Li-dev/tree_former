"""
HMST-v2: Hierarchical Masked Spatial Transformer v2

Architecture:
  RevIN → iTransformer token embedding → Spatial Position Embedding
  → Time Embedding (hour + day-of-week)
  → Hierarchical R-Tree Masked Multi-Head Attention × num_layers
  → FFN → Output Head → RevIN⁻¹

Ablation flags on HMSTv2:
  use_revin      : RevIN on/off
  use_rtree_mask : hierarchical masks on/off (False = full attention)
  random_mask    : replace R-Tree masks with random-group masks of same sparsity
  active_levels  : list [1,2,3] of which R-Tree levels to activate
"""

import numpy as np
import torch
import torch.nn as nn

from model.baseline_model.baselines import RevIN   # shared RevIN


# ─────────────────────────────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    """Embeds hour-of-day + day-of-week into a d_out-dimensional vector."""

    def __init__(self, d_out: int = 32):
        super().__init__()
        self.hour_emb = nn.Embedding(24, 8)
        self.dow_emb  = nn.Embedding(7, 4)
        self.proj     = nn.Linear(12, d_out)

    def forward(self, hour: torch.Tensor,
                dow: torch.Tensor) -> torch.Tensor:
        """hour/dow: (B,) long → (B, d_out)"""
        return self.proj(
            torch.cat([self.hour_emb(hour), self.dow_emb(dow)], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalBlock(nn.Module):
    """
    One Transformer block with Hierarchical Masked Spatial Attention + FFN.
    num_levels independent Q/K/V branches, each with its own R-Tree level mask.
    Outputs are concatenated and fused back to d_model.
    """

    def __init__(self, d_model: int, d_k: int,
                 num_levels: int, masks: list, dropout: float = 0.1):
        super().__init__()
        self.num_levels = num_levels
        self.d_k        = d_k
        self.scale      = d_k ** -0.5

        self.W_q = nn.ModuleList(
            [nn.Linear(d_model, d_k, bias=False) for _ in range(num_levels)])
        self.W_k = nn.ModuleList(
            [nn.Linear(d_model, d_k, bias=False) for _ in range(num_levels)])
        self.W_v = nn.ModuleList(
            [nn.Linear(d_model, d_k, bias=False) for _ in range(num_levels)])

        for l, mask in enumerate(masks):
            if isinstance(mask, torch.Tensor):
                mask_t = mask.clone().detach().to(dtype=torch.float32)
            else:
                mask_t = torch.tensor(mask, dtype=torch.float32)
            self.register_buffer(f"mask_{l}", mask_t)

        self.fusion    = nn.Linear(num_levels * d_k, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, d_model) → (B, N, d_model)"""
        ctx = []
        for l in range(self.num_levels):
            Q = self.W_q[l](h)
            K = self.W_k[l](h)
            V = self.W_v[l](h)
            mask  = getattr(self, f"mask_{l}")
            score = (torch.matmul(Q, K.transpose(-2, -1)) * self.scale
                     + mask.unsqueeze(0))
            attn  = self.attn_drop(torch.softmax(score, dim=-1))
            ctx.append(torch.matmul(attn, V))

        h = self.attn_norm(h + self.fusion(torch.cat(ctx, dim=-1)))
        h = self.ffn_norm(h + self.ffn(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
def _random_mask_like(template_mask: np.ndarray,
                      num_grids: int) -> np.ndarray:
    """Build a random-group mask matching template_mask's sparsity pattern
    but with randomly permuted grid assignments."""
    perm       = np.random.permutation(num_grids)
    rand_mask  = np.full((num_grids, num_grids), -1e9, dtype=np.float32)
    for i in range(num_grids):
        for j in range(num_grids):
            if template_mask[i, j] == 0.0:
                rand_mask[perm[i], perm[j]] = 0.0
    return rand_mask


# ─────────────────────────────────────────────────────────────────────────────
class HMSTv2(nn.Module):
    """
    Hierarchical Masked Spatial Transformer v2 (proposed model).

    Parameters
    ----------
    num_grids    : int   – number of grids (N)
    lookback     : int   – input sequence length (L)
    d_model      : int   – token embedding dimension
    d_k          : int   – key/query dimension per attention head
    num_layers   : int   – number of Transformer blocks
    dropout      : float – dropout rate
    forest_masks : list  – [identity, L1, L2, L3] masks, each (N, N) float32
    use_revin    : bool  – enable RevIN (distribution-shift robustness)
    use_rtree_mask : bool – enable R-Tree hierarchical masks
    random_mask  : bool  – replace R-Tree masks with random-group masks
    active_levels : list – subset of [1,2,3] levels to activate
    """

    def __init__(
        self,
        num_grids: int   = 500,
        lookback:  int   = 12,
        d_model:   int   = 32,
        d_k:       int   = 16,
        num_layers: int  = 2,
        dropout:   float = 0.1,
        forest_masks: list | None = None,
        use_revin:      bool = True,
        use_rtree_mask: bool = True,
        random_mask:    bool = False,
        active_levels:  list | None = None,
        use_spatial_emb: bool = True,
        use_time_emb:    bool = True,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.revin     = RevIN()
        self.use_spatial_emb = use_spatial_emb
        self.use_time_emb    = use_time_emb

        # ── Token embedding (iTransformer style: L → d_model per grid) ──
        self.input_proj  = nn.Linear(lookback, d_model)
        # ── Spatial position embedding ──
        self.spatial_emb = nn.Embedding(num_grids, d_model)
        # ── Time context embedding ──
        self.time_emb    = TimeEmbedding(d_out=d_model)

        # ── Prepare hierarchical masks ──
        # forest_masks[0]=identity, [1]=L1 micro, [2]=L2 mid, [3]=L3 root
        lvl_srcs   = forest_masks[1:]       # use L1, L2, L3
        num_levels = len(lvl_srcs)
        if active_levels is None:
            active_levels = list(range(1, num_levels + 1))

        ident_mask = np.full((num_grids, num_grids), -1e9, dtype=np.float32)
        np.fill_diagonal(ident_mask, 0.0)

        masks = []
        for l_idx, src in enumerate(lvl_srcs):
            lnum = l_idx + 1
            if not use_rtree_mask:
                # Full attention (no spatial constraint)
                masks.append(np.zeros((num_grids, num_grids), dtype=np.float32))
            elif random_mask:
                # Random grouping with same sparsity
                masks.append(_random_mask_like(src, num_grids))
            elif lnum not in active_levels:
                # Inactive level → self-only (identity)
                masks.append(ident_mask.copy())
            else:
                masks.append(src)

        # ── Transformer blocks ──
        self.blocks   = nn.ModuleList([
            HierarchicalBlock(d_model, d_k, num_levels, masks, dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                hour: torch.Tensor,
                dow:  torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, L) – raw population flow
        hour : (B,) long – prediction hour (0-23)
        dow  : (B,) long – prediction day-of-week (0-6)
        Returns: (B, N, 1) raw scale predictions
        """
        if self.use_revin:
            x = self.revin.norm(x)

        B, N, L  = x.shape
        grid_ids = torch.arange(N, device=x.device)

        h = self.input_proj(x)                              # (B, N, d_model)
        if self.use_spatial_emb:
            h = h + self.spatial_emb(grid_ids).unsqueeze(0)    # + spatial identity
        if self.use_time_emb:
            h = h + self.time_emb(hour, dow).unsqueeze(1)      # + time context

        for block in self.blocks:
            h = block(h)

        out = self.out_proj(h)          # (B, N, 1) normalised
        if self.use_revin:
            out = self.revin.denorm(out)
        return out
