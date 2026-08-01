"""
rtreeformer.utils.masks
=======================
Hierarchical R-Tree attention mask builders for R-Treeformer.

A *forest mask* is a list of four (N, N) float32 numpy arrays:

    forest_masks[0] - Level 0: self-loop identity (diagonal = 0, rest = -1e9)
    forest_masks[1] - Level 1: fine-grained micro-clusters
    forest_masks[2] - Level 2: mid-level region groups
    forest_masks[3] - Level 3: root-level partitions

Usage::

    from rtreeformer.utils.masks import build_forest_masks
    rtree_masks = build_forest_masks(
        num_grids      = NUM_GRIDS,
        K_roots        = K_ROOTS,
        root_labels    = root_labels,
        forest_levels  = forest_levels,
    )
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
def _identity_mask(num_grids: int) -> np.ndarray:
    """(N, N) mask where only diagonal = 0.0, off-diagonal = -1e9."""
    m = np.full((num_grids, num_grids), -1e9, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _group_mask(
    num_grids: int,
    K_roots: int,
    root_labels: np.ndarray,
    forest_levels: list,
    levels_idx: int,
) -> np.ndarray:
    """
    Build an attention mask from hierarchical group assignments at a given
    R-Tree level.

    Parameters
    ----------
    num_grids     : total number of selected grids (N)
    K_roots       : number of root regions
    root_labels   : (N,) int array — which root region each grid belongs to
    forest_levels : list of per-region level lists (built by SubTreeBuilder)
    levels_idx    : which level of the hierarchy to use (1 = fine, 3 = coarse)
    """
    mask = np.full((num_grids, num_grids), -1e9, dtype=np.float32)
    for r in range(K_roots):
        region_sel = np.where(root_labels == r)[0]
        levels     = forest_levels[r]
        lvl_idx    = min(levels_idx, len(levels) - 1)
        groups     = levels[lvl_idx]
        for grp in groups:
            grp_sel = region_sel[grp]
            for u in grp_sel:
                for v in grp_sel:
                    mask[u, v] = 0.0
    return mask


def _root_partition_mask(
    num_grids: int,
    K_roots: int,
    root_labels: np.ndarray,
) -> np.ndarray:
    """Level-3 mask: all grids in the same root region attend to each other."""
    mask = np.full((num_grids, num_grids), -1e9, dtype=np.float32)
    for r in range(K_roots):
        grp_sel = np.where(root_labels == r)[0]
        for u in grp_sel:
            for v in grp_sel:
                mask[u, v] = 0.0
    return mask


# ---------------------------------------------------------------------------
def build_forest_masks(
    num_grids: int,
    K_roots: int,
    root_labels: np.ndarray,
    forest_levels: list,
) -> list[np.ndarray]:
    """
    Construct the full list of four hierarchical attention masks.

    Returns
    -------
    list of four (num_grids, num_grids) float32 arrays:
        [identity_mask, level1_mask, level2_mask, level3_mask]
    """
    return [
        _identity_mask(num_grids),
        _group_mask(num_grids, K_roots, root_labels, forest_levels, levels_idx=1),
        _group_mask(num_grids, K_roots, root_labels, forest_levels, levels_idx=2),
        _root_partition_mask(num_grids, K_roots, root_labels),
    ]
