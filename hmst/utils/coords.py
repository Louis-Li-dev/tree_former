from __future__ import annotations

import numpy as np


def infer_grid_shape_and_order(coords: np.ndarray, decimal: int = 6) -> tuple[tuple[int, int], np.ndarray]:
    """Infer a regular grid shape and row-major order from 2D coordinates.

    Returns:
        (grid_h, grid_w), order

    The returned order maps the original coordinate list index to the
    flattened row-major grid index.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    lon_vals = np.unique(np.round(coords[:, 0], decimal))
    lat_vals = np.unique(np.round(coords[:, 1], decimal))
    if lon_vals.size * lat_vals.size != coords.shape[0]:
        raise ValueError(
            f"Coordinates do not form a complete regular grid: "
            f"{coords.shape[0]} points, but {lat_vals.size} latitudes × "
            f"{lon_vals.size} longitudes != {coords.shape[0]}"
        )

    lon_vals.sort()
    lat_vals.sort()
    grid_h, grid_w = lat_vals.size, lon_vals.size

    lon_to_idx = {v: i for i, v in enumerate(lon_vals)}
    lat_to_idx = {v: i for i, v in enumerate(lat_vals)}

    order = np.empty(coords.shape[0], dtype=np.int64)
    rounded = np.round(coords, decimal)
    for original_idx, (lon, lat) in enumerate(rounded):
        lon_idx = lon_to_idx[lon]
        lat_idx = lat_to_idx[lat]
        order[original_idx] = lat_idx * grid_w + lon_idx

    return (grid_h, grid_w), order
