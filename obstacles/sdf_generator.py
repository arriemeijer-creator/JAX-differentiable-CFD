"""
Convert building polygons to a signed distance field (SDF) on a structured grid.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from typing import List


def polygons_to_sdf(polygons: List[np.ndarray], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Convert a list of polygons to a signed distance field on a structured grid.
    
    Args:
        polygons: List of (N,2) numpy arrays, each representing a polygon
        X: 2D meshgrid array of x coordinates
        Y: 2D meshgrid array of y coordinates
    
    Returns:
        sdf: 2D array where sdf > 0 in fluid, sdf < 0 in solid
    """
    nx, ny = X.shape
    
    # Flatten grid coordinates for point-in-polygon tests
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Rasterize polygons into binary mask (1 inside buildings, 0 outside)
    mask = np.zeros(nx * ny, dtype=np.float32)
    
    for poly in polygons:
        # Create matplotlib Path for vectorized point-in-polygon
        path = Path(poly)
        # Test all points at once (vectorized)
        inside = path.contains_points(points)
        mask = np.maximum(mask, inside.astype(np.float32))
    
    mask = mask.reshape(nx, ny)
    
    # Compute distance fields
    # distance_edt computes distance to nearest True pixel
    # For outside distance: invert mask (fluid becomes True)
    distance_outside = distance_transform_edt(1.0 - mask)
    # For inside distance: use mask directly (solid is True)
    distance_inside = distance_transform_edt(mask)
    
    # Construct SDF: positive in fluid, negative in solid
    sdf = distance_outside - distance_inside
    
    return sdf
