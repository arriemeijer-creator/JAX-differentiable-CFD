"""
Load building footprints from GeoJSON (Overture Maps format).
Extracts polygon geometries and converts to local Cartesian coordinates.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Tuple


def load_building_polygons(filepath: str, bbox: Tuple[float, float, float, float]) -> List[np.ndarray]:
    """
    Load building footprints from a GeoJSON file and convert to local coordinates.
    
    Args:
        filepath: Path to GeoJSON file (Overture Maps format)
        bbox: Bounding box as (xmin, xmax, ymin, ymax) in meters (local coordinates)
    
    Returns:
        List of polygons, where each polygon is an (N,2) numpy array of (x,y) coordinates
    """
    # Validate file exists and has content
    import os
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    print(f"Loading GeoJSON file: {filepath} (size: {file_size} bytes)")
    
    # Load GeoJSON with fallback for malformed files
    try:
        gdf = gpd.read_file(filepath, engine='pyogrio')
    except Exception as e:
        print(f"pyogrio failed, trying fiona: {e}")
        try:
            gdf = gpd.read_file(filepath, engine='fiona')
        except Exception as e2:
            raise ValueError(f"Failed to read GeoJSON file: {e}. Also tried fiona: {e2}")
    
    # Filter for building features (Overture Maps uses 'building' in the type or properties)
    if 'type' in gdf.columns:
        gdf = gdf[gdf['type'] == 'building']
    elif 'building' in gdf.columns:
        gdf = gdf[gdf['building'].notna()]
    else:
        # If no explicit building filter, assume all features are buildings
        pass
    
    # Extract only polygon geometries
    polygons = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        
        # Handle both Polygon and MultiPolygon
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue  # Skip non-polygon geometries
        
        for poly in polys:
            # Get exterior coordinates
            coords = np.array(poly.exterior.coords)
            
            # Convert from (lon, lat) to (x, y) using equirectangular projection
            # First, compute the center of the bounding box for projection reference
            lon_center = np.mean(coords[:, 0])
            lat_center = np.mean(coords[:, 1])
            
            # Earth radius in meters
            R = 6371000.0
            
            # Equirectangular projection around the center
            x = R * np.radians(coords[:, 0] - lon_center) * np.cos(np.radians(lat_center))
            y = R * np.radians(coords[:, 1] - lat_center)
            
            # Stack into (N,2) array
            poly_local = np.column_stack([x, y])
            polygons.append(poly_local)
    
    # Compute the center of all polygons to align with bbox
    if polygons:
        all_coords = np.vstack(polygons)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        # Position buildings so left edge starts at 25% of domain width
        target_x_min = bbox[0] + 0.25 * (bbox[1] - bbox[0])
        shift_x = target_x_min - x_min
        
        # Center vertically within bbox
        bbox_center_y = (bbox[2] + bbox[3]) / 2
        poly_center_y = (y_min + y_max) / 2
        shift_y = bbox_center_y - poly_center_y
        
        # Apply shift and crop to bbox
        cropped_polygons = []
        for poly in polygons:
            poly_shifted = poly + np.array([shift_x, shift_y])
            
            # Simple bbox crop: keep polygons that intersect with bbox
            poly_x_min, poly_x_max = poly_shifted[:, 0].min(), poly_shifted[:, 0].max()
            poly_y_min, poly_y_max = poly_shifted[:, 1].min(), poly_shifted[:, 1].max()
            
            # Check if polygon intersects with bbox
            if (poly_x_max > bbox[0] and poly_x_min < bbox[1] and
                poly_y_max > bbox[2] and poly_y_min < bbox[3]):
                cropped_polygons.append(poly_shifted)
        
        return cropped_polygons
    
    return []
