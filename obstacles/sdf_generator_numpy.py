"""
Pure Python/Numpy version of SDF generator - no JAX dependency.
Converts building polygons to a signed distance field on a structured grid.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from matplotlib.path import Path
from typing import List


def polygons_to_sdf_numpy(polygons: List[np.ndarray], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Convert a list of polygons to a signed distance field on a structured grid.
    Pure Python/Numpy version - no JAX dependency.
    
    Args:
        polygons: List of (N,2) numpy arrays, each representing a polygon
        X: 2D meshgrid array of x coordinates
        Y: 2D meshgrid array of y coordinates
    
    Returns:
        sdf: 2D numpy array where sdf > 0 in fluid, sdf < 0 in solid
    """
    print(f"DEBUG: polygons_to_sdf_numpy called with {len(polygons)} polygons")
    nx, ny = X.shape
    
    # Flatten grid coordinates for point-in-polygon tests
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Initialize SDF as all positive (fluid)
    sdf = np.ones((nx, ny), dtype=np.float32) * 10.0
    individual_sdfs = []
    
    print(f"DEBUG: Initializing SDF with shape {sdf.shape}, type {type(sdf)}")
    
    # Process each building individually to avoid merging
    for i, poly in enumerate(polygons):
        # Create binary mask for this building only
        building_mask = np.zeros(nx * ny, dtype=np.float32)
        path = Path(poly)
        inside = path.contains_points(points)
        building_mask = inside.astype(np.float32)
        building_mask = building_mask.reshape(nx, ny)
        
        # Compute SDF for this building only
        distance_outside = distance_transform_edt(1.0 - building_mask)
        distance_inside = distance_transform_edt(building_mask)
        building_sdf = distance_outside - distance_inside
        
        # Take minimum with current SDF (union of buildings)
        sdf = np.minimum(sdf, building_sdf)
        
        # Store individual SDF for separate contour extraction
        individual_sdfs.append(building_sdf)
        
        if i < 10:  # Debug first 10 buildings
            negative_count = np.sum(building_sdf < 0)
            print(f"DEBUG: Building {i}: {negative_count} negative cells, SDF range: [{np.min(building_sdf):.3f}, {np.max(building_sdf):.3f}]")
        elif i == 10:  # Debug message after processing first 10 buildings
            print(f"DEBUG: Processed {i+1} buildings so far...")
    
    # No smoothing applied to preserve sharp building edges
    # Gaussian smoothing was causing buildings to merge into blobs
    # and lose their distinct geometric shapes
    # Keeping original SDF from distance_transform_edt for accuracy
    
    print(f"DEBUG: Final SDF range: [{np.min(sdf):.3f}, {np.max(sdf):.3f}], negative cells: {np.sum(sdf < 0)}")
    print(f"DEBUG: SDF type: {type(sdf)}, shape: {sdf.shape if hasattr(sdf, 'shape') else 'N/A'}")
    
    return sdf, individual_sdfs


def load_building_polygons_numpy(filepath: str, bbox: tuple, max_buildings: int = None, spinbox_widget=None) -> List[np.ndarray]:
    """
    Load building footprints from a GeoJSON file and convert to local coordinates.
    Pure Python version - no JAX dependency.
    
    Args:
        filepath: Path to GeoJSON file (Overture Maps format)
        bbox: Bounding box as (xmin, xmax, ymin, ymax) in meters (local coordinates)
    
    Returns:
        List of polygons, where each polygon is an (N,2) numpy array of (x,y) coordinates
    """
    import os
    import json
    print(f"DEBUG: Starting load_building_polygons_numpy for {filepath}")
    
    # Validate file exists and has content
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    print(f"Loading GeoJSON file: {filepath} (size: {file_size} bytes)")
    
    # Load GeoJSON as JSON (fallback if geopandas fails)
    try:
        import geopandas as gpd
        # Load GeoJSON with fallback for malformed files
        try:
            gdf = gpd.read_file(filepath, engine='pyogrio')
        except Exception as e:
            print(f"pyogrio failed, trying fiona: {e}")
            try:
                gdf = gpd.read_file(filepath, engine='fiona')
            except Exception as e2:
                print(f"Both engines failed, trying JSON fallback: {e2}")
                raise Exception("Use JSON fallback")
        
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
            from shapely.geometry import Polygon, MultiPolygon
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
        
    except Exception as e:
        print(f"Geopandas failed, using JSON fallback: {e}")
        
        # JSON fallback - manually parse GeoJSON
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        polygons = []
        if 'features' in data:
            print(f"DEBUG: Found {len(data['features'])} features in GeoJSON")
            for i, feature in enumerate(data['features']):
                if feature.get('type') != 'Feature':
                    continue
                
                geometry = feature.get('geometry', {})
                if geometry.get('type') not in ['Polygon', 'MultiPolygon']:
                    continue
                
                coords = geometry.get('coordinates', [])
                
                if geometry['type'] == 'Polygon':
                    # Polygon: [[[x1, y1], [x2, y2], ...]]
                    if coords and len(coords) > 0:
                        exterior = np.array(coords[0])
                        if len(exterior) > 0:
                            # Convert to local coordinates
                            lon_center = np.mean(exterior[:, 0])
                            lat_center = np.mean(exterior[:, 1])
                            R = 6371000.0
                            x = R * np.radians(exterior[:, 0] - lon_center) * np.cos(np.radians(lat_center))
                            y = R * np.radians(exterior[:, 1] - lat_center)
                            poly_local = np.column_stack([x, y])
                            polygons.append(poly_local)
                            print(f"DEBUG: Created polygon {len(polygons)-1}: {len(poly_local)} vertices")
                
                elif geometry['type'] == 'MultiPolygon':
                    # MultiPolygon: [[[...], [...]], [[...], [...]]]
                    for polygon_coords in coords:
                        if polygon_coords and len(polygon_coords) > 0:
                            exterior = np.array(polygon_coords[0])
                            if len(exterior) > 0:
                                # Convert to local coordinates
                                lon_center = np.mean(exterior[:, 0])
                                lat_center = np.mean(exterior[:, 1])
                                R = 6371000.0
                                x = R * np.radians(exterior[:, 0] - lon_center) * np.cos(np.radians(lat_center))
                                y = R * np.radians(exterior[:, 1] - lat_center)
                                poly_local = np.column_stack([x, y])
                                polygons.append(poly_local)
    
    # Compute the center of all polygons to align with bbox
    if polygons:
        all_coords = np.vstack(polygons)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        print(f"DEBUG: Original polygons bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
        print(f"DEBUG: Target bbox: x=[{bbox[0]:.1f}, {bbox[1]:.1f}], y=[{bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Calculate scaling factor to fit buildings in domain
        poly_width = x_max - x_min
        poly_height = y_max - y_min
        bbox_width = bbox[1] - bbox[0]
        bbox_height = bbox[3] - bbox[2]
        
        # Use smaller scale to ensure buildings fit with some margin
        scale_x = (bbox_width * 0.8) / poly_width  # 80% of domain width
        scale_y = (bbox_height * 0.8) / poly_height  # 80% of domain height
        scale = min(scale_x, scale_y)  # Use uniform scaling
        
        print(f"DEBUG: Building size: {poly_width:.1f}x{poly_height:.1f}m")
        print(f"DEBUG: Domain size: {bbox_width:.1f}x{bbox_height:.1f}m")
        print(f"DEBUG: Applying scale factor: {scale:.6f}")
        
        # Apply scaling to all polygons
        scaled_polygons = []
        for poly in polygons:
            scaled_poly = poly * scale
            scaled_polygons.append(scaled_poly)
        
        # Check individual building sizes and apply additional scaling if needed
        # We want each building to be small enough that 10 buildings fit with spacing
        max_building_width = bbox_width / 5.0  # Allow 5 columns of buildings
        max_building_height = bbox_height / 3.0  # Allow 3 rows of buildings
        
        additional_scale = 1.0
        for i, poly in enumerate(scaled_polygons):
            p_width = poly[:, 0].max() - poly[:, 0].min()
            p_height = poly[:, 1].max() - poly[:, 1].min()
            if p_width > max_building_width:
                s = max_building_width / p_width
                additional_scale = min(additional_scale, s)
            if p_height > max_building_height:
                s = max_building_height / p_height
                additional_scale = min(additional_scale, s)
        
        if additional_scale < 1.0:
            print(f"DEBUG: Applying additional per-building scale: {additional_scale:.6f}")
            for i in range(len(scaled_polygons)):
                scaled_polygons[i] = scaled_polygons[i] * additional_scale
            scale = scale * additional_scale
        
        # Recalculate bounds after scaling
        all_scaled_coords = np.vstack(scaled_polygons)
        x_min_scaled, x_max_scaled = all_scaled_coords[:, 0].min(), all_scaled_coords[:, 0].max()
        y_min_scaled, y_max_scaled = all_scaled_coords[:, 1].min(), all_scaled_coords[:, 1].max()
        
        # Position buildings so left edge starts at 25% of domain width
        # This leaves smaller empty space upstream and more downstream space
        target_x_min = bbox[0] + 0.25 * (bbox[1] - bbox[0])
        shift_x = target_x_min - x_min_scaled
        
        # Center vertically within bbox
        bbox_center_y = (bbox[2] + bbox[3]) / 2
        poly_center_y = (y_min_scaled + y_max_scaled) / 2
        shift_y = bbox_center_y - poly_center_y
        
        print(f"DEBUG: Scaled bounds: x=[{x_min_scaled:.1f}, {x_max_scaled:.1f}], y=[{y_min_scaled:.1f}, {y_max_scaled:.1f}]")
        print(f"DEBUG: Applying shift: x={shift_x:.1f}, y={shift_y:.1f}")
        
        # Apply shift and crop to bbox
        cropped_polygons = []
        for i, poly in enumerate(scaled_polygons):
            poly_shifted = poly + np.array([shift_x, shift_y])
            
            # Simple bbox crop: keep polygons that intersect with bbox
            poly_x_min, poly_x_max = poly_shifted[:, 0].min(), poly_shifted[:, 0].max()
            poly_y_min, poly_y_max = poly_shifted[:, 1].min(), poly_shifted[:, 1].max()
            
            if i < 3:  # Debug first few polygons
                print(f"DEBUG: Polygon {i}: x=[{poly_x_min:.1f}, {poly_x_max:.1f}], y=[{poly_y_min:.1f}, {poly_y_max:.1f}]")
            
            # Check if polygon intersects with bbox
            if (poly_x_max > bbox[0] and poly_x_min < bbox[1] and
                poly_y_max > bbox[2] and poly_y_min < bbox[3]):
                cropped_polygons.append(poly_shifted)
        
        # Filter buildings to prevent overcrowding and improve performance
        max_buildings = max_buildings if max_buildings is not None else len(cropped_polygons)
        print(f"DEBUG: Total polygons before filter: {len(cropped_polygons)}, max_buildings: {max_buildings}")
        print(f"DEBUG: Total polygons before filter: {len(cropped_polygons)}, max_buildings: {max_buildings}")
        
        if len(cropped_polygons) > max_buildings:
            print(f"DEBUG: Filtering from {len(cropped_polygons)} to {max_buildings} buildings")
            # Sort by area and keep largest ones
            areas = []
            for poly in cropped_polygons:
                coords = np.array(poly)
                area = 0.0
                for i in range(len(coords) - 1):
                    area += coords[i][0] * coords[i+1][1] - coords[i+1][0] * coords[i][1]
                areas.append(abs(area / 2.0))
            
            # Sort by area (largest first) and keep top max_buildings
            sorted_indices = np.argsort(areas)[::-1][:max_buildings]
            filtered_polygons = [cropped_polygons[i] for i in sorted_indices]
            # Add spacing between buildings to prevent overlapping, but keep within domain
            spaced_polygons = []
            for i, poly in enumerate(filtered_polygons):
                # Add spacing offset based on building index, but ensure within domain
                spacing_x = (i % 5) * 2.0  # 2m spacing in x direction (5 columns)
                spacing_y = (i // 5) * 1.0  # 1m spacing in y direction (2 rows)
                spaced_poly = poly + np.array([spacing_x, spacing_y])
                
                # Check if spaced building is still within domain bounds
                poly_coords = spaced_poly
                if (poly_coords[:, 0].min() >= bbox[0] - 1.0 and 
                    poly_coords[:, 0].max() <= bbox[1] + 1.0 and
                    poly_coords[:, 1].min() >= bbox[2] - 1.0 and 
                    poly_coords[:, 1].max() <= bbox[3] + 1.0):
                    spaced_polygons.append(spaced_poly)
                else:
                    # Keep original building if spacing would push it outside domain
                    spaced_polygons.append(poly)
            
            filtered_polygons = spaced_polygons
            print(f"DEBUG: Applied spacing to {len(filtered_polygons)} buildings")
            
            filtered_polygons = spaced_polygons
            print(f"DEBUG: Filtered to {len(filtered_polygons)} buildings with spacing")
        else:
            filtered_polygons = cropped_polygons
            print(f"DEBUG: No filtering needed, kept {len(filtered_polygons)} buildings")
            
            print(f"DEBUG: Kept {len(filtered_polygons)} largest buildings (filtered from {len(cropped_polygons)})")
        
        print(f"DEBUG: Returning {len(filtered_polygons)} polygons to SDF generation")
        return filtered_polygons
    
    return []
