import jax.numpy as jnp
import numpy as np
from scipy.ndimage import distance_transform_edt


def generate_urban_sdf(nx=512, ny=192):
    """
    Generate a 2D urban signed distance field.

    Returns
    -------
    sdf : jnp.ndarray
        Shape (ny, nx)

        sdf > 0 : fluid
        sdf < 0 : inside buildings
        sdf = 0 : wall boundary
    """

    solid = np.zeros((ny, nx), dtype=np.uint8)

    rng = np.random.default_rng(42)

    def add_building(x0, y0, w, h):
        x1 = np.clip(x0 + w, 0, nx)
        y1 = np.clip(y0 + h, 0, ny)

        solid[y0:y1, x0:x1] = 1

    # ========================================================
    # Enhanced urban layout with better scattering
    # ========================================================

    # Create buildings scattered across the entire domain
    num_buildings = 20
    
    # Define multiple zones for better distribution
    zones = [
        (0.2, 0.4, 0.2, 0.8),   # Left zone
        (0.4, 0.6, 0.1, 0.9),   # Center-left zone  
        (0.6, 0.8, 0.2, 0.8),   # Center-right zone
        (0.8, 0.95, 0.1, 0.9),  # Right zone
    ]
    
    buildings_added = 0
    
    # Add buildings in different zones for better scattering
    for zone_x_min, zone_x_max, zone_y_min, zone_y_max in zones:
        # Calculate number of buildings for this zone
        zone_buildings = num_buildings // 4
        if buildings_added + zone_buildings > num_buildings:
            zone_buildings = num_buildings - buildings_added
        
        for _ in range(zone_buildings):
            # Random position within zone
            x_center = rng.uniform(zone_x_min * nx, zone_x_max * nx)
            y_center = rng.uniform(zone_y_min * ny, zone_y_max * ny)
            
            # Random building size
            w = rng.integers(15, 35)
            h = rng.integers(8, 25)
            
            # Random offset for more natural look
            x_offset = rng.uniform(-10, 10)
            y_offset = rng.uniform(-10, 10)
            
            # Calculate building position
            x0 = int(x_center - w/2 + x_offset)
            y0 = int(y_center - h/2 + y_offset)
            
            # Ensure building stays within bounds
            x0 = np.clip(x0, 0, nx - w)
            y0 = np.clip(y0, 0, ny - h)
            
            add_building(x0, y0, w, h)
            buildings_added += 1
    
    # Add a few larger buildings for variety
    remaining_buildings = num_buildings - buildings_added
    for i in range(remaining_buildings):
        # Larger buildings in strategic positions
        positions = [
            (int(0.3 * nx), int(0.3 * ny), 40, 30),
            (int(0.7 * nx), int(0.6 * ny), 35, 25),
        ]
        if i < len(positions):
            x0, y0, w, h = positions[i]
            add_building(x0, y0, w, h)

    # ========================================================
    # Signed distance field
    # ========================================================

    outside = distance_transform_edt(solid == 0)
    inside = distance_transform_edt(solid == 1)

    sdf = outside - inside

    return jnp.array(sdf, dtype=jnp.float32)


# ============================================================
# Usage
# ============================================================

if __name__ == "__main__":
    sdf = generate_urban_sdf()

    print(sdf.shape)
    print(sdf.min(), sdf.max())
