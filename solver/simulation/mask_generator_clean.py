import jax.numpy as jnp

def _compute_mask(self):
    """Compute the obstacle mask based on simulation parameters."""
    
    if hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'naca_airfoil':
        from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
        
        # Parse NACA designation
        naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
        if len(naca_str) == 4:
            m, p, t = parse_naca_4digit(naca_str)
            airfoil_type = '4-digit'
        elif len(naca_str) == 5:
            cl, p, m, t = parse_naca_5digit(naca_str)
            airfoil_type = '5-digit'
        else:
            raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
        
        naca_params = NACAParams(
            airfoil_type=airfoil_type,
            designation=self.sim_params.naca_airfoil,
            chord_length=self.sim_params.naca_chord,
            angle_of_attack=self.sim_params.naca_angle,
            position_x=self.sim_params.naca_x,
            position_y=self.sim_params.naca_y
        )
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        # Get SDF from NACA function, then apply SHARP threshold
        from obstacles.naca_airfoils import naca_surface_distance
        if airfoil_type == '4-digit':
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, m, p, t)
        else:  # 5-digit
            sdf = naca_surface_distance(self.grid.X, self.grid.Y, naca_params.chord_length,
                                       naca_params.angle_of_attack, naca_params.position_x,
                                       naca_params.position_y, cl, p, m, t)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'cow':
        from obstacles.cow import sdf_cow_side
        # Compute cow position relative to grid bounds
        # Use cow_x and cow_y from sim_params if available, otherwise use defaults
        cow_x = getattr(self.sim_params, 'cow_x', self.grid.lx * 0.25)  # 25% of domain width default
        cow_y = getattr(self.sim_params, 'cow_y', self.grid.ly * 0.35)  # 35% of domain height default
        # Compute scale factor based on grid dimensions relative to reference (20x3.75)
        ref_lx = 20.0
        ref_ly = 3.75
        scale_x = self.grid.lx / ref_lx
        scale_y = self.grid.ly / ref_ly
        cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        sdf = sdf_cow_side(self.grid.X, self.grid.Y, cow_x, cow_y, cow_scale)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'three_cylinder_array':
        from obstacles.cylinder_array import sdf_three_cylinders
        cylinder_x = getattr(self.sim_params, 'cylinder_x', 5.0)
        cylinder_y = getattr(self.sim_params, 'cylinder_y', self.grid.ly / 2.0)
        cylinder_diameter = getattr(self.sim_params, 'cylinder_diameter', 0.5)
        cylinder_spacing = getattr(self.sim_params, 'cylinder_spacing', 0.5)
        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        sdf = sdf_three_cylinders(self.grid.X, self.grid.Y, cylinder_x, cylinder_y, cylinder_diameter, cylinder_spacing)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'solid_wall':
        # Solid wall: thin vertical wall with configurable position and y-bounds
        X, Y = self.grid.X, self.grid.Y

        # Get actual Y bounds of grid
        y_min = jnp.min(Y)
        y_max = jnp.max(Y)
        y_range = y_max - y_min

        # Wall parameters from sim_params
        wall_x = self.grid.lx * getattr(self.sim_params, 'solid_wall_x', 0.25)
        wall_thickness = self.grid.lx * 0.02  # 2% of domain width

        # Wall y-bounds from sim_params (as percentages of domain)
        wall_y_start = y_min + y_range * getattr(self.sim_params, 'solid_wall_y_bottom', 0.0)
        wall_y_end = y_min + y_range * getattr(self.sim_params, 'solid_wall_y_top', 0.5)

        # Create signed distance function (SDF) for wall
        # Distance to wall centerline in X, clamped in Y
        dx_wall = jnp.abs(X - wall_x) - wall_thickness / 2

        # For Y: distance to wall's Y-range (0 inside range)
        dy_above = Y - wall_y_end  # positive above wall
        dy_below = wall_y_start - Y  # positive below wall
        dy_wall = jnp.maximum(0, jnp.maximum(dy_above, dy_below))  # 0 inside wall height

        # Combined SDF: outside if outside in X OR outside in Y
        sdf = jnp.maximum(dx_wall, dy_wall)

        # SHARP mask: use simple threshold on SDF (no sigmoid smoothing)
        # Use user's epsilon setting from slider (eps = eps_multiplier * dx)
        epsilon = self.sim_params.eps  # User-controlled via GUI slider (now used as threshold)
        # SHARP mask: 1 in fluid (sdf > 0), 0 in solid (sdf < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'custom':
        from obstacles.freeform_drawer import create_freeform_mask_smooth
        custom_mask = getattr(self.sim_params, 'custom_mask', None)
        if custom_mask is not None:
            # Use user's epsilon setting from slider
            epsilon = self.sim_params.eps
            # Get obstacle center position from sliders
            center_x = getattr(self.sim_params, 'custom_x', self.grid.lx * 0.25)
            center_y = getattr(self.sim_params, 'custom_y', self.grid.ly * 0.5)
            # Scale custom obstacle to fit in domain while preserving aspect ratio
            # Use the smaller dimension to determine scale, so the drawing fits
            mask_height, mask_width = custom_mask.shape
            
            # Calculate scale to fit in domain (use 60% of smaller dimension)
            domain_min_dim = min(self.grid.lx, self.grid.ly)
            scale = domain_min_dim * 0.6
            
            # Use same scale for both dimensions to preserve aspect ratio
            scale_x = scale
            scale_y = scale
            
            # Calculate offset to center the obstacle at the specified position
            # offset is the center position
            offset_x = center_x
            offset_y = center_y
            
            mask = create_freeform_mask_smooth(self.grid.X, self.grid.Y, custom_mask, 
                                              scale_x=scale_x, scale_y=scale_y,
                                              offset_x=offset_x, offset_y=offset_y,
                                              smooth_width=epsilon)
            return mask
        else:
            # Fallback to cylinder if no custom mask - SHARP mask
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            epsilon = self.sim_params.eps
            # SHARP mask: 1 in fluid (phi > 0), 0 in solid (phi < 0)
            # Use epsilon as a small threshold to avoid numerical issues at exact boundary
            mask = jnp.where(phi > -epsilon, 1.0, 0.0)
            return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'urban_map':
        # Urban map: use precomputed SDF field from sim_params
        sdf_field = getattr(self.sim_params, 'sdf_field', None)
        print(f"DEBUG urban_map: obstacle_type={getattr(self.sim_params, 'obstacle_type', 'NOT_SET')}")
        print(f"DEBUG urban_map: sdf_field is {type(sdf_field)}, shape: {sdf_field.shape if sdf_field is not None else 'None'}")
        
        if sdf_field is None:
            print("ERROR: urban_map selected but no sdf_field set, falling back to cylinder")
            # Fall back to cylinder to prevent crash
            X, Y = self.grid.X, self.grid.Y
            phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
            epsilon = self.sim_params.eps
            mask = jnp.where(phi > -epsilon, 1.0, 0.0)
            print(f"FALLBACK: Using cylinder mask - min={mask.min():.3f}, max={mask.max():.3f}")
            return mask
        
        # Convert numpy SDF to JAX if needed
        import numpy as np
        if isinstance(sdf_field, np.ndarray):
            sdf_field = jnp.array(sdf_field)
        
        print(f"DEBUG urban_map: SDF min={sdf_field.min():.3f}, max={sdf_field.max():.3f}, mean={sdf_field.mean():.3f}")
        
        epsilon = self.sim_params.eps
        print(f"DEBUG urban_map: epsilon={epsilon:.6f}")
        
        # Check for individual building SDFs to preserve disconnected buildings
        individual_sdfs = getattr(self.sim_params, 'individual_sdfs', [])
        
        if individual_sdfs and len(individual_sdfs) > 0:
            print(f"DEBUG urban_map: Computing mask from {len(individual_sdfs)} individual building SDFs")
            # Start with all fluid (1.0)
            combined_mask = jnp.ones_like(self.grid.X)
            
            for i, ind_sdf in enumerate(individual_sdfs):
                if ind_sdf is None:
                    continue
                # Convert to JAX array if needed
                if isinstance(ind_sdf, np.ndarray):
                    ind_sdf = jnp.array(ind_sdf)
                
                # Use SHARP threshold for each building (not smooth sigmoid)
                # This prevents building gradients from merging
                ind_mask = jnp.where(ind_sdf > -epsilon, 1.0, 0.0)
                
                # Combine with minimum: if ANY building is solid (0), cell is solid
                combined_mask = jnp.minimum(combined_mask, ind_mask)
                
                solid_cells = jnp.sum(ind_mask < 0.5)
                print(f"DEBUG urban_map: Building {i} solid cells: {solid_cells}")
            
            print(f"DEBUG urban_map: Combined mask min={combined_mask.min():.3f}, max={combined_mask.max():.3f}, mean={combined_mask.mean():.3f}")
            return combined_mask
        else:
            # Fallback: use smooth sigmoid on combined SDF
            print(f"DEBUG urban_map: No individual SDFs, using combined SDF with smooth mask")
            from .mask_to_sdf import sdf_to_mask
            mask = sdf_to_mask(sdf_field, eps=epsilon, smooth=True)
            print(f"DEBUG urban_map: Mask min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
            return mask
    elif hasattr(self.sim_params, 'obstacle_type') and self.sim_params.obstacle_type == 'tesla_valve':
        
        def sdf_tesla_valve_simple(X, Y, valve_x, valve_y, stage_length, num_stages, main_width, branch_width, angle):
            """Hard-coded, bulletproof Tesla valve SDF"""
            
            # Shift to valve coordinates
            total_width = stage_length * num_stages
            x_shifted = X - valve_x
            y_shifted = Y - valve_y
            
            # Start with the main channel (open space, so POSITIVE SDF)
            # Main channel is a rectangle from -total_width/2 to total_width/2 in x,
            # and from -main_width/2 to main_width/2 in y
            channel_x = jnp.abs(x_shifted) - total_width/2
            channel_y = jnp.abs(y_shifted) - main_width/2
            channel_sdf = jnp.maximum(channel_x, channel_y)  # Positive outside channel, negative inside channel
            
            # Now add diagonal vanes as SOLID obstacles (NEGATIVE SDF where they are)
            vane_sdf = jnp.full_like(X, 1e6)  # Start with large positive (no vane)
            
            # Diagonal vane geometry
            # We'll create a rotated rectangle for ONE vane, then duplicate it
            for stage in range(num_stages):
                # Stage x-range
                stage_start = -total_width/2 + stage * stage_length
                stage_end = stage_start + stage_length
                stage_center_x = stage_start + stage_length/2
                
                # Vane position within stage (offset from left edge)
                vane_offset = stage_length * 0.3  # 30% into stage
                vane_x = stage_start + vane_offset
                
                # Length of vane: must span from wall to reach into channel
                # Calculate required length to reach center of channel
                required_len = (main_width/2) / jnp.sin(jnp.abs(angle))
                vane_len = required_len * 1.1  # 10% extra for overlap
                
                # ---------- TOP VANE (starts at top wall, points down-right) ----------
                # Vane from (vane_x, main_width/2) to (vane_x + vane_len*cos(angle), main_width/2 - vane_len*sin(angle))
                start_x = vane_x
                start_y = main_width/2
                end_x = start_x + vane_len * jnp.cos(angle)
                end_y = start_y - vane_len * jnp.sin(angle)
                center_x = (start_x + end_x) / 2
                center_y = (start_y + end_y) / 2
                
                # Rotated rectangle SDF
                cos_a = jnp.cos(angle)
                sin_a = jnp.sin(angle)
                x_rel = x_shifted - center_x
                y_rel = y_shifted - center_y
                x_rot = x_rel * cos_a + y_rel * sin_a
                y_rot = y_rel * cos_a - x_rel * sin_a
                dx = jnp.abs(x_rot) - vane_len/2
                dy = jnp.abs(y_rot) - branch_width/2
                outside = jnp.sqrt(jnp.maximum(dx, 0)**2 + jnp.maximum(dy, 0)**2)
                inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
                top_vane = -(outside + inside)  # NEGATIVE inside of vane (solid)
                
                # ---------- BOTTOM VANE (starts at bottom wall, points up-right) ----------
                start_x = vane_x
                start_y = -main_width/2
                end_x = start_x + vane_len * jnp.cos(angle)
                end_y = start_y + vane_len * jnp.sin(angle)
                center_x = (start_x + end_x) / 2
                center_y = (start_y + end_y) / 2
                
                x_rel = x_shifted - center_x
                y_rel = y_shifted - center_y
                x_rot = x_rel * cos_a + y_rel * sin_a
                y_rot = y_rel * cos_a - x_rel * sin_a
                dx = jnp.abs(x_rot) - vane_len/2
                dy = jnp.abs(y_rot) - branch_width/2
                outside = jnp.sqrt(jnp.maximum(dx, 0)**2 + jnp.maximum(dy, 0)**2)
                inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
                bottom_vane = -(outside + inside)  # NEGATIVE inside of vane (solid)
                
                # Combine vanes (take the minimum/most negative SDF)
                vane_sdf = jnp.minimum(vane_sdf, top_vane)
                vane_sdf = jnp.minimum(vane_sdf, bottom_vane)
            
            # Final SDF: channel (positive outside) combined with vanes (negative where solid)
            # Take the minimum: if either channel OR vane says solid, it's solid
            final_sdf = jnp.minimum(channel_sdf, vane_sdf)
            
            return final_sdf
        
        # Parameters from UI
        num_stages = getattr(self.sim_params, 'tesla_valve_stages', 3)
        stage_length = getattr(self.sim_params, 'tesla_valve_stage_length', 1.5)
        main_width = getattr(self.sim_params, 'tesla_valve_main_width', 0.4)
        branch_width = getattr(self.sim_params, 'tesla_valve_branch_width', 0.15)
        branch_angle = getattr(self.sim_params, 'tesla_valve_branch_angle', 0.6)  # radians
        valve_x = getattr(self.sim_params, 'tesla_valve_x', self.grid.lx * 0.5)
        valve_y = getattr(self.sim_params, 'tesla_valve_y', self.grid.ly * 0.5)
        is_forward = getattr(self.sim_params, 'tesla_valve_forward', True)
        
        if not is_forward:
            branch_angle = -branch_angle
        
        # Compute SDF using bulletproof function
        sdf = sdf_tesla_valve_simple(self.grid.X, self.grid.Y, valve_x, valve_y, 
                                      stage_length, num_stages, main_width, 
                                      branch_width, branch_angle)
        
        # Convert to mask
        epsilon = self.sim_params.eps
        mask = jnp.where(sdf > -epsilon, 1.0, 0.0)
        
        return mask
    else:
        # Fallback to cylinder if no custom mask - SHARP mask
        X, Y = self.grid.X, self.grid.Y
        phi = jnp.sqrt((X - self.geom.center_x)**2 + (Y - self.geom.center_y)**2) - self.geom.radius
        epsilon = self.sim_params.eps
        # SHARP mask: 1 in fluid (phi > 0), 0 in solid (phi < 0)
        # Use epsilon as a small threshold to avoid numerical issues at exact boundary
        mask = jnp.where(phi > -epsilon, 1.0, 0.0)
        return mask
