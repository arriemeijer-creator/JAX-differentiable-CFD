import jax.numpy as jnp

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
