"""
Boundary conditions for LBM
"""

import jax
import jax.numpy as jnp


def apply_bounce_back(f: jnp.ndarray, mask: jnp.ndarray, 
                      opposite: jnp.ndarray, eta_max: float = 1.0) -> jnp.ndarray:
    """
    Apply bounce-back boundary condition at obstacle nodes only
    (NOT for top/bottom walls - those use free-slip)
    
    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask (1 = fluid, 0 = solid, with smooth transitions or continuous grayscale)
        opposite: Opposite direction indices (9,)
        eta_max: Maximum penalization strength (default 1.0 for full bounce-back)
    
    Returns:
        f_bb: Distribution with bounce-back applied
    """
    f_bb = f.copy()
    
    # Check if mask is continuous (grayscale) or binary
    # For continuous masks, apply partial bounce-back based on mask value
    # For binary masks, apply full bounce-back
    
    # Expand mask to match f dimensions: (nx, ny) -> (9, nx, ny)
    mask_expanded = jnp.expand_dims(mask, axis=0)  # (1, nx, ny)
    
    # Compute penalization coefficient based on mask (linear scaling like NS solver)
    # mask = 1.0 (fluid) -> eta = 0 (no penalization)
    # mask = 0.0 (solid) -> eta = eta_max (full penalization)
    # mask = 0.5 (grey) -> eta = 0.5 * eta_max (half penalization)
    solid_fraction = 1.0 - mask_expanded
    eta = eta_max * solid_fraction
    
    # Apply partial bounce-back: blend between original and bounced-back distributions
    # The blending is controlled by eta (penalization coefficient)
    # eta = 0 -> no bounce-back (fluid)
    # eta = eta_max -> full bounce-back (solid)
    # eta = 0.5 * eta_max -> partial bounce-back (grey)
    f_bounced = f[opposite]
    f_bb = (1.0 - eta) * f + eta * f_bounced
    
    return f_bb


def apply_inlet_outlet(f: jnp.ndarray, rho_inlet: float, u_inlet: float,
                       cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                       cs_squared: float, nx: int, ny: int, mask: jnp.ndarray = None,
                       opposite: jnp.ndarray = None, outlet_type: str = 'convective',
                       bc_mode: str = 'supply') -> jnp.ndarray:
    """
    Apply equilibrium boundary conditions at inlet and outlet
    for channel/von Karman flows
    
    Args:
        f: Distribution function (9, nx, ny)
        rho_inlet: Inlet density
        u_inlet: Inlet velocity (x-direction)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        mask: Obstacle mask (1=fluid, 0=solid) - used to avoid setting inlet on obstacle
        opposite: Opposite direction indices (9,) - required for JIT compatibility
        outlet_type: Type of outlet boundary ('convective', 'zou_he', 'extrapolation')
        bc_mode: Boundary condition mode ('supply' = inlet left, outlet right; 'extract' = inlet right, outlet left)
    
    Returns:
        f_bc: Distribution with inlet/outlet conditions
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    if bc_mode == 'supply':
        # Supply mode: inlet on left (x=0), outlet on right (x=nx-1)
        # Inlet (left boundary, x=0)
        u_inlet_field = jnp.full((ny, 1), u_inlet)
        v_inlet_field = jnp.zeros((ny, 1))
        rho_inlet_field = jnp.full((ny, 1), rho_inlet)
        
        f_eq_inlet = equilibrium(rho_inlet_field, u_inlet_field, v_inlet_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, 0, :].set(f_eq_inlet[:, :, 0])
        
        # Outlet (right boundary, x=nx-1)
        if outlet_type == 'convective':
            f_bc = apply_convective_outlet(f_bc, u_inlet, dt=1.0, dx=1.0, nx=nx, ny=ny)
        elif outlet_type == 'zou_he':
            f_bc = apply_zou_he_outlet(f_bc, rho_outlet=1.0, cx=cx, cy=cy, nx=nx, ny=ny)
        else:  # 'extrapolation' or default
            f_bc = apply_extrapolation_outlet(f_bc, nx=nx, ny=ny, order=1)
    else:
        # Extract mode: inlet on right (x=nx-1), outlet on left (x=0)
        # Inlet (right boundary, x=nx-1) - flow goes from right to left
        u_inlet_field = jnp.full((ny, 1), -u_inlet)  # Negative velocity for leftward flow
        v_inlet_field = jnp.zeros((ny, 1))
        rho_inlet_field = jnp.full((ny, 1), rho_inlet)
        
        f_eq_inlet = equilibrium(rho_inlet_field, u_inlet_field, v_inlet_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, -1, :].set(f_eq_inlet[:, :, 0])
        
        # Outlet (left boundary, x=0) - apply selected outlet type with negative reference velocity
        if outlet_type == 'convective':
            f_bc = apply_convective_outlet_left(f_bc, -u_inlet, dt=1.0, dx=1.0, nx=nx, ny=ny)
        elif outlet_type == 'zou_he':
            f_bc = apply_zou_he_outlet_left(f_bc, rho_outlet=1.0, cx=cx, cy=cy, nx=nx, ny=ny)
        else:  # 'extrapolation' or default
            f_bc = apply_extrapolation_outlet_left(f_bc, nx=nx, ny=ny, order=1)
    
    # Free-slip boundary conditions for top and bottom walls
    # Set normal velocity to zero, preserve tangential velocity
    if opposite is None:
        raise ValueError("opposite array must be provided for JIT compatibility")
    
    # For free-slip, we need to compute macroscopic variables at the wall
    # and set the distribution functions to equilibrium with v=0 (no normal flow)
    from .collision import equilibrium
    
    # Bottom wall (y=0) - free slip
    # Get macroscopic values at the wall
    rho_wall = jnp.sum(f_bc[:, :, 0], axis=0)
    u_wall = (f_bc[1, :, 0] - f_bc[3, :, 0] + f_bc[5, :, 0] - f_bc[6, :, 0] - f_bc[7, :, 0] + f_bc[8, :, 0]) / rho_wall
    v_wall = 0.0  # No normal velocity for free-slip
    
    # Set equilibrium with v=0 at bottom wall
    u_wall_2d = u_wall[None, :]
    v_wall_2d = jnp.zeros_like(u_wall_2d)
    rho_wall_2d = rho_wall[None, :]
    
    f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
    f_bc = f_bc.at[:, :, 0].set(f_eq_wall[:, :, 0])
    
    # Top wall (y=ny-1) - free slip
    rho_wall = jnp.sum(f_bc[:, :, -1], axis=0)
    u_wall = (f_bc[1, :, -1] - f_bc[3, :, -1] + f_bc[5, :, -1] - f_bc[6, :, -1] - f_bc[7, :, -1] + f_bc[8, :, -1]) / rho_wall
    v_wall = 0.0  # No normal velocity for free-slip
    
    u_wall_2d = u_wall[None, :]
    v_wall_2d = jnp.zeros_like(u_wall_2d)
    rho_wall_2d = rho_wall[None, :]
    
    f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
    f_bc = f_bc.at[:, :, -1].set(f_eq_wall[:, :, 0])
    
    return f_bc


def apply_lid_driven_cavity_bc(f: jnp.ndarray, u_lid: float,
                                cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                                cs_squared: float, nx: int, ny: int, opposite: jnp.ndarray) -> jnp.ndarray:
    """
    Apply lid-driven cavity boundary conditions
    - Top wall: moving lid with velocity u_lid
    - Other walls: no-slip (bounce-back)
    
    Args:
        f: Distribution function (9, nx, ny)
        u_lid: Lid velocity
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        opposite: Opposite direction indices (9,)
    
    Returns:
        f_bc: Distribution with cavity BCs
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Top wall (y=ny-1): moving lid
    # Create 2D fields for the boundary (nx, 1) to match equilibrium expectations
    u_lid_field = jnp.full((nx, 1), u_lid)
    v_lid_field = jnp.zeros((nx, 1))
    rho_lid_field = jnp.ones((nx, 1))  # Assume unit density
    
    f_eq_lid = equilibrium(rho_lid_field, u_lid_field, v_lid_field, cx, cy, w, cs_squared)
    
    # Apply to top boundary (squeeze the extra dimension)
    f_bc = f_bc.at[:, :, -1].set(f_eq_lid[:, :, 0])
    
    # Bottom wall (y=0): no-slip (bounce-back)
    f_bc = f_bc.at[:, :, 0].set(f_bc[opposite, :, 0])
    
    # Left wall (x=0): no-slip (bounce-back)
    f_bc = f_bc.at[:, 0, :].set(f_bc[opposite, 0, :])
    
    # Right wall (x=nx-1): no-slip (bounce-back)
    f_bc = f_bc.at[:, -1, :].set(f_bc[opposite, -1, :])
    
    return f_bc


def apply_taylor_green_bc(f: jnp.ndarray) -> jnp.ndarray:
    """
    Apply periodic boundary conditions for Taylor-Green vortex
    (Note: streaming already handles periodic via jnp.roll)
    
    Args:
        f: Distribution function (9, nx, ny)
    
    Returns:
        f_bc: Distribution (unchanged for periodic)
    """
    # Periodic BCs are handled by jnp.roll in streaming step
    return f


def apply_kelvin_helmholtz_bc(f: jnp.ndarray) -> jnp.ndarray:
    """
    Apply periodic boundary conditions for Kelvin-Helmholtz instability
    (Note: streaming already handles periodic via jnp.roll)
    
    Args:
        f: Distribution function (9, nx, ny)
    
    Returns:
        f_bc: Distribution (unchanged for periodic)
    """
    # Periodic BCs are handled by jnp.roll in streaming step
    # KH is fully periodic - no boundary conditions needed
    return f


def apply_convective_outlet(f: jnp.ndarray, u_ref: float, dt: float = 1.0, dx: float = 1.0,
                             nx: int = 0, ny: int = 0) -> jnp.ndarray:
    """
    Convective (Orlanski-type) outlet boundary condition.
    Assumes flow exits at reference velocity u_ref.
    
    Args:
        f: Distribution function (9, nx, ny)
        u_ref: Reference convection velocity (usually u_inlet)
        dt: Time step (default 1.0 for lattice units)
        dx: Grid spacing (default 1.0 for lattice units)
        nx, ny: Grid dimensions
    
    Returns:
        f_out: Updated distribution at outlet
    """
    f_out = f.copy()
    
    # Courant number for convection
    c = u_ref * dt / dx
    c = jnp.clip(c, 0.0, 1.0)  # Stability limit
    
    # First-order upwind convection for all directions
    for i in range(9):
        f_out = f_out.at[i, -1, :].set(
            f[i, -2, :] * (1 - c) + f[i, -3, :] * c
        )
    
    return f_out


def apply_zou_he_outlet(f: jnp.ndarray, rho_outlet: float = 1.0,
                         cx: jnp.ndarray = None, cy: jnp.ndarray = None,
                         nx: int = 0, ny: int = 0) -> jnp.ndarray:
    """
    Zou/He pressure boundary condition at outlet (right wall).
    Prescribes outlet density, computes unknown populations using vectorized operations.
    
    For D2Q9 at x = nx-1 (right wall):
    Unknown: f1, f5, f8
    Known:   f3, f6, f7 (from streaming)
    
    Args:
        f: Distribution function (9, nx, ny)
        rho_outlet: Prescribed outlet density (default 1.0)
        cx, cy: Lattice velocities
        nx, ny: Grid dimensions
    
    Returns:
        f_out: Updated distribution at outlet
    """
    f_out = f.copy()
    
    # At outlet boundary (x = nx-1)
    x_idx = -1
    
    # Known populations at outlet (vectorized)
    f3 = f[3, x_idx, :]  # left-going
    f6 = f[6, x_idx, :]  # bottom-left
    f7 = f[7, x_idx, :]  # top-left
    f2 = f[2, x_idx, :]  # up
    f4 = f[4, x_idx, :]  # down
    
    # Assume zero-gradient for velocity at outlet (u_x = 0)
    u_x = 0.0
    
    # Unknown populations (Zou/He relations) - vectorized
    f1 = f3 + (2/3) * rho_outlet * u_x
    f5 = f6 - 0.5 * (f2 - f4) + 0.5 * rho_outlet * cy[5]
    f8 = f7 + 0.5 * (f2 - f4) + 0.5 * rho_outlet * cy[8]
    
    f_out = f_out.at[1, x_idx, :].set(f1)
    f_out = f_out.at[5, x_idx, :].set(f5)
    f_out = f_out.at[8, x_idx, :].set(f8)
    
    return f_out


def apply_extrapolation_outlet(f: jnp.ndarray, nx: int = 0, ny: int = 0,
                                 order: int = 1) -> jnp.ndarray:
    """
    Extrapolation outlet boundary condition (right side).
    
    Args:
        f: Distribution function (9, nx, ny)
        nx, ny: Grid dimensions
        order: 1 for zero-gradient, 2 for linear extrapolation
    
    Returns:
        f_out: Updated distribution at outlet
    """
    f_out = f.copy()
    
    if order == 1:
        # Zero-gradient (copy from interior)
        f_out = f_out.at[:, -1, :].set(f[:, -2, :])
    else:
        # Linear extrapolation from two interior points
        f_out = f_out.at[:, -1, :].set(2 * f[:, -2, :] - f[:, -3, :])
    
    return f_out


def apply_convective_outlet_left(f: jnp.ndarray, u_ref: float, dt: float = 1.0, dx: float = 1.0,
                                  nx: int = 0, ny: int = 0) -> jnp.ndarray:
    """
    Convective (Orlanski-type) outlet boundary condition on left side.
    Assumes flow exits at reference velocity u_ref (negative for leftward flow).
    
    Args:
        f: Distribution function (9, nx, ny)
        u_ref: Reference convection velocity (negative for leftward flow)
        dt: Time step (default 1.0 for lattice units)
        dx: Grid spacing (default 1.0 for lattice units)
        nx, ny: Grid dimensions
    
    Returns:
        f_out: Updated distribution at left outlet
    """
    f_out = f.copy()
    
    # Courant number for convection (use absolute value for stability)
    c = jnp.abs(u_ref) * dt / dx
    c = jnp.clip(c, 0.0, 1.0)  # Stability limit
    
    # First-order upwind convection for all directions (left boundary)
    for i in range(9):
        f_out = f_out.at[i, 0, :].set(
            f[i, 1, :] * (1 - c) + f[i, 2, :] * c
        )
    
    return f_out


def apply_zou_he_outlet_left(f: jnp.ndarray, rho_outlet: float = 1.0,
                              cx: jnp.ndarray = None, cy: jnp.ndarray = None,
                              nx: int = 0, ny: int = 0) -> jnp.ndarray:
    """
    Zou/He pressure boundary condition at outlet (left wall).
    Prescribes outlet density, computes unknown populations using vectorized operations.
    
    For D2Q9 at x = 0 (left wall):
    Unknown: f3, f6, f7
    Known:   f1, f5, f8 (from streaming)
    
    Args:
        f: Distribution function (9, nx, ny)
        rho_outlet: Prescribed outlet density (default 1.0)
        cx, cy: Lattice velocities
        nx, ny: Grid dimensions
    
    Returns:
        f_out: Updated distribution at left outlet
    """
    f_out = f.copy()
    
    # At outlet boundary (x = 0)
    x_idx = 0
    
    # Known populations at outlet (vectorized)
    f1 = f[1, x_idx, :]  # right-going
    f5 = f[5, x_idx, :]  # bottom-right
    f8 = f[8, x_idx, :]  # top-right
    f2 = f[2, x_idx, :]  # up
    f4 = f[4, x_idx, :]  # down
    
    # Assume zero-gradient for velocity at outlet (u_x = 0)
    u_x = 0.0
    
    # Unknown populations (Zou/He relations) - vectorized
    f3 = f1 - (2/3) * rho_outlet * u_x
    f6 = f5 + 0.5 * (f2 - f4) - 0.5 * rho_outlet * cy[6]
    f7 = f8 - 0.5 * (f2 - f4) - 0.5 * rho_outlet * cy[7]
    
    f_out = f_out.at[3, x_idx, :].set(f3)
    f_out = f_out.at[6, x_idx, :].set(f6)
    f_out = f_out.at[7, x_idx, :].set(f7)
    
    return f_out


def apply_extrapolation_outlet_left(f: jnp.ndarray, nx: int = 0, ny: int = 0,
                                     order: int = 1) -> jnp.ndarray:
    """
    Extrapolation outlet boundary condition (left side).
    
    Args:
        f: Distribution function (9, nx, ny)
        nx, ny: Grid dimensions
        order: 1 for zero-gradient, 2 for linear extrapolation
    
    Returns:
        f_out: Updated distribution at left outlet
    """
    f_out = f.copy()
    
    if order == 1:
        # Zero-gradient (copy from interior)
        f_out = f_out.at[:, 0, :].set(f[:, 1, :])
    else:
        # Linear extrapolation from two interior points
        f_out = f_out.at[:, 0, :].set(2 * f[:, 1, :] - f[:, 2, :])
    
    return f_out


def apply_boundary_conditions(f: jnp.ndarray, mask: jnp.ndarray,
                               opposite: jnp.ndarray, flow_type: str = 'von_karman',
                               u_inlet: float = 0.0, nx: int = 0, ny: int = 0,
                               cx: jnp.ndarray = None, cy: jnp.ndarray = None,
                               w: jnp.ndarray = None, cs_squared: float = None,
                               outlet_type: str = 'convective', bc_mode: str = 'supply') -> jnp.ndarray:
    """
    Apply all boundary conditions based on flow type
    
    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask
        opposite: Opposite direction indices (9,)
        flow_type: Type of flow ('von_karman', 'lid_driven_cavity', 'taylor_green')
        u_inlet: Inlet velocity for channel flows
        nx: Grid size in x
        ny: Grid size in y
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        outlet_type: Type of outlet boundary ('convective', 'zou_he', 'extrapolation')
        bc_mode: Boundary condition mode ('supply' or 'extract')
    
    Returns:
        f_bc: Distribution with boundary conditions
    """
    # Apply bounce-back for obstacles
    f_bc = apply_bounce_back(f, mask, opposite)
    
    # Apply flow-specific boundary conditions
    if flow_type == 'von_karman':
        f_bc = apply_inlet_outlet(f_bc, 1.0, u_inlet, cx, cy, w, cs_squared, nx, ny, mask, opposite, outlet_type, bc_mode)
    elif flow_type == 'lid_driven_cavity':
        f_bc = apply_lid_driven_cavity_bc(f_bc, u_inlet, cx, cy, w, cs_squared, nx, ny, opposite)
    elif flow_type == 'taylor_green':
        f_bc = apply_taylor_green_bc(f_bc)
    elif flow_type == 'kelvin_helmholtz':
        f_bc = apply_kelvin_helmholtz_bc(f_bc)
    
    return f_bc
