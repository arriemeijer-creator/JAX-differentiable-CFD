"""
Boundary conditions for LBM
"""

import jax
import jax.numpy as jnp
from typing import Tuple


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
        nx: Grid size in x (unused, extracted from f.shape)
        ny: Grid size in y (unused, extracted from f.shape)
        mask: Obstacle mask (1=fluid, 0=solid) - used to avoid setting inlet on obstacle
        opposite: Opposite direction indices (9,) - required for JIT compatibility
        outlet_type: Type of outlet boundary ('convective', 'zou_he', 'extrapolation')
        bc_mode: Boundary condition mode ('supply' = inlet left, outlet right; 'extract' = inlet right, outlet left)
    
    Returns:
        f_bc: Distribution with inlet/outlet conditions
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Extract grid dimensions from f shape to avoid JAX tracing issues
    nx = f.shape[1]
    ny = f.shape[2]
    
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
    
    # Extract grid dimensions from f shape to avoid JAX tracing issues
    nx = f.shape[1]
    ny = f.shape[2]
    
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


def apply_inlet_bc(f: jnp.ndarray, boundary: str, u_inlet: float, rho_inlet: float,
                    cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                    cs_squared: float, nx: int, ny: int) -> jnp.ndarray:
    """
    Apply inlet boundary condition (equilibrium with prescribed velocity)
    
    Args:
        f: Distribution function (9, nx, ny)
        boundary: Which boundary to apply ('left', 'right', 'top', 'bottom')
        u_inlet: Inlet velocity magnitude
        rho_inlet: Inlet density
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
    
    Returns:
        f_bc: Distribution with inlet condition applied
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Derive shapes from input array
    nx_actual = f.shape[1]
    ny_actual = f.shape[2]
    
    if boundary == 'left':
        # Inlet on left boundary (x=0), flow goes right
        u_field = jnp.full((ny_actual, 1), u_inlet)
        v_field = jnp.zeros((ny_actual, 1))
        rho_field = jnp.full((ny_actual, 1), rho_inlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, 0, :].set(f_eq[:, :, 0])
    elif boundary == 'right':
        # Inlet on right boundary (x=nx-1), flow goes left
        u_field = jnp.full((ny_actual, 1), -u_inlet)
        v_field = jnp.zeros((ny_actual, 1))
        rho_field = jnp.full((ny_actual, 1), rho_inlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, -1, :].set(f_eq[:, :, 0])
    elif boundary == 'top':
        # Inlet on top boundary (y=ny-1), flow goes down
        u_field = jnp.zeros((nx_actual, 1))
        v_field = jnp.full((nx_actual, 1), -u_inlet)
        rho_field = jnp.full((nx_actual, 1), rho_inlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, -1].set(f_eq[:, :, 0])
    elif boundary == 'bottom':
        # Inlet on bottom boundary (y=0), flow goes up
        u_field = jnp.zeros((nx_actual, 1))
        v_field = jnp.full((nx_actual, 1), u_inlet)
        rho_field = jnp.full((nx_actual, 1), rho_inlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, 0].set(f_eq[:, :, 0])
    
    return f_bc


def apply_outlet_bc(f: jnp.ndarray, boundary: str, u_ref: float,
                    cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                    cs_squared: float, nx: int, ny: int, outlet_type: str = 'convective',
                    outlet_pressure: float = 0.0) -> jnp.ndarray:
    """
    Apply outlet boundary condition (convective or extrapolation)
    
    Args:
        f: Distribution function (9, nx, ny)
        boundary: Which boundary to apply ('left', 'right', 'top', 'bottom')
        u_ref: Reference velocity for convection
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        outlet_type: Type of outlet ('convective', 'extrapolation')
        outlet_pressure: Outlet pressure in Pa (negative for suction)
    
    Returns:
        f_bc: Distribution with outlet condition applied
    """
    f_bc = f.copy()
    
    # Derive shapes from input array
    nx_actual = f.shape[1]
    ny_actual = f.shape[2]
    
    # Convert pressure to density (LBM units)
    # In LBM, pressure p = rho * cs^2, so rho = p / cs^2
    # For small pressure differences, we can use rho_outlet = rho_ref + delta_p / cs^2
    # Add scaling factor to convert physical Pa to LBM units (typical LBM pressure differences are ~0.01-0.1)
    pressure_scale = 0.001  # Scale factor: 1 Pa = 0.001 LBM pressure units
    rho_ref = 1.0  # Reference density
    rho_outlet = rho_ref + (outlet_pressure * pressure_scale) / cs_squared
    
    if boundary == 'left':
        if outlet_type == 'convective':
            f_bc = apply_convective_outlet_left(f_bc, u_ref, dt=1.0, dx=1.0, nx=nx_actual, ny=ny_actual)
        else:  # extrapolation
            f_bc = apply_extrapolation_outlet_left(f_bc, nx=nx_actual, ny=ny_actual, order=1)
        # Apply pressure correction if non-zero (use JAX conditional)
        f_bc = jax.lax.cond(
            outlet_pressure != 0.0,
            lambda f: apply_pressure_outlet(f, boundary, rho_outlet, cx, cy, w, cs_squared, nx_actual, ny_actual),
            lambda f: f,
            f_bc
        )
    elif boundary == 'right':
        if outlet_type == 'convective':
            f_bc = apply_convective_outlet(f_bc, u_ref, dt=1.0, dx=1.0, nx=nx_actual, ny=ny_actual)
        else:  # extrapolation
            f_bc = apply_extrapolation_outlet(f_bc, nx=nx_actual, ny=ny_actual, order=1)
        # Apply pressure correction if non-zero (use JAX conditional)
        f_bc = jax.lax.cond(
            outlet_pressure != 0.0,
            lambda f: apply_pressure_outlet(f, boundary, rho_outlet, cx, cy, w, cs_squared, nx_actual, ny_actual),
            lambda f: f,
            f_bc
        )
    elif boundary == 'top':
        # Zero-gradient extrapolation for top
        f_bc = f_bc.at[:, :, -1].set(f[:, :, -2])
        # Apply pressure correction if non-zero (use JAX conditional)
        f_bc = jax.lax.cond(
            outlet_pressure != 0.0,
            lambda f: apply_pressure_outlet(f, boundary, rho_outlet, cx, cy, w, cs_squared, nx_actual, ny_actual),
            lambda f: f,
            f_bc
        )
    elif boundary == 'bottom':
        # Zero-gradient extrapolation for bottom
        f_bc = f_bc.at[:, :, 0].set(f[:, :, 1])
        # Apply pressure correction if non-zero (use JAX conditional)
        f_bc = jax.lax.cond(
            outlet_pressure != 0.0,
            lambda f: apply_pressure_outlet(f, boundary, rho_outlet, cx, cy, w, cs_squared, nx_actual, ny_actual),
            lambda f: f,
            f_bc
        )
    
    return f_bc


def apply_pressure_outlet(f: jnp.ndarray, boundary: str, rho_outlet: float,
                         cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                         cs_squared: float, nx: int, ny: int) -> jnp.ndarray:
    """
    Apply pressure-based outlet boundary condition using equilibrium distribution
    
    Args:
        f: Distribution function (9, nx, ny)
        boundary: Which boundary to apply ('left', 'right', 'top', 'bottom')
        rho_outlet: Outlet density (derived from pressure)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
    
    Returns:
        f_bc: Distribution with pressure outlet condition applied
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    # Get velocity at the boundary (use extrapolated velocity from interior)
    if boundary == 'left':
        # Use velocity from interior (x=1)
        u_boundary = jnp.mean(f_bc[1, 1, :] - f_bc[3, 1, :] + f_bc[5, 1, :] - f_bc[7, 1, :]) / rho_outlet
        v_boundary = jnp.mean(f_bc[2, 1, :] - f_bc[4, 1, :] + f_bc[5, 1, :] - f_bc[6, 1, :]) / rho_outlet
        # Set equilibrium at left boundary (x=0)
        u_field = jnp.full((1, ny), u_boundary)
        v_field = jnp.full((1, ny), v_boundary)
        rho_field = jnp.full((1, ny), rho_outlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, 0, :].set(f_eq[:, 0, :])
    elif boundary == 'right':
        # Use velocity from interior (x=nx-2)
        u_boundary = jnp.mean(f_bc[1, nx-2, :] - f_bc[3, nx-2, :] + f_bc[5, nx-2, :] - f_bc[7, nx-2, :]) / rho_outlet
        v_boundary = jnp.mean(f_bc[2, nx-2, :] - f_bc[4, nx-2, :] + f_bc[5, nx-2, :] - f_bc[6, nx-2, :]) / rho_outlet
        # Set equilibrium at right boundary (x=nx-1)
        u_field = jnp.full((1, ny), u_boundary)
        v_field = jnp.full((1, ny), v_boundary)
        rho_field = jnp.full((1, ny), rho_outlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, -1, :].set(f_eq[:, 0, :])
    elif boundary == 'top':
        # Use velocity from interior (y=ny-2)
        u_boundary = jnp.mean(f_bc[1, :, ny-2] - f_bc[3, :, ny-2] + f_bc[5, :, ny-2] - f_bc[7, :, ny-2]) / rho_outlet
        v_boundary = jnp.mean(f_bc[2, :, ny-2] - f_bc[4, :, ny-2] + f_bc[5, :, ny-2] - f_bc[6, :, ny-2]) / rho_outlet
        # Set equilibrium at top boundary (y=ny-1)
        u_field = jnp.full((nx, 1), u_boundary)
        v_field = jnp.full((nx, 1), v_boundary)
        rho_field = jnp.full((nx, 1), rho_outlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, -1].set(f_eq[:, :, 0])
    elif boundary == 'bottom':
        # Use velocity from interior (y=1)
        u_boundary = jnp.mean(f_bc[1, :, 1] - f_bc[3, :, 1] + f_bc[5, :, 1] - f_bc[7, :, 1]) / rho_outlet
        v_boundary = jnp.mean(f_bc[2, :, 1] - f_bc[4, :, 1] + f_bc[5, :, 1] - f_bc[6, :, 1]) / rho_outlet
        # Set equilibrium at bottom boundary (y=0)
        u_field = jnp.full((nx, 1), u_boundary)
        v_field = jnp.full((nx, 1), v_boundary)
        rho_field = jnp.full((nx, 1), rho_outlet)
        f_eq = equilibrium(rho_field, u_field, v_field, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, 0].set(f_eq[:, :, 0])
    
    return f_bc


def apply_farfield_bc(f: jnp.ndarray, boundary: str,
                      cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                      cs_squared: float, nx: int, ny: int) -> jnp.ndarray:
    """
    Apply far-field boundary condition (free-slip, zero normal velocity)
    
    Args:
        f: Distribution function (9, nx, ny)
        boundary: Which boundary to apply ('left', 'right', 'top', 'bottom')
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
    
    Returns:
        f_bc: Distribution with far-field condition applied
    """
    from .collision import equilibrium
    
    f_bc = f.copy()
    
    if boundary == 'left':
        # Free-slip left wall: set normal velocity (u) to zero, preserve tangential (v)
        rho_wall = jnp.sum(f_bc[:, 0, :], axis=0)
        u_wall = jnp.zeros_like(rho_wall)  # No normal velocity
        v_wall = (f_bc[2, 0, :] - f_bc[4, 0, :] + f_bc[5, 0, :] - f_bc[6, 0, :] - f_bc[7, 0, :] + f_bc[8, 0, :]) / rho_wall
        
        u_wall_2d = u_wall[None, :]
        v_wall_2d = v_wall[None, :]
        rho_wall_2d = rho_wall[None, :]
        
        f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, 0, :].set(f_eq_wall[:, :, 0])
    elif boundary == 'right':
        # Free-slip right wall: set normal velocity (u) to zero, preserve tangential (v)
        rho_wall = jnp.sum(f_bc[:, -1, :], axis=0)
        u_wall = jnp.zeros_like(rho_wall)  # No normal velocity
        v_wall = (f_bc[2, -1, :] - f_bc[4, -1, :] + f_bc[5, -1, :] - f_bc[6, -1, :] - f_bc[7, -1, :] + f_bc[8, -1, :]) / rho_wall
        
        u_wall_2d = u_wall[None, :]
        v_wall_2d = v_wall[None, :]
        rho_wall_2d = rho_wall[None, :]
        
        f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, -1, :].set(f_eq_wall[:, :, 0])
    elif boundary == 'top':
        # Free-slip top wall: set normal velocity (v) to zero, preserve tangential (u)
        rho_wall = jnp.sum(f_bc[:, :, -1], axis=0)
        u_wall = (f_bc[1, :, -1] - f_bc[3, :, -1] + f_bc[5, :, -1] - f_bc[6, :, -1] - f_bc[7, :, -1] + f_bc[8, :, -1]) / rho_wall
        v_wall = jnp.zeros_like(rho_wall)  # No normal velocity
        
        u_wall_2d = u_wall[None, :]
        v_wall_2d = jnp.zeros_like(u_wall_2d)
        rho_wall_2d = rho_wall[None, :]
        
        f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, -1].set(f_eq_wall[:, :, 0])
    elif boundary == 'bottom':
        # Free-slip bottom wall: set normal velocity (v) to zero, preserve tangential (u)
        rho_wall = jnp.sum(f_bc[:, :, 0], axis=0)
        u_wall = (f_bc[1, :, 0] - f_bc[3, :, 0] + f_bc[5, :, 0] - f_bc[6, :, 0] - f_bc[7, :, 0] + f_bc[8, :, 0]) / rho_wall
        v_wall = jnp.zeros_like(rho_wall)  # No normal velocity
        
        u_wall_2d = u_wall[None, :]
        v_wall_2d = jnp.zeros_like(u_wall_2d)
        rho_wall_2d = rho_wall[None, :]
        
        f_eq_wall = equilibrium(rho_wall_2d, u_wall_2d, v_wall_2d, cx, cy, w, cs_squared)
        f_bc = f_bc.at[:, :, 0].set(f_eq_wall[:, :, 0])
    
    return f_bc


def apply_wall_bc(f: jnp.ndarray, boundary: str, opposite: jnp.ndarray) -> jnp.ndarray:
    """
    Apply wall boundary condition (no-slip bounce-back)
    
    Args:
        f: Distribution function (9, nx, ny)
        boundary: Which boundary to apply ('left', 'right', 'top', 'bottom')
        opposite: Opposite direction indices (9,)
    
    Returns:
        f_bc: Distribution with wall condition applied
    """
    f_bc = f.copy()
    
    if boundary == 'left':
        f_bc = f_bc.at[:, 0, :].set(f_bc[opposite, 0, :])
    elif boundary == 'right':
        f_bc = f_bc.at[:, -1, :].set(f_bc[opposite, -1, :])
    elif boundary == 'top':
        f_bc = f_bc.at[:, :, -1].set(f_bc[opposite, :, -1])
    elif boundary == 'bottom':
        f_bc = f_bc.at[:, :, 0].set(f_bc[opposite, :, 0])
    
    return f_bc


def apply_per_boundary_conditions(f: jnp.ndarray, mask: jnp.ndarray,
                                   opposite: jnp.ndarray,
                                   bc_left: str, bc_right: str, bc_top: str, bc_bottom: str,
                                   u_inlet: float, rho_inlet: float,
                                   cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                                   cs_squared: float, nx: int, ny: int,
                                   outlet_type: str = 'convective',
                                   outlet_pressure_left: float = 0.0,
                                   outlet_pressure_right: float = 0.0,
                                   outlet_pressure_top: float = 0.0,
                                   outlet_pressure_bottom: float = 0.0) -> jnp.ndarray:
    """
    Apply boundary conditions based on per-boundary configuration
    
    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask
        opposite: Opposite direction indices (9,)
        bc_left: Boundary condition type for left ('inlet', 'outlet', 'farfield', 'wall')
        bc_right: Boundary condition type for right ('inlet', 'outlet', 'farfield', 'wall')
        bc_top: Boundary condition type for top ('inlet', 'outlet', 'farfield', 'wall')
        bc_bottom: Boundary condition type for bottom ('inlet', 'outlet', 'farfield', 'wall')
        u_inlet: Inlet velocity magnitude
        rho_inlet: Inlet density
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        nx: Grid size in x
        ny: Grid size in y
        outlet_type: Type of outlet boundary ('convective', 'extrapolation')
        outlet_pressure_left: Outlet pressure for left boundary (Pa)
        outlet_pressure_right: Outlet pressure for right boundary (Pa)
        outlet_pressure_top: Outlet pressure for top boundary (Pa)
        outlet_pressure_bottom: Outlet pressure for bottom boundary (Pa)
    
    Returns:
        f_bc: Distribution with boundary conditions applied
    """
    # Apply bounce-back for obstacles first
    f_bc = apply_bounce_back(f, mask, opposite)
    
    # Apply left boundary condition
    if bc_left == 'inlet':
        f_bc = apply_inlet_bc(f_bc, 'left', u_inlet, rho_inlet, cx, cy, w, cs_squared, nx, ny)
    elif bc_left == 'outlet':
        f_bc = apply_outlet_bc(f_bc, 'left', u_inlet, cx, cy, w, cs_squared, nx, ny, outlet_type, outlet_pressure_left)
    elif bc_left == 'farfield':
        f_bc = apply_farfield_bc(f_bc, 'left', cx, cy, w, cs_squared, nx, ny)
    elif bc_left == 'wall':
        f_bc = apply_wall_bc(f_bc, 'left', opposite)
    
    # Apply right boundary condition
    if bc_right == 'inlet':
        f_bc = apply_inlet_bc(f_bc, 'right', u_inlet, rho_inlet, cx, cy, w, cs_squared, nx, ny)
    elif bc_right == 'outlet':
        f_bc = apply_outlet_bc(f_bc, 'right', u_inlet, cx, cy, w, cs_squared, nx, ny, outlet_type, outlet_pressure_right)
    elif bc_right == 'farfield':
        f_bc = apply_farfield_bc(f_bc, 'right', cx, cy, w, cs_squared, nx, ny)
    elif bc_right == 'wall':
        f_bc = apply_wall_bc(f_bc, 'right', opposite)
    
    # Apply top boundary condition
    if bc_top == 'inlet':
        f_bc = apply_inlet_bc(f_bc, 'top', u_inlet, rho_inlet, cx, cy, w, cs_squared, nx, ny)
    elif bc_top == 'outlet':
        f_bc = apply_outlet_bc(f_bc, 'top', u_inlet, cx, cy, w, cs_squared, nx, ny, outlet_type, outlet_pressure_top)
    elif bc_top == 'farfield':
        f_bc = apply_farfield_bc(f_bc, 'top', cx, cy, w, cs_squared, nx, ny)
    elif bc_top == 'wall':
        f_bc = apply_wall_bc(f_bc, 'top', opposite)
    
    # Apply bottom boundary condition
    if bc_bottom == 'inlet':
        f_bc = apply_inlet_bc(f_bc, 'bottom', u_inlet, rho_inlet, cx, cy, w, cs_squared, nx, ny)
    elif bc_bottom == 'outlet':
        f_bc = apply_outlet_bc(f_bc, 'bottom', u_inlet, cx, cy, w, cs_squared, nx, ny, outlet_type, outlet_pressure_bottom)
    elif bc_bottom == 'farfield':
        f_bc = apply_farfield_bc(f_bc, 'bottom', cx, cy, w, cs_squared, nx, ny)
    elif bc_bottom == 'wall':
        f_bc = apply_wall_bc(f_bc, 'bottom', opposite)
    
    return f_bc


def apply_boundary_conditions(f: jnp.ndarray, mask: jnp.ndarray,
                               opposite: jnp.ndarray, flow_type: str = 'von_karman',
                               u_inlet: float = 0.0, nx: int = 0, ny: int = 0,
                               cx: jnp.ndarray = None, cy: jnp.ndarray = None,
                               w: jnp.ndarray = None, cs_squared: float = None,
                               outlet_type: str = 'convective', bc_mode: str = 'supply',
                               bc_left: str = None, bc_right: str = None,
                               bc_top: str = None, bc_bottom: str = None,
                               custom_inlet_mask: jnp.ndarray = None,
                               custom_outlet_mask: jnp.ndarray = None,
                               custom_inlet_angle: float = 0.0,
                               custom_inlet_velocity: float = None,
                               T: jnp.ndarray = None,
                               thermal_inlet_temp: float = 20.0,
                               outlet_pressure_left: float = 0.0,
                               outlet_pressure_right: float = 0.0,
                               outlet_pressure_top: float = 0.0,
                               outlet_pressure_bottom: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply all boundary conditions based on flow type or per-boundary configuration
    
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
        bc_left: Boundary condition type for left (optional, uses default if None)
        bc_right: Boundary condition type for right (optional, uses default if None)
        bc_top: Boundary condition type for top (optional, uses default if None)
        bc_bottom: Boundary condition type for bottom (optional, uses default if None)
        custom_inlet_mask: Custom inlet mask from PNG (nx, ny)
        custom_outlet_mask: Custom outlet mask from PNG (nx, ny)
        custom_inlet_angle: Inlet direction for blue inlet pixels in degrees
        custom_inlet_velocity: Custom velocity magnitude for blue inlet regions (overrides u_inlet)
        outlet_pressure_left: Outlet pressure for left boundary (Pa)
        outlet_pressure_right: Outlet pressure for right boundary (Pa)
        outlet_pressure_top: Outlet pressure for top boundary (Pa)
        outlet_pressure_bottom: Outlet pressure for bottom boundary (Pa)
    
    Returns:
        f_bc: Distribution with boundary conditions
        T: Temperature field (may be modified by boundary conditions)
    """
    # If custom inlet/outlet masks are provided, use them
    if custom_inlet_mask is not None or custom_outlet_mask is not None:
        return apply_custom_boundary_conditions(
            f, mask, opposite, u_inlet, cx, cy, w, cs_squared,
            custom_inlet_mask, custom_outlet_mask,
            custom_inlet_angle=custom_inlet_angle,
            custom_inlet_velocity=custom_inlet_velocity,
            T=T,
            thermal_inlet_temp=thermal_inlet_temp
        )
    
    # If per-boundary configuration is provided, use it
    if bc_left is not None or bc_right is not None or bc_top is not None or bc_bottom is not None:
        # Use per-boundary configuration
        bc_left = bc_left if bc_left is not None else 'inlet'
        bc_right = bc_right if bc_right is not None else 'outlet'
        bc_top = bc_top if bc_top is not None else 'farfield'
        bc_bottom = bc_bottom if bc_bottom is not None else 'farfield'
        
        f_bc = apply_per_boundary_conditions(
            f, mask, opposite, bc_left, bc_right, bc_top, bc_bottom,
            u_inlet, 1.0, cx, cy, w, cs_squared, nx, ny, outlet_type,
            outlet_pressure_left, outlet_pressure_right, outlet_pressure_top, outlet_pressure_bottom
        )
        return f_bc, T
    
    # Otherwise, use flow-specific default boundary conditions
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
    
    return f_bc, T


def apply_custom_boundary_conditions(f: jnp.ndarray, mask: jnp.ndarray,
                                     opposite: jnp.ndarray, u_inlet: float,
                                     cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray,
                                     cs_squared: float,
                                     custom_inlet_mask: jnp.ndarray = None,
                                     custom_outlet_mask: jnp.ndarray = None,
                                     custom_inlet_angle: float = 0.0,
                                     custom_inlet_velocity: float = None,
                                     T: jnp.ndarray = None,
                                     thermal_inlet_temp: float = 20.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply custom boundary conditions based on PNG masks
    - Blue pixels: inlet (supply with prescribed velocity and temperature)
    - Red scale: solid temperature (R=0 -> min_temp, R=255 -> max_temp)
    - White pixels: fluid
    - Other colors: solid boundary

    Args:
        f: Distribution function (9, nx, ny)
        mask: Obstacle mask (1=fluid, 0=solid)
        opposite: Opposite direction indices (9,)
        u_inlet: Global inlet velocity magnitude (used if custom_inlet_velocity not provided)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        custom_inlet_mask: Custom inlet mask (nx, ny), 1 where inlet, 0 otherwise
        custom_outlet_mask: Custom outlet mask (nx, ny), 1 where outlet, 0 otherwise
        custom_inlet_angle: Inlet velocity angle in degrees
        custom_inlet_velocity: Custom velocity magnitude for blue inlet regions (overrides u_inlet)
        T: Temperature field (nx, ny)
        thermal_inlet_temp: Inlet temperature

    Returns:
        f_bc: Distribution with custom boundary conditions
        T: Temperature field with inlet temperature applied
    """
    from .collision import equilibrium

    f_bc = f.copy()

    # Apply bounce-back for obstacles (solid regions)
    f_bc = apply_bounce_back(f_bc, mask, opposite)

    # Apply inlet boundary condition at custom inlet locations
    if custom_inlet_mask is not None:
        # For inlet pixels, set equilibrium with prescribed velocity
        # Map user angle to inlet velocity vector
        # 0 degrees -> positive x, 90 degrees -> positive y
        # Could be extended to infer direction from mask geometry

        # Get macroscopic density at inlet locations
        rho_inlet = jnp.sum(f_bc, axis=0)

        # Use custom inlet velocity if provided, otherwise fall back to global u_inlet
        inlet_velocity = custom_inlet_velocity if custom_inlet_velocity is not None else u_inlet

        # Create velocity fields (base inlet velocity only - spin affects obstacle, not inlet)
        angle_rad = custom_inlet_angle * jnp.pi / 180.0
        u_field = jnp.where(custom_inlet_mask > 0.5, inlet_velocity * jnp.cos(angle_rad), 0.0)
        v_field = jnp.where(custom_inlet_mask > 0.5, inlet_velocity * jnp.sin(angle_rad), 0.0)
        
        # Compute equilibrium at all locations
        f_eq = equilibrium(rho_inlet, u_field, v_field, cx, cy, w, cs_squared)

        # Blend factor to avoid 'trapping' at inlet for high velocities.
        # alpha close to 1 enforces equilibrium strongly; lower values allow interior dynamics.
        alpha = jnp.clip(0.5 + 0.5 * (jnp.abs(u_inlet) / 0.3), 0.3, 0.95)

        # Apply relaxed equilibrium only at inlet locations (vectorized)
        for i in range(9):
            f_bc = f_bc.at[i].set(jnp.where(custom_inlet_mask > 0.5,
                                            alpha * f_eq[i] + (1.0 - alpha) * f_bc[i],
                                            f_bc[i]))

        # Apply inlet temperature at inlet locations
        # Note: Temperature boundary conditions are applied here for custom masks
        # since the standard _apply_temperature_boundary skips custom masks
        if T is not None:
            T = T.at[:].set(jnp.where(custom_inlet_mask > 0.5, thermal_inlet_temp, T))
    
    # Apply outlet boundary condition at custom outlet locations
    if custom_outlet_mask is not None:
        # For outlet pixels, apply a convective (Orlanski-type) update along the
        # flow direction (assumed same as inlet angle). This copies interior
        # populations into outlet pixels using first-order upwind to let flow exit.
        # Determine integer shift from inlet angle (points from inlet into domain).
        angle_rad = custom_inlet_angle * jnp.pi / 180.0
        sx = jnp.int32(jnp.sign(jnp.cos(angle_rad)))
        sy = jnp.int32(jnp.sign(jnp.sin(angle_rad)))

        # Courant number for convection
        dt = 1.0
        dx = 1.0
        c = jnp.clip(jnp.abs(u_inlet) * dt / dx, 0.0, 1.0)

        # Interior neighbor (one cell inside domain from outlet pixel)
        f_interior = jnp.roll(f_bc, shift=(-sx, -sy), axis=(1, 2))
        f_interior2 = jnp.roll(f_bc, shift=(-2 * sx, -2 * sy), axis=(1, 2))

        # Update outlet pixels with upwinded interior values
        for i in range(9):
            upwind = f_interior[i] * (1.0 - c) + f_interior2[i] * c
            f_bc = f_bc.at[i].set(jnp.where(custom_outlet_mask > 0.5, upwind, f_bc[i]))

    return f_bc, T
