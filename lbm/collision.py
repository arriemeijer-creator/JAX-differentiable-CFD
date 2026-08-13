"""
Collision operators for LBM (BGK, MRT, LES)
"""

import jax
import jax.numpy as jnp


def equilibrium(rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, 
                cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray, 
                cs_squared: float) -> jnp.ndarray:
    """
    Compute equilibrium distribution function (f_eq)
    
    Args:
        rho: Density field (nx, ny)
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
    
    Returns:
        f_eq: Equilibrium distribution (9, nx, ny)
    """
    # Expand dimensions for broadcasting
    # rho, u, v: (nx, ny) -> (1, nx, ny)
    rho = rho[None, :, :]
    u = u[None, :, :]
    v = v[None, :, :]
    
    # Expand lattice vectors
    # cx, cy, w: (9,) -> (9, 1, 1)
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    w = w[:, None, None]
    
    # Compute velocity squared
    u_sq = u**2 + v**2
    
    # Compute dot product c_i · u
    c_dot_u = cx * u + cy * v
    
    # Compute equilibrium distribution
    # f_eq = w_i * rho * (1 + (c_i·u)/cs² + (c_i·u)²/(2*cs⁴) - u²/(2*cs²))
    term1 = 1.0 + c_dot_u / cs_squared
    term2 = (c_dot_u**2) / (2.0 * cs_squared**2)
    term3 = u_sq / (2.0 * cs_squared)
    
    f_eq = w * rho * (term1 + term2 - term3)
    
    return f_eq


def bgk_collision(f: jnp.ndarray, f_eq: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    BGK (Bhatnagar-Gross-Krook) collision operator
    
    Args:
        f: Distribution function (9, nx, ny)
        f_eq: Equilibrium distribution (9, nx, ny)
        omega: Collision frequency (1/tau)
    
    Returns:
        f_post: Post-collision distribution (9, nx, ny)
    """
    f_post = f - omega * (f - f_eq)
    return f_post


def compute_strain_rate(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute strain rate magnitude S = sqrt(2 * S_ij * S_ij)
    
    Args:
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y
    
    Returns:
        S_mag: Strain rate magnitude (nx, ny)
    """
    # Compute velocity gradients
    du_dx = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx)
    du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dy)
    dv_dx = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * dx)
    dv_dy = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dy)
    
    # Compute strain rate tensor components
    S_xx = du_dx
    S_yy = dv_dy
    S_xy = 0.5 * (du_dy + dv_dx)
    
    # Compute strain rate magnitude
    S_mag = jnp.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
    
    return S_mag


def box_filter_2d(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Apply box filter to a 2D field (simple averaging)
    
    Args:
        field: Input field (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y
    
    Returns:
        filtered: Filtered field (nx, ny)
    """
    # Simple 3x3 box filter
    kernel = jnp.ones((3, 3)) / 9.0
    
    # Pad the field
    padded = jnp.pad(field, 1, mode='wrap')
    
    # Apply convolution using roll operations
    filtered = jnp.zeros_like(field)
    for i in range(3):
        for j in range(3):
            shifted = jnp.roll(padded, shift=(i-1, j-1), axis=(0, 1))
            filtered += kernel[i, j] * shifted[1:-1, 1:-1]
    
    return filtered


@jax.jit
def constant_smagorinsky_lbm(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, 
                               delta: float, C_s: float = 0.17) -> jnp.ndarray:
    """
    Constant coefficient Smagorinsky model for LBM
    Computes eddy viscosity: nu_sgs = (C_s * delta)^2 * |S|
    
    Args:
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y
        delta: Filter width (typically = grid spacing)
        C_s: Smagorinsky constant
    
    Returns:
        nu_sgs: Eddy viscosity field (nx, ny)
    """
    # Compute strain rate magnitude
    S_mag = compute_strain_rate(u, v, dx, dy)
    
    # Compute eddy viscosity with scaling factor for visible effect
    # Scale by 100 to make effect more visible in LBM
    nu_sgs = 100.0 * (C_s * delta)**2 * S_mag
    
    return nu_sgs


@jax.jit
def dynamic_smagorinsky_lbm(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, 
                              delta: float, alpha: float = 2.0) -> jnp.ndarray:
    """
    Dynamic Smagorinsky model for LBM
    Dynamically computes C_s based on Germano identity
    
    Args:
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y
        delta: Filter width
        alpha: Test filter ratio (typically 2.0)
    
    Returns:
        nu_sgs: Eddy viscosity field (nx, ny)
    """
    # Step 1: Compute strain rate at grid level
    S_mag = compute_strain_rate(u, v, dx, dy)
    
    # Step 2: Apply test filter (coarser grid)
    u_test = box_filter_2d(u, dx, dy)
    v_test = box_filter_2d(v, dx, dy)
    
    # Step 3: Compute strain rate at test level
    S_mag_test = compute_strain_rate(u_test, v_test, dx, dy)
    
    # Step 4: Compute Leonard stress L_ij = ũ_iũ_j - (u_i u_j)_test
    uu_test = box_filter_2d(u * u, dx, dy)
    vv_test = box_filter_2d(v * v, dx, dy)
    uv_test = box_filter_2d(u * v, dx, dy)
    
    L_xx = u_test * u_test - uu_test
    L_yy = v_test * v_test - vv_test
    L_xy = u_test * v_test - uv_test
    
    # Step 5: Compute dynamic C_s (simplified)
    # Use local strain rate to compute C_s
    numerator = jnp.abs(L_xx) + jnp.abs(L_yy) + 2.0 * jnp.abs(L_xy)
    denominator = 2.0 * delta**2 * (alpha**2 * S_mag_test**2 + 1e-10)
    
    C_s_squared = numerator / denominator
    C_s = jnp.sqrt(C_s_squared)
    
    # Clip C_s to reasonable range
    C_s = jnp.clip(C_s, 0.0, 0.5)
    
    # Step 6: Compute eddy viscosity with dynamic C_s
    # Scale by 100 to make effect more visible in LBM
    nu_sgs = 100.0 * (C_s * delta)**2 * S_mag
    
    return nu_sgs


def bgk_collision_les(f: jnp.ndarray, f_eq: jnp.ndarray, omega: float, 
                       nu_sgs: jnp.ndarray, cs_squared: float) -> jnp.ndarray:
    """
    BGK collision with LES (Smagorinsky model)
    Adjusts relaxation time based on local eddy viscosity
    
    Args:
        f: Distribution function (9, nx, ny)
        f_eq: Equilibrium distribution (9, nx, ny)
        omega: Base collision frequency (1/tau)
        nu_sgs: Eddy viscosity field (nx, ny)
        cs_squared: Speed of sound squared
    
    Returns:
        f_post: Post-collision distribution (9, nx, ny)
    """
    # Compute effective relaxation time: tau_eff = tau + nu_sgs / cs^2
    tau = 1.0 / omega
    tau_eff = tau + nu_sgs / cs_squared
    omega_eff = 1.0 / tau_eff
    
    # Apply BGK with effective omega
    omega_eff = omega_eff[None, :, :]  # Expand to (1, nx, ny) for broadcasting
    f_post = f - omega_eff * (f - f_eq)
    
    return f_post


@jax.jit(static_argnames=['use_les', 'les_model'])
def collision_step(f: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                   cx: jnp.ndarray, cy: jnp.ndarray, w: jnp.ndarray, 
                   cs_squared: float, omega: float, use_les: bool = False,
                   les_model: str = 'dynamic_smagorinsky', smagorinsky_constant: float = 0.17,
                   dx: float = 0.1, dy: float = 0.1) -> jnp.ndarray:
    """
    Complete collision step: compute equilibrium and apply BGK with optional LES
    
    Args:
        f: Distribution function (9, nx, ny)
        rho: Density field (nx, ny)
        u: x-velocity field (nx, ny)
        v: y-velocity field (nx, ny)
        cx: Lattice velocity x-components (9,)
        cy: Lattice velocity y-components (9,)
        w: Lattice weights (9,)
        cs_squared: Speed of sound squared
        omega: Collision frequency
        use_les: Whether to use LES
        les_model: LES model ('dynamic_smagorinsky' or 'smagorinsky')
        smagorinsky_constant: Smagorinsky constant
        dx: Grid spacing in x
        dy: Grid spacing in y
    
    Returns:
        f_post: Post-collision distribution (9, nx, ny)
    """
    f_eq = equilibrium(rho, u, v, cx, cy, w, cs_squared)
    
    if use_les:
        # Compute eddy viscosity
        delta = (dx * dy) ** 0.5
        
        if les_model == 'smagorinsky':
            nu_sgs = constant_smagorinsky_lbm(u, v, dx, dy, delta, smagorinsky_constant)
        else:  # dynamic_smagorinsky
            nu_sgs = dynamic_smagorinsky_lbm(u, v, dx, dy, delta)
        
        # Apply BGK with LES
        f_post = bgk_collision_les(f, f_eq, omega, nu_sgs, cs_squared)
    else:
        # Standard BGK
        f_post = bgk_collision(f, f_eq, omega)
    
    return f_post
