import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def jos_stam_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Jos Stam's stable fluids advection (semi-Lagrangian)"""
    nx, ny = u.shape
    
    # Create coordinate grids
    i = jnp.arange(nx)
    j = jnp.arange(ny)
    I, J = jnp.meshgrid(i, j, indexing='ij')
    
    # Convert to physical coordinates
    x = I * dx
    y = J * dy
    
    # Backtrace positions (where fluid came from)
    x_prev = x - dt * u
    y_prev = y - dt * v
    
    # Clamp to domain boundaries
    x_prev = jnp.clip(x_prev, 0.0, (nx-1) * dx)
    y_prev = jnp.clip(y_prev, 0.0, (ny-1) * dy)
    
    # Convert back to grid indices
    i_prev = x_prev / dx
    j_prev = y_prev / dy
    
    # Bilinear interpolation
    i0 = jnp.floor(i_prev).astype(int)
    i1 = jnp.minimum(i0 + 1, nx - 1)
    j0 = jnp.floor(j_prev).astype(int)
    j1 = jnp.minimum(j0 + 1, ny - 1)
    
    # Interpolation weights
    sx = i_prev - i0
    sy = j_prev - j0
    
    # Bilinear interpolation for u
    u00 = u[i0, j0]
    u01 = u[i0, j1]
    u10 = u[i1, j0]
    u11 = u[i1, j1]
    u_interp = (1-sx) * (1-sy) * u00 + sx * (1-sy) * u10 + (1-sx) * sy * u01 + sx * sy * u11
    
    # Bilinear interpolation for v
    v00 = v[i0, j0]
    v01 = v[i0, j1]
    v10 = v[i1, j0]
    v11 = v[i1, j1]
    v_interp = (1-sx) * (1-sy) * v00 + sx * (1-sy) * v10 + (1-sx) * sy * v01 + sx * sy * v11
    
    # Add diffusion and SGS terms
    def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
                jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)
    
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def sgs_stress_divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, Cs: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        du_dx = grad_x(u, dx)
        du_dy = grad_y(u, dy)
        dv_dx = grad_x(v, dx)
        dv_dy = grad_y(v, dy)
        
        Sxx = du_dx
        Syy = dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        
        mag_S = jnp.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2))
        Delta = jnp.sqrt(dx * dy)
        nu_sgs = (Cs * Delta)**2 * mag_S
        
        tau_xx = -2.0 * nu_sgs * Sxx
        tau_yy = -2.0 * nu_sgs * Syy
        tau_xy = -2.0 * nu_sgs * Sxy
        
        div_tau_x = grad_x(tau_xx, dx) + grad_y(tau_xy, dy)
        div_tau_y = grad_x(tau_xy, dx) + grad_y(tau_yy, dy)
        
        return div_tau_x, div_tau_y
    
    diff_u = nu * laplacian(u_interp, dx, dy)
    diff_v = nu * laplacian(v_interp, dx, dy)
    div_tau_x, div_tau_y = sgs_stress_divergence(u_interp, v_interp, dx, dy, Cs)
    
    u_star = u_interp + dt * (diff_u + div_tau_x)
    v_star = v_interp + dt * (diff_v + div_tau_y)
    
    return u_star, v_star
