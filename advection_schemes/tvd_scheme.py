import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def tvd_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17, limiter: str = 'minmod') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TVD scheme with flux limiters"""
    def flux_limiter(r: jnp.ndarray, limiter_type: str) -> jnp.ndarray:
        if limiter_type == 'minmod':
            return jnp.maximum(0.0, jnp.minimum(1.0, r))
        elif limiter_type == 'superbee':
            return jnp.maximum(0.0, jnp.maximum(jnp.minimum(1.0, 2*r), jnp.minimum(2.0, r)))
        elif limiter_type == 'van_leer':
            return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))
        else:  # minmod
            return jnp.maximum(0.0, jnp.minimum(1.0, r))
    
    def compute_flux(f: jnp.ndarray, axis: int, dt: float, dx: float, dy: float) -> jnp.ndarray:
        if axis == 0:
            f_upwind = jnp.roll(f, 1, axis=0)
            f_lw = 0.5 * (f + jnp.roll(f, -1, axis=0)) - \
                   0.5 * dt/dx * (f - jnp.roll(f, 1, axis=0))
        else:
            f_upwind = jnp.roll(f, 1, axis=1)
            f_lw = 0.5 * (f + jnp.roll(f, -1, axis=1)) - \
                   0.5 * dt/dy * (f - jnp.roll(f, 1, axis=1))
        
        # Flux limiter
        r = (f - jnp.roll(f, 1, axis=axis)) / (jnp.roll(f, -1, axis=axis) - f + 1e-10)
        phi = flux_limiter(r, limiter)
        
        return f_upwind + phi * (f_lw - f_upwind)
    
    # Compute TVD fluxes
    flux_x = compute_flux(u, 0, dt, dx, dy)
    flux_y = compute_flux(v, 1, dt, dx, dy)
    
    adv_x = (flux_x - jnp.roll(flux_x, 1, axis=0)) / dx
    adv_y = (flux_y - jnp.roll(flux_y, 1, axis=1)) / dy
    
    # Diffusion
    def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
                jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)
    
    diff_x = nu * laplacian(u, dx, dy)
    diff_y = nu * laplacian(v, dx, dy)
    
    # SGS model (simplified)
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
    
    div_tau_x, div_tau_y = sgs_stress_divergence(u, v, dx, dy, Cs)
    
    u_star = u + dt * (-adv_x + diff_x + div_tau_x)
    v_star = v + dt * (-adv_y + diff_y + div_tau_y)
    
    return u_star, v_star
