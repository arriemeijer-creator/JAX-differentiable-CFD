import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def quick_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """QUICK scheme - 3rd order upwind-biased"""
    def quick_interp(f: jnp.ndarray, axis: int) -> jnp.ndarray:
        if axis == 0:  # x-direction
            f_upwind = jnp.roll(f, 1, axis=0)  # upstream
            f_center = f
            f_downwind = jnp.roll(f, -1, axis=0)
        else:  # y-direction
            f_upwind = jnp.roll(f, 1, axis=1)
            f_center = f
            f_downwind = jnp.roll(f, -1, axis=1)
        
        # QUICK interpolation: 3/4 upstream + 3/8 downstream - 1/8 upstream-upstream
        return 0.75 * f_center + 0.375 * f_downwind - 0.125 * f_upwind
    
    # Compute fluxes with QUICK
    u_face_x = quick_interp(u, 0)
    v_face_x = quick_interp(v, 0)
    u_face_y = quick_interp(u, 1)
    v_face_y = quick_interp(v, 1)
    
    adv_x = (u_face_x * (jnp.roll(u, -1, axis=0) - u) / dx +
             v_face_y * (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2*dy))
    adv_y = (u_face_x * (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2*dx) +
             v_face_y * (jnp.roll(v, -1, axis=1) - v) / dy)
    
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
