import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def maccormack_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """MacCormack predictor-corrector scheme"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
                jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)
    
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
    
    # Predictor step (forward differences)
    du_dx_pred = (jnp.roll(u, -1, axis=0) - u) / dx
    du_dy_pred = (jnp.roll(u, -1, axis=1) - u) / dy
    dv_dx_pred = (jnp.roll(v, -1, axis=0) - v) / dx
    dv_dy_pred = (jnp.roll(v, -1, axis=1) - v) / dy
    
    adv_x_pred = u * du_dx_pred + v * du_dy_pred
    adv_y_pred = u * dv_dx_pred + v * dv_dy_pred
    diff_x = nu * laplacian(u, dx, dy)
    diff_y = nu * laplacian(v, dx, dy)
    div_tau_x, div_tau_y = sgs_stress_divergence(u, v, dx, dy, Cs)
    
    u_pred = u + dt * (-adv_x_pred + diff_x + div_tau_x)
    v_pred = v + dt * (-adv_y_pred + diff_y + div_tau_y)
    
    # Corrector step (backward differences)
    du_dx_corr = (u_pred - jnp.roll(u_pred, 1, axis=0)) / dx
    du_dy_corr = (u_pred - jnp.roll(u_pred, 1, axis=1)) / dy
    dv_dx_corr = (v_pred - jnp.roll(v_pred, 1, axis=0)) / dx
    dv_dy_corr = (v_pred - jnp.roll(v_pred, 1, axis=1)) / dy
    
    adv_x_corr = u_pred * du_dx_corr + v_pred * du_dy_corr
    adv_y_corr = u_pred * dv_dx_corr + v_pred * dv_dy_corr
    
    u_star = 0.5 * (u + u_pred + dt * (-adv_x_corr + diff_x + div_tau_x))
    v_star = 0.5 * (v + v_pred + dt * (-adv_y_corr + diff_y + div_tau_y))
    
    return u_star, v_star
