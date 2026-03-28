import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def rk3_step(u: jnp.ndarray, v: jnp.ndarray, dt: float, nu: float, dx: float, dy: float, mask: jnp.ndarray, Cs: float = 0.17) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """3rd order Runge-Kutta for time integration"""
    def compute_rhs(u: jnp.ndarray, v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Gradients
        def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
            return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
        
        def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
            return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
        
        du_dx = grad_x(u, dx)
        du_dy = grad_y(u, dy)
        dv_dx = grad_x(v, dx)
        dv_dy = grad_y(v, dy)
        
        # Advection
        adv_x = u * du_dx + v * du_dy
        adv_y = u * dv_dx + v * dv_dy
        
        # Diffusion
        def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
            return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
                    jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)
        
        diff_x = nu * laplacian(u, dx, dy)
        diff_y = nu * laplacian(v, dx, dy)
        
        # SGS model
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
        
        return (-adv_x + diff_x + div_tau_x), (-adv_y + diff_y + div_tau_y)
    
    # RK3 time integration
    rhs_u1, rhs_v1 = compute_rhs(u, v)
    u2 = u + dt * rhs_u1
    v2 = v + dt * rhs_v1
    
    rhs_u2, rhs_v2 = compute_rhs(u2, v2)
    u3 = 0.75 * u + 0.25 * (u2 + dt * rhs_u2)
    v3 = 0.75 * v + 0.25 * (v2 + dt * rhs_v2)
    
    rhs_u3, rhs_v3 = compute_rhs(u3, v3)
    u_star = (1.0/3.0) * u + (2.0/3.0) * (u3 + dt * rhs_u3)
    v_star = (1.0/3.0) * v + (2.0/3.0) * (v3 + dt * rhs_v3)
    
    return u_star, v_star
