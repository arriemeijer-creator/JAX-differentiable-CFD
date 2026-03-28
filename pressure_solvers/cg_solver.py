import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_cg(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
               dx: float, dy: float, dt: float = 0.001, max_iter: int = 100, tol: float = 1e-6) -> jnp.ndarray:
    """Conjugate Gradient solver"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return grad_x(u, dx) + grad_y(v, dy)
    
    nx, ny = u.shape
    b = divergence(u, v, dx, dy) / dt
    
    def apply_laplacian(p: jnp.ndarray) -> jnp.ndarray:
        """Apply discrete Laplacian operator"""
        return (jnp.roll(p, 1, axis=0) + jnp.roll(p, -1, axis=0) +
                jnp.roll(p, 1, axis=1) + jnp.roll(p, -1, axis=1) - 4 * p) / dx**2
    
    # Initialize
    p = jnp.zeros((nx, ny))
    r = b - apply_laplacian(p)  # Residual
    d = r  # Search direction
    
    def body_fun(carry):
        p, r, d, i = carry
        
        # Matrix-vector product
        Ad = apply_laplacian(d)
        
        # Step size
        alpha = jnp.sum(r * r) / (jnp.sum(d * Ad) + 1e-10)
        
        # Update solution and residual
        p_new = p + alpha * d
        r_new = r - alpha * Ad
        
        # New search direction
        beta = jnp.sum(r_new * r_new) / (jnp.sum(r * r) + 1e-10)
        d_new = r_new + beta * d
        
        return p_new, r_new, d_new, i + 1
    
    def cond_fun(carry):
        p, r, d, i = carry
        return (i < max_iter) & (jnp.sqrt(jnp.sum(r**2)) > tol)
    
    p_init = jnp.zeros((nx, ny))
    r_init = b - apply_laplacian(p_init)
    d_init = r_init
    
    p_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, r_init, d_init, 0))
    return p_final
