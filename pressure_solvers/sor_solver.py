import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_sor(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
               dx: float, dy: float, dt: float = 0.001, omega: float = 1.5, max_iter: int = 100, tol: float = 1e-6) -> jnp.ndarray:
    """Successive Over-Relaxation solver"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return grad_x(u, dx) + grad_y(v, dy)
    
    nx, ny = u.shape
    b = divergence(u, v, dx, dy) / dt
    
    def body_fun(carry):
        p, i = carry
        
        # Red-black ordering for better parallelism
        # Red points (i+j even)
        p_red = p.at[::2, ::2].set(
            (1-omega) * p[::2, ::2] + omega/4 * (
                jnp.roll(p, 1, axis=0)[::2, ::2] + 
                jnp.roll(p, -1, axis=0)[::2, ::2] +
                jnp.roll(p, 1, axis=1)[::2, ::2] + 
                jnp.roll(p, -1, axis=1)[::2, ::2] - 
                dx**2 * b[::2, ::2]
            )
        )
        
        # Black points (i+j odd)
        p_black = p_red.at[1::2, 1::2].set(
            (1-omega) * p_red[1::2, 1::2] + omega/4 * (
                jnp.roll(p_red, 1, axis=0)[1::2, 1::2] + 
                jnp.roll(p_red, -1, axis=0)[1::2, 1::2] +
                jnp.roll(p_red, 1, axis=1)[1::2, 1::2] + 
                jnp.roll(p_red, -1, axis=1)[1::2, 1::2] - 
                dx**2 * b[1::2, 1::2]
            )
        )
        
        return p_black, i + 1
    
    def cond_fun(carry):
        p, i = carry
        return i < max_iter
    
    p_init = jnp.zeros((nx, ny))
    p_final, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, 0))
    return p_final
