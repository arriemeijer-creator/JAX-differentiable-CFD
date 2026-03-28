import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_gauss_seidel_rb(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                           dx: float, dy: float, dt: float = 0.001, max_iter: int = 100) -> jnp.ndarray:
    """Red-black Gauss-Seidel for better parallelism"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return grad_x(u, dx) + grad_y(v, dy)
    
    nx, ny = u.shape
    b = divergence(u, v, dx, dy) / dt
    
    def red_update(p: jnp.ndarray) -> jnp.ndarray:
        """Update red points (i+j even)"""
        return p.at[::2, ::2].set(
            (jnp.roll(p, 1, axis=0)[::2, ::2] + 
             jnp.roll(p, -1, axis=0)[::2, ::2] +
             jnp.roll(p, 1, axis=1)[::2, ::2] + 
             jnp.roll(p, -1, axis=1)[::2, ::2] - 
             dx**2 * b[::2, ::2]) / 4.0
        )
    
    def black_update(p: jnp.ndarray) -> jnp.ndarray:
        """Update black points (i+j odd)"""
        return p.at[1::2, 1::2].set(
            (jnp.roll(p, 1, axis=0)[1::2, 1::2] + 
             jnp.roll(p, -1, axis=0)[1::2, 1::2] +
             jnp.roll(p, 1, axis=1)[1::2, 1::2] + 
             jnp.roll(p, -1, axis=1)[1::2, 1::2] - 
             dx**2 * b[1::2, 1::2]) / 4.0
        )
    
    def body_fun(carry):
        p, i = carry
        p = red_update(p)
        p = black_update(p)
        return p, i + 1
    
    def cond_fun(carry):
        p, i = carry
        return i < max_iter
    
    p_init = jnp.zeros((nx, ny))
    p_final, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, 0))
    return p_final
