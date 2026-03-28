import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_jacobi(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
                   dx: float, dy: float, dt: float = 0.001, max_iter: int = 20) -> jnp.ndarray:
    """Original Jacobi iteration solver"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return grad_x(u, dx) + grad_y(v, dy)
    
    div = divergence(u, v, dx, dy)
    b = div / dt
    
    def body_fun(carry):
        p, i = carry
        p_new = 0.25 * (jnp.roll(p, 1, axis=0) + jnp.roll(p, -1, axis=0) + 
                        jnp.roll(p, 1, axis=1) + jnp.roll(p, -1, axis=1) - dx**2 * b)
        
        return p_new, i + 1

    def cond_fun(carry):
        p, i = carry
        return i < max_iter

    p_init = jnp.zeros((u.shape[0], u.shape[1]))
    p_final, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, 0))
    return p_final
