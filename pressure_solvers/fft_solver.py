import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def poisson_fft(u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray,
               dx: float, dy: float, dt: float = 0.001) -> jnp.ndarray:
    """FFT-based Poisson solver for periodic domains"""
    def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    
    def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)
    
    def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        return grad_x(u, dx) + grad_y(v, dy)
    
    nx, ny = u.shape
    div = divergence(u, v, dx, dy)
    b = div / dt
    
    # Wave numbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Avoid division by zero
    K2 = K2.at[0, 0].set(1.0)
    
    # Solve in spectral space
    div_hat = jnp.fft.fft2(div)
    p_hat = -div_hat / K2 / dt
    
    # Set mean pressure to zero
    p_hat = p_hat.at[0, 0].set(0.0)
    
    # Transform back
    p = jnp.real(jnp.fft.ifft2(p_hat))
    return p
