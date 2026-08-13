"""
Shan-Chen Single-Component Multiphase (SC-SCM) model for LBM
Implements two-phase flow using pseudopotential method
"""

import jax.numpy as jnp


def get_psi(rho: jnp.ndarray, psi0: float = 1.0, rho0: float = 1.0) -> jnp.ndarray:
    """
    Calculate pseudopotential field from density using Shan-Chen exponential function.
    
    Args:
        rho: Density field (nx, ny)
        psi0: Pseudopotential scaling factor
        rho0: Reference density
    
    Returns:
        psi: Pseudopotential field (nx, ny)
    """
    return psi0 * (1.0 - jnp.exp(-rho / rho0))


def compute_sc_force(psi: jnp.ndarray, G: float, weights: jnp.ndarray, 
                    ex: jnp.ndarray, ey: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Shan-Chen interparticle interaction force using jnp.roll.
    
    Args:
        psi: Pseudopotential field (nx, ny)
        G: Interaction strength (negative for attraction)
        weights: D2Q9 directional weights (9,)
        ex: Lattice velocity x components (9,)
        ey: Lattice velocity y components (9,)
    
    Returns:
        force: Force field (2, nx, ny) where force[0] is fx, force[1] is fy
    """
    fx = jnp.zeros_like(psi)
    fy = jnp.zeros_like(psi)
    
    # Sum over all 8 neighbor directions (skip i=0, the center)
    for i in range(1, 9):
        # Shift psi field in direction of lattice vector e_i
        # Note: negative shift because we want psi at x + e_i
        psi_neighbor = jnp.roll(psi, shift=(-ex[i], -ey[i]), axis=(0, 1))
        term = weights[i] * psi_neighbor
        fx += term * ex[i]
        fy += term * ey[i]
    
    # Apply the Shan-Chen force formula
    fx = -G * psi * fx
    fy = -G * psi * fy
    
    return jnp.stack([fx, fy], axis=0)


def apply_force_to_velocity(u_star: jnp.ndarray, v_star: jnp.ndarray, 
                            force: jnp.ndarray, rho: jnp.ndarray, 
                            tau: float, gravity: float = 0.0) -> tuple:
    """
    Apply Shan-Chen force to compute physical velocity.
    
    Args:
        u_star: Unforced velocity x component (nx, ny)
        v_star: Unforced velocity y component (nx, ny)
        force: Force field (2, nx, ny)
        rho: Density field (nx, ny)
        tau: Relaxation time
        gravity: Gravity acceleration (negative for downward)
    
    Returns:
        u: Physical velocity x component (nx, ny)
        v: Physical velocity y component (nx, ny)
    """
    fx = force[0]
    fy = force[1]
    
    # Add gravity force (buoyancy: F_g = rho * g)
    fy = fy + rho * gravity
    
    # Physical velocity = unforced velocity + (tau * force) / rho
    u = u_star + (tau * fx) / rho
    v = v_star + (tau * fy) / rho
    
    return u, v
