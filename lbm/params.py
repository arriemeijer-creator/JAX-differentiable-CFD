"""
LBM-specific parameters and configuration
"""

import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LBMSimulationParams:
    """Parameters specific to LBM simulation"""
    
    # Lattice parameters
    lattice_type: str = "D2Q9"  # Lattice type (D2Q9, D2Q7, etc.)
    tau: float = 0.6  # Relaxation time (must be > 0.5 for stability)
    omega: float = 1.0 / 0.6  # Collision frequency (1/tau)
    
    # Boundary conditions
    boundary_type: str = "bounce_back"  # bounce_back, specular, etc.
    inlet_type: str = "equilibrium"  # equilibrium, zou-he, etc.
    outlet_type: str = "equilibrium"  # equilibrium, open, etc.
    
    # Per-boundary configuration (left, right, top, bottom)
    bc_left: str = "inlet"  # inlet, outlet, farfield, wall
    bc_right: str = "outlet"  # inlet, outlet, farfield, wall
    bc_top: str = "farfield"  # inlet, outlet, farfield, wall
    bc_bottom: str = "farfield"  # inlet, outlet, farfield, wall
    
    # Outlet pressure values (in Pa, negative values indicate suction)
    outlet_pressure_left: float = 0.0
    outlet_pressure_right: float = 0.0
    outlet_pressure_top: float = 0.0
    outlet_pressure_bottom: float = 0.0
    
    # Collision model
    collision_model: str = "BGK"  # BGK, MRT, TRT
    use_mrt: bool = False
    vorticity_confinement: float = 0.0  # Vorticity confinement coefficient (0-1)
    
    # Force terms (optional)
    force_x: float = 0.0
    force_y: float = 0.0
    
    # Initialization
    initial_density: float = 1.0
    initial_velocity_x: float = 0.0
    initial_velocity_y: float = 0.0
    
    # Startup velocity ramp
    startup_ramp_steps: int = 100  # Number of steps to ramp up velocity
    enable_startup_ramp: bool = True  # Enable/disable velocity ramp
    
    # Two-phase flow (Shan-Chen model)
    enable_two_phase: bool = False  # Enable two-phase flow
    G: float = -5.0  # Interaction strength (negative for attraction)
    psi0: float = 1.0  # Pseudopotential scaling factor
    rho0: float = 1.0  # Reference density for pseudopotential
    gravity: float = 0.0  # Gravity acceleration (0 for no gravity, negative for downward)
    two_phase_init: str = "droplet"  # Initialization type: droplet, channel, bubble
    
    # Passive scalar (dye) for visualization
    enable_dye: bool = True
    dye_diffusivity: float = 0.01

    # Thermal Boussinesq support
    enable_thermal: bool = False
    thermal_diffusivity: float = 0.05
    thermal_expansion_coeff: float = 0.003
    thermal_gravity: float = 1.0
    thermal_reference: float = 0.0
    # Temperature settings (in °C)
    thermal_inlet_temp: float = 30.0  # Inlet temperature
    thermal_ambient_temp: float = 20.0  # Starting mean air ambient temperature
    thermal_solid_min_temp: float = 0.0  # Minimum solid temperature (for red scale R=0)
    thermal_solid_max_temp: float = 100.0  # Maximum solid temperature (for red scale R=255)
    
    # PNG mask spin (rotating obstacles)
    enable_spin: bool = False  # Enable spin for custom PNG masks
    spin_rpm: float = 0.0  # Spin rate in revolutions per minute (-100 to +100)
    spin_center_x: float = 0.5  # Spin center X (normalized 0-1)
    spin_center_y: float = 0.5  # Spin center Y (normalized 0-1)
    spin_update_interval: int = 5  # Update mask rotation every N steps to reduce JIT recompilation
    
    # LES/SGS turbulence modeling
    use_les: bool = False  # Enable LES for LBM
    les_model: str = 'dynamic_smagorinsky'  # 'dynamic_smagorinsky' or 'smagorinsky'
    smagorinsky_constant: float = 0.25  # Constant C_s for Smagorinsky model (increased for more visible effect)
    
    def __post_init__(self):
        """Update omega when tau is set"""
        self.omega = 1.0 / self.tau
    
    @property
    def viscosity(self) -> float:
        """Compute kinematic viscosity from relaxation time"""
        cs_squared = 1.0 / 3.0  # Speed of sound squared for D2Q9
        return cs_squared * (self.tau - 0.5)
    
    @viscosity.setter
    def viscosity(self, nu: float):
        """Set relaxation time from kinematic viscosity"""
        cs_squared = 1.0 / 3.0
        self.tau = 0.5 + nu / cs_squared
        self.omega = 1.0 / self.tau
