import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict
import time
from dataclasses import dataclass, field

# Import adaptive dt controller
try:
    from timestepping.adaptive_dt import AdaptiveDtController, update_adaptive_dt, set_adaptive_dt
    print("Successfully imported adaptive dt controller and functions")
except ImportError as e:
    print(f"Failed to import adaptive dt controller: {e}")
    AdaptiveDtController = None
    update_adaptive_dt = None
    set_adaptive_dt = None

# Import advection schemes and utilities
try:
    from advection_schemes import (upwind_step, maccormack_step, jos_stam_step, 
                                   quick_step, weno5_step, tvd_step, rk3_step, spectral_step,
                                   AdvectionParams, check_cfl, adaptive_dt, spectral_dealias_2_3)
    print("Successfully imported advection schemes and utilities")
except ImportError as e:
    print(f"Failed to import advection schemes: {e}")
    upwind_step = maccormack_step = jos_stam_step = None
    quick_step = weno5_step = tvd_step = rk3_step = spectral_step = None
    AdvectionParams = check_cfl = adaptive_dt = spectral_dealias_2_3 = None

# Import pressure solvers
try:
    from pressure_solvers import (poisson_jacobi, poisson_fft, poisson_adi, poisson_sor, 
                                 poisson_gauss_seidel_rb, poisson_cg, poisson_multigrid)
    print("Successfully imported pressure solvers")
except ImportError as e:
    print(f"Failed to import pressure solvers: {e}")
    poisson_jacobi = poisson_fft = poisson_adi = poisson_sor = None
    poisson_gauss_seidel_rb = poisson_cg = poisson_multigrid = None

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_debug_nans', False)


@dataclass
class GridParams:
    nx: int
    ny: int
    lx: float
    ly: float
    dx: float = field(init=False)
    dy: float = field(init=False)
    x: jnp.ndarray = field(init=False)
    y: jnp.ndarray = field(init=False)
    X: jnp.ndarray = field(init=False)
    Y: jnp.ndarray = field(init=False)
    
    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = jnp.linspace(0, self.lx, self.nx)
        self.y = jnp.linspace(0, self.ly, self.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')

@dataclass
class FlowParams:
    Re: float
    U_inf: float
    nu: float = field(init=False)
    
    def __post_init__(self):
        self.nu = self.U_inf * (2.0 * 0.18) / self.Re

@dataclass
class GeometryParams:
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    radius: jnp.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'radius': self.radius
        }

@dataclass
class CavityGeometryParams:
    """Parameters for lid-driven cavity flow"""
    lid_velocity: float = 1.0  # Top wall velocity
    cavity_width: float = 1.0  # Cavity width
    cavity_height: float = 1.0  # Cavity height

@dataclass
class ChannelGeometryParams:
    """Parameters for channel flow"""
    inlet_velocity: float = 1.0  # Inlet velocity
    channel_height: float = 1.0  # Channel height
    channel_length: float = 4.0  # Channel length

@dataclass
class BackwardStepGeometryParams:
    """Parameters for backward-facing step flow"""
    inlet_velocity: float = 1.0  # Inlet velocity
    step_height: float = 0.5  # Step height
    channel_height: float = 1.0  # Channel height
    channel_length: float = 10.0  # Channel length

@dataclass
class TaylorGreenGeometryParams:
    """Parameters for Taylor-Green vortex flow"""
    domain_size: float = 2 * jnp.pi  # Domain size (2π for Taylor-Green)
    initial_amplitude: float = 1.0  # Initial vortex amplitude

@dataclass
class SimulationParams:
    Cs: float = 0.17
    eps: float = 0.05
    advection_scheme: str = 'upwind'  # 'upwind', 'maccormack', 'jos_stam', 'quick', 'weno5', 'tvd', 'rk3', 'spectral'
    limiter: str = 'minmod'  # for TVD
    weno_epsilon: float = 1e-6  # for WENO
    dealias: bool = True  # for spectral
    max_cfl: float = 0.5  # maximum CFL number
    adaptive_dt: bool = False  # enable adaptive timestepping
    fixed_dt: float = 0.001  # User-specified fixed timestep
    pressure_solver: str = 'jacobi'  # 'jacobi', 'fft', 'adi', 'sor', 'gauss_seidel_rb', 'cg', 'multigrid'
    sor_omega: float = 1.5  # SOR relaxation parameter
    pressure_max_iter: int = 100  # max iterations for iterative solvers
    flow_type: str = 'von_karman'  # 'von_karman', 'lid_driven_cavity', 'channel_flow', 'backward_step', 'taylor_green'
    dt_min: float = 1e-5  # Minimum timestep (safety)
    dt_max: float = 0.01   # Maximum timestep (stability)


@jax.jit
def grad_x(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)

@jax.jit
def grad_y(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)

@jax.jit
def laplacian(f: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return (jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0) +
            jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1) - 4 * f) / (dx**2)

@jax.jit
def divergence(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return grad_x(u, dx) + grad_y(v, dy)

@jax.jit
def vorticity(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return grad_x(v, dx) - grad_y(u, dy)


@jax.jit
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


@jax.jit
def sdf_cylinder(x: jnp.ndarray, y: jnp.ndarray, center_x: float, center_y: float, radius: float) -> jnp.ndarray:
    return jnp.sqrt((x - center_x)**2 + (y - center_y)**2) - radius

@jax.jit
def smooth_mask(phi: jnp.ndarray, eps: float = 0.05) -> jnp.ndarray:
    return jax.nn.sigmoid(phi / eps)

def create_mask_from_params(X: jnp.ndarray, Y: jnp.ndarray, params: Dict, eps: float = 0.05) -> jnp.ndarray:
    phi = sdf_cylinder(X, Y, params['center_x'], params['center_y'], params['radius'])
    return smooth_mask(phi, eps)

@jax.jit
def create_cavity_mask(X: jnp.ndarray, Y: jnp.ndarray, cavity_width: float, cavity_height: float) -> jnp.ndarray:
    """Create mask for lid-driven cavity (1 inside cavity, 0 outside)"""
    # Simple rectangular cavity - ensure it matches the grid shape
    mask = (X >= 0) & (X <= cavity_width) & (Y >= 0) & (Y <= cavity_height)
    return mask.astype(float)

@jax.jit
def apply_cavity_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, lid_velocity: float, 
                                    cavity_width: float, cavity_height: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for lid-driven cavity"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Top wall (moving lid)
    u_bc = u_bc.at[:, -1].set(lid_velocity)
    v_bc = v_bc.at[:, -1].set(0.0)
    
    # Bottom wall (no-slip)
    u_bc = u_bc.at[:, 0].set(0.0)
    v_bc = v_bc.at[:, 0].set(0.0)
    
    # Left wall (no-slip)
    u_bc = u_bc.at[0, :].set(0.0)
    v_bc = v_bc.at[0, :].set(0.0)
    
    # Right wall (no-slip)
    u_bc = u_bc.at[-1, :].set(0.0)
    v_bc = v_bc.at[-1, :].set(0.0)
    
    return u_bc, v_bc

@jax.jit
def create_channel_mask(X: jnp.ndarray, Y: jnp.ndarray, channel_height: float, channel_length: float) -> jnp.ndarray:
    """Create mask for channel flow (1 inside channel, 0 outside)"""
    mask = (X >= 0) & (X <= channel_length) & (Y >= 0) & (Y <= channel_height)
    return mask.astype(float)

def apply_channel_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, inlet_velocity: float,
                                    channel_height: float, channel_length: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for channel flow"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Inlet (left boundary) - parabolic profile
    # Use jnp.arange instead of linspace for JIT compatibility
    y_indices = jnp.arange(ny)
    y = y_indices * (channel_height / (ny - 1)) if ny > 1 else jnp.array([0.0])
    parabolic_profile = 6 * inlet_velocity * y * (channel_height - y) / (channel_height**2)
    u_bc = u_bc.at[0, :].set(parabolic_profile)
    v_bc = v_bc.at[0, :].set(0.0)
    
    # Outlet (right boundary) - zero gradient
    u_bc = u_bc.at[-1, :].set(u_bc.at[-2, :].get())
    v_bc = v_bc.at[-1, :].set(v_bc.at[-2, :].get())
    
    # Top and bottom walls (no-slip)
    u_bc = u_bc.at[:, 0].set(0.0)
    v_bc = v_bc.at[:, 0].set(0.0)
    u_bc = u_bc.at[:, -1].set(0.0)
    v_bc = v_bc.at[:, -1].set(0.0)
    
    return u_bc, v_bc

@jax.jit
def create_backward_step_mask(X: jnp.ndarray, Y: jnp.ndarray, step_height: float, 
                            channel_height: float, channel_length: float) -> jnp.ndarray:
    """Create mask for backward-facing step flow"""
    # Channel region
    channel_mask = (X >= 0) & (X <= channel_length) & (Y >= 0) & (Y <= channel_height)
    
    # Step region (remove from channel)
    step_mask = (X >= 0) & (X <= channel_length/4) & (Y >= 0) & (Y <= step_height)
    
    # Final mask: channel minus step
    mask = channel_mask & ~step_mask
    return mask.astype(float)

def apply_backward_step_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, inlet_velocity: float,
                                          step_height: float, channel_height: float, 
                                          channel_length: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for backward-facing step flow"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Inlet (left boundary) - parabolic profile above step
    # Use jnp.arange instead of linspace for JIT compatibility
    y_indices = jnp.arange(ny)
    y = y_indices * (channel_height / (ny - 1)) if ny > 1 else jnp.array([0.0])
    inlet_height = channel_height - step_height
    parabolic_profile = 6 * inlet_velocity * (y - step_height) * (channel_height - y) / (inlet_height**2)
    # Set only above step height
    inlet_mask = y >= step_height
    u_bc = u_bc.at[0, :].set(jnp.where(inlet_mask, parabolic_profile, 0.0))
    v_bc = v_bc.at[0, :].set(0.0)
    
    # Outlet (right boundary) - zero gradient
    u_bc = u_bc.at[-1, :].set(u_bc.at[-2, :].get())
    v_bc = v_bc.at[-1, :].set(v_bc.at[-2, :].get())
    
    # Top wall (no-slip)
    u_bc = u_bc.at[:, -1].set(0.0)
    v_bc = v_bc.at[:, -1].set(0.0)
    
    # Bottom wall and step (no-slip)
    u_bc = u_bc.at[:, 0].set(0.0)
    v_bc = v_bc.at[:, 0].set(0.0)
    
    # Step vertical face (no-slip)
    step_x_idx = int((channel_length/4) / (channel_length/nx)) if nx > 0 else 0
    if step_x_idx < nx:
        step_y_end = int((step_height / channel_height) * ny) if ny > 0 else 0
        u_bc = u_bc.at[step_x_idx, :step_y_end].set(0.0)
        v_bc = v_bc.at[step_x_idx, :step_y_end].set(0.0)
    
    return u_bc, v_bc

@jax.jit
def create_taylor_green_mask(X: jnp.ndarray, Y: jnp.ndarray, domain_size: float) -> jnp.ndarray:
    """Create mask for Taylor-Green vortex (full domain)"""
    # Taylor-Green is periodic, so mask is all ones
    return jnp.ones_like(X)

@jax.jit
def apply_taylor_green_boundary_conditions(u: jnp.ndarray, v: jnp.ndarray, amplitude: float,
                                         domain_size: float, nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply boundary conditions for Taylor-Green vortex (periodic)"""
    # Taylor-Green is periodic, so no boundary conditions needed
    # The flow is already periodic through the advection scheme
    return u, v


@jax.jit
def compute_forces(u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, mask: jnp.ndarray,
                   dx: float, dy: float, nu: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)
    du_dx = grad_x(u, dx)
    du_dy = grad_y(u, dy)
    dv_dx = grad_x(v, dx)
    dv_dy = grad_y(v, dy)
    sigma_xx = -p + 2.0 * nu * du_dx
    sigma_yy = -p + 2.0 * nu * dv_dy
    sigma_xy = nu * (du_dy + dv_dx)
    dm_dx = grad_x(mask, dx)
    dm_dy = grad_y(mask, dy)
    boundary_norm = jnp.sqrt(dm_dx**2 + dm_dy**2)
    drag = jnp.sum((sigma_xx * dm_dx + sigma_xy * dm_dy) * boundary_norm) * dx * dy
    lift = jnp.sum((sigma_xy * dm_dx + sigma_yy * dm_dy) * boundary_norm) * dx * dy
    
    return drag, lift


class BaselineSolver:
    
    def __init__(self,
                 grid: GridParams,
                 flow: FlowParams,
                 geom: GeometryParams,
                 sim_params: SimulationParams,
                 dt: float = None,  # Optional: if None, use sim_params.fixed_dt
                 seed: int = 42):
        
        self.grid = grid
        self.flow = flow
        self.geom = geom
        self.sim_params = sim_params
        
        # Initialize adaptive dt controller
        if AdaptiveDtController is not None:
            self.adaptive_controller = AdaptiveDtController(
                flow_type=sim_params.flow_type,
                dt_min=sim_params.dt_min,
                dt_max=sim_params.dt_max
            )
        else:
            self.adaptive_controller = None
        
        # Smart dt initialization
        if dt is not None:
            # User passed specific dt (overrides everything)
            self.dt = dt
            self.sim_params.fixed_dt = dt
            self.sim_params.adaptive_dt = False
            print(f"Using user-specified fixed dt = {dt:.6f}")
        elif self.sim_params.adaptive_dt:
            # Adaptive mode - start with estimated safe dt
            if self.adaptive_controller is not None:
                self.dt = self.adaptive_controller.get_initial_dt(
                    self.flow.U_inf, self.grid.dx, self.grid.dy
                )
            else:
                self.dt = self._estimate_safe_dt()
            print(f"Using adaptive timestepping, starting dt = {self.dt:.6f}")
        else:
            # Fixed mode from params
            self.dt = self.sim_params.fixed_dt
            print(f"Using fixed dt = {self.dt:.6f} from simulation parameters")
        
        # Pre-compute mask once for performance
        self.mask = self._compute_mask()
        
        # Initialize based on flow type
        if self.sim_params.flow_type == 'lid_driven_cavity':
            self._initialize_cavity_flow()
        elif self.sim_params.flow_type == 'channel_flow':
            self._initialize_channel_flow()
        elif self.sim_params.flow_type == 'backward_step':
            self._initialize_backward_step_flow()
        elif self.sim_params.flow_type == 'taylor_green':
            self._initialize_taylor_green_flow()
        else:  # von_karman
            self._initialize_von_karman_flow()
        
        self._step_jit = jax.jit(self._step)
        self._vorticity = jax.jit(vorticity, static_argnums=(2, 3))
        self._divergence = jax.jit(divergence, static_argnums=(2, 3))
        
        self.history = {'time': [], 'ke': [], 'enstrophy': [], 'drag': [], 'lift': [], 'dt': []}
        self.iteration = 0
    
    def _initialize_von_karman_flow(self):
        """Initialize von Kármán vortex shedding flow"""
        self.u = jnp.ones((self.grid.nx, self.grid.ny)) * self.flow.U_inf
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        self._add_initial_perturbation()
    
    def _compute_mask(self):
        """Compute mask once based on flow type"""
        if self.sim_params.flow_type == 'lid_driven_cavity':
            # For cavity flow, use simple rectangular mask
            return create_cavity_mask(self.grid.X, self.grid.Y, 1.0, 1.0)
        elif self.sim_params.flow_type == 'channel_flow':
            # For channel flow, use rectangular mask
            return create_channel_mask(self.grid.X, self.grid.Y, 1.0, 4.0)
        elif self.sim_params.flow_type == 'backward_step':
            # For backward step flow, use step mask
            return create_backward_step_mask(self.grid.X, self.grid.Y, 0.5, 1.0, 10.0)
        elif self.sim_params.flow_type == 'taylor_green':
            # For Taylor-Green, full domain mask
            return create_taylor_green_mask(self.grid.X, self.grid.Y, 2*jnp.pi)
        else:  # von_karman
            geom_dict = self.geom.to_dict()
            return create_mask_from_params(self.grid.X, self.grid.Y, geom_dict, self.sim_params.eps)
    
    def _initialize_cavity_flow(self):
        """Initialize lid-driven cavity flow"""
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        # Apply lid boundary condition
        self.u = self.u.at[:, -1].set(self.flow.U_inf)  # Top wall velocity
    
    def _initialize_channel_flow(self):
        """Initialize channel flow"""
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        # Apply parabolic inlet profile
        y_indices = jnp.arange(self.grid.ny)
        y = y_indices * (1.0 / (self.grid.ny - 1)) if self.grid.ny > 1 else jnp.array([0.0])
        parabolic_profile = 6 * self.flow.U_inf * y * (1.0 - y)  # Normalized channel height
        self.u = self.u.at[0, :].set(parabolic_profile)
    
    def _initialize_backward_step_flow(self):
        """Initialize backward-facing step flow"""
        self.u = jnp.zeros((self.grid.nx, self.grid.ny))
        self.v = jnp.zeros((self.grid.nx, self.grid.ny))
        # Apply parabolic inlet profile above step
        y_indices = jnp.arange(self.grid.ny)
        y = y_indices * (1.0 / (self.grid.ny - 1)) if self.grid.ny > 1 else jnp.array([0.0])
        inlet_height = 0.5  # 1.0 - 0.5
        parabolic_profile = 6 * self.flow.U_inf * (y - 0.5) * (1.0 - y) / (inlet_height**2)
        # Set only above step height
        inlet_mask = y >= 0.5
        self.u = self.u.at[0, :].set(jnp.where(inlet_mask, parabolic_profile, 0.0))
    
    def _initialize_taylor_green_flow(self):
        """Initialize Taylor-Green vortex"""
        X, Y = self.grid.X, self.grid.Y
        # Taylor-Green vortex initial conditions
        self.u = self.flow.U_inf * jnp.sin(X) * jnp.cos(Y)
        self.v = -self.flow.U_inf * jnp.cos(X) * jnp.sin(Y)
    
    def _check_stability(self) -> bool:
        """Check if simulation is numerically stable"""
        if self.adaptive_controller is not None:
            return self.adaptive_controller.check_stability(self.u, self.v, self.flow.U_inf)
        else:
            # Fallback stability check
            try:
                # Check for NaN or Inf in velocity fields
                u_finite = jnp.all(jnp.isfinite(self.u))
                v_finite = jnp.all(jnp.isfinite(self.v))
                
                if not u_finite or not v_finite:
                    return False
                
                # Check for velocity explosion (should be < 100x U_inf)
                max_vel = float(jnp.max(jnp.sqrt(self.u**2 + self.v**2)))
                if max_vel > 100 * self.flow.U_inf:
                    return False
                
                return True
            except:
                return False
    
    def _reset_flow(self):
        """Reset flow to initial condition when instability detected"""
        # Reinitialize based on flow type
        if self.sim_params.flow_type == 'lid_driven_cavity':
            self._initialize_cavity_flow()
        elif self.sim_params.flow_type == 'channel_flow':
            self._initialize_channel_flow()
        elif self.sim_params.flow_type == 'backward_step':
            self._initialize_backward_step_flow()
        elif self.sim_params.flow_type == 'taylor_green':
            self._initialize_taylor_green_flow()
        else:  # von_karman
            self._initialize_von_karman_flow()
        
        # Reset iteration counter but keep dt at safe value
        self.iteration = 0
        
        # Reset dt to safer value if adaptive
        if self.sim_params.adaptive_dt:
            if self.adaptive_controller is not None:
                self.dt = self.adaptive_controller.get_initial_dt(
                    self.flow.U_inf, self.grid.dx, self.grid.dy
                )
            else:
                raise ValueError("AdaptiveDtController is not available")
            self._step_jit = jax.jit(self._step)
        
        print(f"Flow reset with safe dt = {self.dt:.6f}")
    
    def check_cfl_condition(self) -> float:
        """Check current CFL condition"""
        if check_cfl is not None:
            return check_cfl(self.u, self.v, self.dt, self.grid.dx, self.grid.dy)
        else:
            # Fallback CFL calculation
            max_vel = jnp.max(jnp.sqrt(self.u**2 + self.v**2))
            cfl_x = max_vel * self.dt / self.grid.dx
            cfl_y = max_vel * self.dt / self.grid.dy
            return max(cfl_x, cfl_y)
    
    
    def apply_flow_type(self, flow_type: str):
        valid_flow_types = ['von_karman', 'lid_driven_cavity', 'channel_flow', 'backward_step', 'taylor_green']
        if flow_type not in valid_flow_types:
            raise ValueError(f"Flow type must be one of {valid_flow_types}")
        
        old_flow_type = self.sim_params.flow_type
        self.sim_params.flow_type = flow_type
        
        # Update grid parameters for different flow types
        if flow_type == 'lid_driven_cavity':
            # Square cavity domain
            self.grid = GridParams(nx=128, ny=128, lx=1.0, ly=1.0)
        elif flow_type == 'channel_flow':
            # Rectangular channel domain
            self.grid = GridParams(nx=256, ny=64, lx=4.0, ly=1.0)
        elif flow_type == 'backward_step':
            # Longer domain for step flow
            self.grid = GridParams(nx=512, ny=128, lx=10.0, ly=1.0)
        elif flow_type == 'taylor_green':
            # Square domain for Taylor-Green (2π x 2π)
            self.grid = GridParams(nx=128, ny=128, lx=2*jnp.pi, ly=2*jnp.pi)
        else:  # von_karman
            # Original von Kármán domain
            self.grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)
        
        # Recreate grid coordinates
        x = jnp.linspace(0, self.grid.lx, self.grid.nx)
        y = jnp.linspace(0, self.grid.ly, self.grid.ny)
        self.grid.X, self.grid.Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Reinitialize the flow
        if flow_type == 'lid_driven_cavity':
            self._initialize_cavity_flow()
        elif flow_type == 'channel_flow':
            self._initialize_channel_flow()
        elif flow_type == 'backward_step':
            self._initialize_backward_step_flow()
        elif flow_type == 'taylor_green':
            self._initialize_taylor_green_flow()
        else:  # von_karman
            self._initialize_von_karman_flow()
        
        # Recompile JIT functions
        self._step_jit = jax.jit(self._step)
        
        # Reset history with all required keys including 'dt'
        self.history = {'time': [], 'ke': [], 'enstrophy': [], 'drag': [], 'lift': [], 'dt': []}
        self.iteration = 0
        
        print(f"Flow type changed to {flow_type}")
        print(f"Grid updated to {self.grid.nx}x{self.grid.ny} ({self.grid.lx}x{self.grid.ly})")
        print(f"Flow reinitialized and JIT functions recompiled")
    
    def set_fixed_dt(self, dt: float):
        """Switch to fixed timestep mode"""
        self.sim_params.adaptive_dt = False
        self.sim_params.fixed_dt = dt
        self.dt = dt
        self._step_jit = jax.jit(self._step)
        print(f"Switched to fixed dt = {dt:.6f}")

    def update_adaptive_dt(self):
        """Robust adaptive timestep update with explosion prevention"""
        if update_adaptive_dt is not None:
            update_adaptive_dt(self)
        else:
            print("Warning: update_adaptive_dt function not available")

    def set_adaptive_dt(self, max_cfl: float = None, dt_min: float = None, dt_max: float = None):
        """Switch to adaptive timestep mode"""
        if set_adaptive_dt is not None:
            set_adaptive_dt(self, max_cfl, dt_min, dt_max)
        else:
            print("Warning: set_adaptive_dt function not available")

    
    def get_dt_info(self) -> dict:
        """Get current dt information"""
        current_cfl = self.check_cfl_condition()
        return {
            'dt': self.dt,
            'adaptive': self.sim_params.adaptive_dt,
            'cfl': float(current_cfl),
            'max_cfl': self.sim_params.max_cfl,
            'dt_range': (self.sim_params.dt_min, self.sim_params.dt_max)
        }
    
    def get_recommended_dt(self, flow_type: str) -> dict:
        """Get recommended dt settings for different flow types"""
        recommendations = {
            'von_karman': {
                'fixed_dt': 0.001,
                'adaptive_dt': True,
                'max_cfl': 0.5,
                'dt_range': (1e-5, 0.002)
            },
            'lid_driven_cavity': {
                'fixed_dt': 0.001,
                'adaptive_dt': True,
                'max_cfl': 0.6,
                'dt_range': (1e-5, 0.005)
            },
            'channel_flow': {
                'fixed_dt': 0.001,
                'adaptive_dt': True,
                'max_cfl': 0.8,
                'dt_range': (1e-5, 0.005)
            },
            'backward_step': {
                'fixed_dt': 0.0005,
                'adaptive_dt': True,
                'max_cfl': 0.4,
                'dt_range': (1e-6, 0.001)
            },
            'taylor_green': {
                'fixed_dt': 0.01,
                'adaptive_dt': False,  # Periodic flows can use larger fixed dt
                'max_cfl': None,
                'dt_range': (0.001, 0.02)
            }
        }
        return recommendations.get(flow_type, recommendations['von_karman'])
    
    def apply_pressure_solver(self, solver: str):
        valid_solvers = ['jacobi', 'fft', 'adi', 'sor', 'gauss_seidel_rb', 'cg', 'multigrid']
        if solver not in valid_solvers:
            raise ValueError(f"Pressure solver must be one of {valid_solvers}")
        
        # Check if solver is available
        solver_func = globals().get(f'poisson_{solver}')
        if solver_func is None:
            print(f"Warning: {solver} pressure solver not available, falling back to jacobi")
            solver = 'jacobi'
        
        self.sim_params.pressure_solver = solver
        self._step_jit = jax.jit(self._step)
        print(f"Pressure solver changed to {solver}")
        print(f"JIT functions recompiled with new pressure solver")
    
    def apply_advection_scheme(self, scheme: str):
        valid_schemes = ['upwind', 'maccormack', 'jos_stam', 'quick', 'weno5', 'tvd', 'rk3', 'spectral']
        if scheme not in valid_schemes:
            raise ValueError(f"Scheme must be one of {valid_schemes}")
        
        # Check if scheme is available
        if scheme in ['quick', 'weno5', 'tvd', 'rk3', 'spectral']:
            scheme_func = globals().get(f'{scheme}_step')
            if scheme_func is None:
                print(f"Warning: {scheme} scheme not available, falling back to upwind")
                scheme = 'upwind'
        
        self.sim_params.advection_scheme = scheme
        self._step_jit = jax.jit(self._step)
        print(f"Advection scheme changed to {scheme}")
        print(f"JIT functions recompiled with new scheme")
    
    def _add_initial_perturbation(self):
        X, Y = self.grid.X, self.grid.Y
        cylinder_x = float(self.geom.center_x)
        cylinder_radius = float(self.geom.radius)
        perturbation = 0.05 * jnp.sin(2 * jnp.pi * Y / self.grid.ly) * \
                       jnp.exp(-((X - cylinder_x - 2*cylinder_radius)**2) / (2 * cylinder_radius**2))
        self.u = self.u + perturbation
        
    def _step(self, u, v, dt):
        dx, dy = self.grid.dx, self.grid.dy
        
        # Use pre-computed mask (much faster!)
        mask = self.mask
        
        if self.sim_params.advection_scheme == 'maccormack' and maccormack_step is not None:
            u_star, v_star = maccormack_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
        elif self.sim_params.advection_scheme == 'jos_stam' and jos_stam_step is not None:
            u_star, v_star = jos_stam_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
        elif self.sim_params.advection_scheme == 'upwind' and upwind_step is not None:
            u_star, v_star = upwind_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
        elif self.sim_params.advection_scheme == 'quick' and quick_step is not None:
            u_star, v_star = quick_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
        elif self.sim_params.advection_scheme == 'weno5' and weno5_step is not None:
            u_star, v_star = weno5_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs, self.sim_params.weno_epsilon)
        elif self.sim_params.advection_scheme == 'tvd' and tvd_step is not None:
            u_star, v_star = tvd_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs, self.sim_params.limiter)
        elif self.sim_params.advection_scheme == 'rk3' and rk3_step is not None:
            u_star, v_star = rk3_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
        elif self.sim_params.advection_scheme == 'spectral' and spectral_step is not None:
            u_star, v_star = spectral_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs, self.sim_params.dealias)
        else:  # fallback to upwind if scheme not available
            if upwind_step is not None:
                u_star, v_star = upwind_step(u, v, dt, self.flow.nu, dx, dy, mask, self.sim_params.Cs)
            else:
                # Fallback to basic upwind implementation
                du_dx = grad_x(u, dx)
                du_dy = grad_y(u, dy)
                dv_dx = grad_x(v, dx)
                dv_dy = grad_y(v, dy)
                adv_x = u * du_dx + v * du_dy
                adv_y = u * dv_dx + v * dv_dy
                diff_x = self.flow.nu * laplacian(u, dx, dy)
                diff_y = self.flow.nu * laplacian(v, dx, dy)
                u_star = u + dt * (-adv_x + diff_x)
                v_star = v + dt * (-adv_y + diff_y)
        
        div_star = divergence(u_star, v_star, dx, dy)
        
        # Use selected pressure solver
        if self.sim_params.pressure_solver == 'jacobi' and poisson_jacobi is not None:
            p = poisson_jacobi(u_star, v_star, mask, dx, dy, dt, max_iter=self.sim_params.pressure_max_iter)
        elif self.sim_params.pressure_solver == 'fft' and poisson_fft is not None:
            p = poisson_fft(u_star, v_star, mask, dx, dy, dt)
        elif self.sim_params.pressure_solver == 'adi' and poisson_adi is not None:
            p = poisson_adi(u_star, v_star, mask, dx, dy, dt, max_iter=self.sim_params.pressure_max_iter)
        elif self.sim_params.pressure_solver == 'sor' and poisson_sor is not None:
            p = poisson_sor(u_star, v_star, mask, dx, dy, dt, omega=self.sim_params.sor_omega, max_iter=self.sim_params.pressure_max_iter)
        elif self.sim_params.pressure_solver == 'gauss_seidel_rb' and poisson_gauss_seidel_rb is not None:
            p = poisson_gauss_seidel_rb(u_star, v_star, mask, dx, dy, dt, max_iter=self.sim_params.pressure_max_iter)
        elif self.sim_params.pressure_solver == 'cg' and poisson_cg is not None:
            p = poisson_cg(u_star, v_star, mask, dx, dy, dt, max_iter=self.sim_params.pressure_max_iter)
        elif self.sim_params.pressure_solver == 'multigrid' and poisson_multigrid is not None:
            p = poisson_multigrid(u_star, v_star, mask, dx, dy, dt)
        else:  # fallback to basic Jacobi
            div = divergence(u_star, v_star, dx, dy)
            b = div / dt
            
            def body_fun(carry):
                p, i = carry
                p_new = 0.25 * (jnp.roll(p, 1, axis=0) + jnp.roll(p, -1, axis=0) + 
                                jnp.roll(p, 1, axis=1) + jnp.roll(p, -1, axis=1) - dx**2 * b)
                return p_new, i + 1

            def cond_fun(carry):
                p, i = carry
                return i < self.sim_params.pressure_max_iter

            p_init = jnp.zeros((u_star.shape[0], u_star.shape[1]))
            p, _ = jax.lax.while_loop(cond_fun, body_fun, (p_init, 0))
        
        dp_dx, dp_dy = grad_x(p, dx), grad_y(p, dy)
        u_corr = u_star - self.dt * dp_dx
        v_corr = v_star - self.dt * dp_dy
        
        # Apply different boundary conditions based on flow type
        if self.sim_params.flow_type == 'lid_driven_cavity':
            u_corr, v_corr = apply_cavity_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 1.0, 1.0, self.grid.nx, self.grid.ny)
        elif self.sim_params.flow_type == 'channel_flow':
            u_corr, v_corr = apply_channel_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 1.0, 4.0, self.grid.nx, self.grid.ny)
        elif self.sim_params.flow_type == 'backward_step':
            u_corr, v_corr = apply_backward_step_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 0.5, 1.0, 10.0, self.grid.nx, self.grid.ny)
        elif self.sim_params.flow_type == 'taylor_green':
            u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, self.flow.U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
        else:  # von_karman
            u_corr = u_corr * mask
            v_corr = v_corr * mask
            u_corr = u_corr.at[0, :].set(1.0)
            v_corr = v_corr.at[0, :].set(0.0)
            u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
            v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
            v_corr = v_corr.at[:, 0].set(0.0)
            v_corr = v_corr.at[:, -1].set(0.0)
        
        return u_corr, v_corr, mask, p
    
    def step(self, compute_vorticity=True, compute_energy=True, compute_drag_lift=True):
        self.u, self.v, mask, self.current_pressure = self._step_jit(self.u, self.v, self.dt)
        self.iteration += 1
        
        # Check for numerical instability
        if not self._check_stability():
            print("WARNING: Numerical instability detected! Resetting flow...")
            self._reset_flow()
        
        # Track dt in history
        self.history['dt'].append(self.dt)
        
        # Check CFL and update adaptive timestep if enabled
        if self.sim_params.adaptive_dt and self.adaptive_controller is not None:
            new_dt = self.adaptive_controller.update_adaptive_dt(
                self.u, self.v, self.dt, self.grid.dx, self.grid.dy, 
                self.flow.U_inf, self.sim_params.max_cfl
            )
            if new_dt != self.dt:
                old_dt = self.dt
                self.dt = float(new_dt)
                # No recompilation needed - dt is now a dynamic parameter
                print(f"Adaptive dt updated: {old_dt:.6f} -> {self.dt:.6f}")
        if compute_vorticity:
            vort = self._vorticity(self.u, self.v, self.grid.dx, self.grid.dy)
        else:
            vort = jnp.zeros_like(self.u)
        
        # Only compute energy if requested
        if compute_energy:
            ke = 0.5 * jnp.sum(mask * (self.u**2 + self.v**2)) * self.grid.dx * self.grid.dy
            enst = 0.5 * jnp.sum(mask * vort**2) * self.grid.dx * self.grid.dy if compute_vorticity else 0.0
        else:
            ke = 0.0
            enst = 0.0
        
        # Only compute pressure and forces if requested
        if compute_drag_lift:
            # Use cached pressure from _step (no recomputation!)
            drag, lift = compute_forces(self.u, self.v, self.current_pressure, mask, self.grid.dx, self.grid.dy, self.flow.nu)
        else:
            drag = 0.0
            lift = 0.0
        
        # Only update history if quantities are computed
        if compute_energy:
            self.history['time'].append(self.iteration * self.dt)
            self.history['ke'].append(float(ke))
            self.history['enstrophy'].append(float(enst))
        
        if compute_drag_lift:
            if not compute_energy:  # Add time if energy wasn't computed
                self.history['time'].append(self.iteration * self.dt)
            self.history['drag'].append(float(drag))
            self.history['lift'].append(float(lift))
        
        return self.u, self.v, vort, ke, enst, drag, lift
    
    def get_cached_pressure(self):
        """Get the pressure field computed during the last step"""
        return self.current_pressure
    
    def run_interactive(self):
        """Run with interactive dt control"""
        print("\nInteractive Mode - dt Controls:")
        print("  'a' - Toggle adaptive mode")
        print("  'f X' - Set fixed dt to X")
        print("  'c' - Show current dt info")
        print("  'q' - Quit")
        
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'a':
                if self.sim_params.adaptive_dt:
                    self.set_fixed_dt(self.dt)
                else:
                    self.set_adaptive_dt()
            
            elif cmd.startswith('f'):
                try:
                    new_dt = float(cmd.split()[1])
                    self.set_fixed_dt(new_dt)
                except:
                    print("Usage: f 0.001")
            
            elif cmd == 'c':
                info = self.get_dt_info()
                print(f"dt: {info['dt']:.6f}, Adaptive: {info['adaptive']}, CFL: {info['cfl']:.3f}")
            
            elif cmd == 'q':
                break
            
            # Run one step
            self.step()
    
    def run_simulation(self, n_steps: int = 10000, verbose: bool = True):
        print(f"\n=== Running Baseline Simulation ({n_steps} steps) ===")
        t0 = time.time()
        
        for step in range(n_steps):
            u, v, vort, ke, enst, drag, lift = self.step()
            
            if verbose and step % 500 == 0:
                elapsed = time.time() - t0
                speed = step / elapsed if elapsed > 0 else 0
                print(f"Step {step:6d}, Time={step*self.dt:.3f}, "
                      f"KE={ke:.6f}, Drag={drag:.4f}, Lift={lift:.4f}, "
                      f"Speed={speed:.1f} steps/sec")
        
        elapsed = time.time() - t0
        print(f"\nSimulation completed: {n_steps} steps in {elapsed:.1f}s "
              f"({n_steps/elapsed:.1f} steps/sec)")
        
        return u, v


def main():
    print("=" * 70)
    print("Clean Baseline Navier-Stokes Solver")
    print("=" * 70)
    
    grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)
    flow = FlowParams(Re=150.0, U_inf=1.0)
    geom = GeometryParams(center_x=jnp.array(2.5), center_y=jnp.array(2.25), radius=jnp.array(0.18))
    egce = EGCEParams(Cs=0.17, eps=0.05)
    
    print(f"\nConfiguration:")
    print(f"  Grid: {grid.nx} × {grid.ny}")
    print(f"  Domain: {grid.lx} × {grid.ly}")
    print(f"  Re = {flow.Re:.1f}, U_inf = {flow.U_inf}")
    print(f"  dt = {0.001}, ν = {flow.nu:.6f}")
    print(f"  Cylinder: center=({float(geom.center_x):.1f}, {float(geom.center_y):.1f}), "
          f"radius={float(geom.radius):.3f}")
    
    solver = BaselineSolver(grid, flow, geom, egce, dt=0.001)
    
    u, v = solver.run_simulation(n_steps=20000, verbose=True)
    
    print("\n" + "=" * 70)
    print("Final Results:")
    print("=" * 70)
    print(f"Final drag coefficient: {solver.history['drag'][-1]:.4f}")
    print(f"Final lift coefficient: {solver.history['lift'][-1]:.4f}")
    print(f"Kinetic energy: {solver.history['ke'][-1]:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
