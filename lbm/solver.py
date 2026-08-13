"""
LBM Solver class with interface compatible with BaselineSolver
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple
from solver.params import GridParams, FlowParams, GeometryParams, SimulationParams
from .params import LBMSimulationParams
from .lattice import D2Q9Lattice
from .collision import collision_step, equilibrium
from .streaming import streaming_step
from .boundary import apply_boundary_conditions, apply_inlet_outlet, apply_bounce_back
from .operators import macroscopic_variables, compute_pressure
from scipy.ndimage import rotate


class LBMSolver:
    """Lattice Boltzmann Method solver with BaselineSolver-compatible interface"""
    
    def __init__(self,
                 grid: GridParams,
                 flow: FlowParams,
                 geom: GeometryParams,
                 sim_params: SimulationParams,
                 dt: float = None,
                 seed: int = 42):
        
        # Store parameters
        self.grid = grid
        self.flow = flow
        self.geom = geom
        self.sim_params = sim_params
        self.seed = seed
        
        # Initialize LBM-specific parameters
        self.lbm_params = LBMSimulationParams()
        
        # For LBM stability, we need to ensure:
        # 1. tau > 0.5 (stability condition)
        # 2. Mach number < 0.1 (incompressibility)
        # 3. Velocity in lattice units should be small
        
        # Better mapping from physical to lattice units to preserve Reynolds number
        cs_squared = 1.0 / 3.0
        
        # Choose target lattice Reynolds number for aggressive diffusion reduction
        # Much higher Re_lattice = much less diffusion, push stability limits
        # For vortex shedding, aim for Re_lattice > 1000 for strong vortices
        target_Re_lattice = min(5000.0, flow.Re * 10.0)  # Much more aggressive scaling
        
        # Calculate lattice velocity based on physical velocity
        # Scale physical velocity to lattice units while maintaining stability
        physical_velocity = flow.U_inf
        
        # Calculate characteristic length scale (chord length or obstacle diameter)
        if hasattr(sim_params, 'naca_chord'):
            L_char = sim_params.naca_chord
        else:
            L_char = geom.radius * 2  # Diameter for cylinder
        
        # Calculate lattice velocity scaling with aggressive approach for less diffusion
        dx = grid.lx / grid.nx
        
        # Target much higher lattice velocity for less diffusion
        # Push to higher Mach number for better flow physics
        target_U_lattice = 0.25 * jnp.sqrt(cs_squared)  # Mach ~0.25
        
        # Scale physical velocity relative to a reference (e.g., 2.0 m/s)
        reference_velocity = 2.0  # Reference physical velocity
        velocity_scale_factor = physical_velocity / reference_velocity
        
        # Apply scaling to target lattice velocity
        self.U_lattice = target_U_lattice * velocity_scale_factor
        
        # Apply more relaxed stability constraints (Mach < 0.35 for aggressive flow)
        max_U_lattice = 0.35 * jnp.sqrt(cs_squared)
        min_U_lattice = 0.08  # Higher minimum for better flow
        self.U_lattice = jnp.clip(self.U_lattice, min_U_lattice, max_U_lattice)
        
        # Use grid characteristic length as L_lattice
        L_lattice = min(grid.nx, grid.ny)
        
        # Calculate required lattice viscosity
        nu_lattice = self.U_lattice * L_lattice / target_Re_lattice
        
        # Convert to tau
        tau_target = 0.5 + nu_lattice / cs_squared
        
        # Ensure stability bounds
        tau_min = 0.5  # Theoretical stability limit
        tau_max = 2.0   # Upper bound for stability
        self.lbm_params.tau = jnp.clip(tau_target, tau_min, tau_max)
        self.lbm_params.omega = 1.0 / self.lbm_params.tau
        
        # Initialize lattice
        self.lattice = D2Q9Lattice()
        
        # MRT matrix and inverse (orthogonal moment basis for D2Q9)
        self.M = jnp.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1],
            [-4, -1, -1, -1, -1,  2,  2,  2,  2],
            [4, -2, -2, -2, -2,  1,  1,  1,  1],
            [0,  1,  0, -1,  0,  1, -1, -1,  1],
            [0, -2,  0,  2,  0,  1, -1, -1,  1],
            [0,  0,  1,  0, -1,  1,  1, -1, -1],
            [0,  0, -2,  0,  2,  1,  1, -1, -1],
            [0,  1, -1,  1, -1,  0,  0,  0,  0],
            [0,  0,  0,  0,  0,  1, -1,  1, -1]
        ], dtype=jnp.float32)
        
        self.M_inv = jnp.linalg.inv(self.M)
        
        # Relaxation rates: balanced for stability without excessive diffusion
        s_visc = self.lbm_params.omega
        # Use standard values closer to 1.0 for bulk/energy to reduce artificial diffusion
        s_bulk = jnp.clip(1.0, 0.5, 1.9)
        s_energy = jnp.clip(1.0, 0.5, 1.9)  # Energy modes
        s_shear = s_visc  # Shear modes (viscous)
        
        self.s = jnp.array([
            s_bulk, s_energy, s_energy,  # conserved + energy modes
            s_shear, s_shear,           # momentum fluxes
            s_shear, s_shear,           # stress modes
            s_shear, s_shear            # higher-order moments
        ])
        # Clip for stability: all rates in (0,2)
        self.s = jnp.clip(self.s, 0.01, 1.99)
        
        # Initialize timestep (LBM typically uses dt = 1 in lattice units)
        if dt is not None:
            self.dt = dt
        else:
            # For LBM, we can use dt = 1 (lattice units) or scale to physical units
            # For compatibility with NS, we'll use the same dt
            self.dt = sim_params.fixed_dt if sim_params.fixed_dt else 0.001
        
        # Initialize distribution function f
        # f has shape (9, nx, ny)
        nx, ny = grid.nx, grid.ny
        self.f = jnp.zeros((9, nx, ny))
        
        # Initialize macroscopic variables
        self.u = jnp.zeros((nx, ny))
        self.v = jnp.zeros((nx, ny))
        self.rho = jnp.ones((nx, ny))  # Density
        self.p = jnp.zeros((nx, ny))   # Pressure (computed from density)
        
        # Initialize previous velocity (for metrics)
        self.u_prev = jnp.zeros((nx, ny))
        self.v_prev = jnp.zeros((nx, ny))
        
        # Initialize dye/scalar field
        self.c = jnp.zeros((nx, ny))
        # Initialize thermal field
        self.T = jnp.zeros((nx, ny))
        self.mask = self._compute_mask()
        
        # Store original custom mask for rotation (if spin is enabled)
        self.original_custom_mask = None
        if hasattr(self.sim_params, 'custom_mask') and self.sim_params.custom_mask is not None:
            self.original_custom_mask = np.array(self.sim_params.custom_mask).copy()
        
        # Spin state
        self.spin_angle = 0.0  # Current rotation angle in degrees
        self.spin_step_counter = 0  # Counter for spin update interval
        
        # Initialize flow based on flow type
        self._initialize_flow()
        
        # Initialize distribution function from equilibrium
        from .collision import equilibrium
        self.f = equilibrium(self.rho, self.u, self.v, 
                            self.lattice.get_cx(), self.lattice.get_cy(),
                            self.lattice.w, self.lattice.cs_squared)
        
        # JIT cache
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        
        # History for metrics
        self.history = {
            'time': [], 'dt': [], 'drag': [], 'lift': [],
            'l2_change': [], 'rms_change': [], 'l2_change_u': [], 'l2_change_v': [],
            'max_change': [], 'change_99p': [], 'rel_change': [],
            'rms_divergence': [], 'l2_divergence': [],
            'airfoil_metrics': {'CL': [], 'CD': [], 'stagnation_x': [], 'separation_x': [], 
                               'Cp_min': [], 'wake_deficit': [], 'strouhal': [], 'time': []}
        }
        
        self.iteration = 0
        self.compute_airfoil_metrics = False
        self.metrics_frame_skip = 100
        
        # Velocity ramp for gradual startup (reduces initial ringing)
        self.startup_iterations = 20  # Number of iterations for ramp
        self.current_u_inlet = 0.0  # Will be ramped up
        
        # Store current pressure for compatibility
        self.current_pressure = self.p
        
        print(f"LBM Solver initialized: tau={self.lbm_params.tau:.4f}, omega={self.lbm_params.omega:.4f}, "
              f"nu={flow.nu:.6f}, Re={flow.Re:.1f}")
    
    def _initialize_flow(self):
        """Initialize flow based on flow type with smoother profiles to reduce ringing"""
        nx, ny = self.grid.nx, self.grid.ny
        U_lattice = self.U_lattice  # Use scaled lattice velocity
        X, Y = self.grid.X, self.grid.Y
        
        if self.lbm_params.enable_two_phase:
            # 2-phase flow: initialize based on type
            rho_vapor = 0.15
            rho_liquid = 2.0
            interface_width = 2.0
            
            if self.lbm_params.two_phase_init == "droplet":
                # Circular droplet in center
                cx, cy = nx // 2, ny // 2
                radius = min(nx, ny) // 8
                dist_from_center = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
                self.rho = rho_vapor + (rho_liquid - rho_vapor) * 0.5 * (1 - jnp.tanh((dist_from_center - radius) / interface_width))
                
            elif self.lbm_params.two_phase_init == "channel":
                # Liquid channel in middle (horizontal)
                mid_y = ny // 2
                channel_width = ny // 4
                dist_from_center = jnp.abs(Y - mid_y)
                self.rho = rho_vapor + (rho_liquid - rho_vapor) * 0.5 * (1 - jnp.tanh((dist_from_center - channel_width/2) / interface_width))
                
            elif self.lbm_params.two_phase_init == "bubble":
                # Vapor bubble in liquid (inverse of droplet)
                cx, cy = nx // 2, ny // 2
                radius = min(nx, ny) // 8
                dist_from_center = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
                self.rho = rho_liquid - (rho_liquid - rho_vapor) * 0.5 * (1 - jnp.tanh((dist_from_center - radius) / interface_width))
            
            # Zero initial velocity for 2-phase tests
            self.u = jnp.zeros((nx, ny))
            self.v = jnp.zeros((nx, ny))
            
        elif self.sim_params.flow_type == 'lid_driven_cavity':
            # Lid-driven cavity: zero velocity, lid at top
            self.u = jnp.zeros((nx, ny))
            self.v = jnp.zeros((nx, ny))
            self.rho = jnp.ones((nx, ny))
        elif self.sim_params.flow_type == 'taylor_green':
            # Taylor-Green vortex
            self.u = U_lattice * jnp.sin(X) * jnp.cos(Y)
            self.v = -U_lattice * jnp.cos(X) * jnp.sin(Y)
            self.rho = jnp.ones((nx, ny))
        else:
            # von Karman / channel flow: initialize with zero velocity
            # Let inlet boundary condition drive the flow naturally to avoid initial bounce
            self.u = jnp.zeros((nx, ny))
            self.v = jnp.zeros((nx, ny))
            self.rho = jnp.ones((nx, ny))

        # Initialize temperature field to ambient temperature and seed a visible inlet gradient
        self.T = self._initialize_temperature_field()
        
        # Apply mask to initial velocity
        self.u = self.u * self.mask
        self.v = self.v * self.mask
        
        # Initialize previous velocity
        self.u_prev = jnp.copy(self.u)
        self.v_prev = jnp.copy(self.v)
    
    def _initialize_temperature_field(self) -> jnp.ndarray:
        """Initialize the temperature field and seed a visible gradient when thermal coupling is enabled."""
        nx, ny = self.grid.nx, self.grid.ny
        ambient_temp = self.lbm_params.thermal_ambient_temp
        T = jnp.full((nx, ny), ambient_temp, dtype=jnp.float32)

        if self.lbm_params.enable_thermal:
            # Apply custom temperature field from PNG import if available
            if hasattr(self.sim_params, 'custom_temperature_field') and self.sim_params.custom_temperature_field is not None:
                custom_temp = self.sim_params.custom_temperature_field
                import numpy as np
                print(f"DEBUG: Initializing with custom temperature field: shape={custom_temp.shape}, min={custom_temp.min():.2f}, max={custom_temp.max():.2f}, mean={custom_temp.mean():.2f}")
                if custom_temp.shape != (nx, ny):
                    from scipy.ndimage import zoom
                    zoom_x = nx / custom_temp.shape[0]
                    zoom_y = ny / custom_temp.shape[1]
                    custom_temp = zoom(custom_temp, (zoom_x, zoom_y), order=1)
                # Apply custom temperature to the field
                T = jnp.array(custom_temp, dtype=jnp.float32)
            else:
                # Create default gradient if no custom temperature
                x_coords = jnp.arange(nx, dtype=jnp.float32)[:, None]
                y_coords = jnp.arange(ny, dtype=jnp.float32)[None, :]
                inlet_width = max(2, nx // 10)
                inlet_profile = self.lbm_params.thermal_inlet_temp + (
                    ambient_temp - self.lbm_params.thermal_inlet_temp
                ) * jnp.exp(-x_coords / max(inlet_width, 1))
                T = jnp.where(x_coords < inlet_width, inlet_profile, T)

                center_y = ny / 2.0
                vertical_perturbation = 0.5 * (
                    self.lbm_params.thermal_inlet_temp - ambient_temp
                ) * jnp.exp(-((y_coords - center_y) / max(1.0, ny / 6.0)) ** 2)
                T = T + vertical_perturbation

        import numpy as np
        print(f"DEBUG: Final T initialization: min={np.array(T).min():.2f}, max={np.array(T).max():.2f}, mean={np.array(T).mean():.2f}")
        return T

    def _apply_temperature_boundary(self, T: jnp.ndarray) -> jnp.ndarray:
        """Preserve a visible inlet temperature boundary for the thermal field."""
        if not self.lbm_params.enable_thermal:
            return T

        # Apply inlet temperature at custom inlet locations after advection-diffusion
        if hasattr(self.sim_params, 'custom_inlet_mask') and self.sim_params.custom_inlet_mask is not None:
            import numpy as np
            T_np = np.array(T)
            inlet_mask = np.array(self.sim_params.custom_inlet_mask)
            inlet_temp = self.lbm_params.thermal_inlet_temp
            T_np = np.where(inlet_mask > 0.5, inlet_temp, T_np)
            return jnp.array(T_np)

        # Default inlet boundary (left side)
        T = T.at[:, 0].set(self.lbm_params.thermal_inlet_temp)
        return T

    def _compute_mask(self) -> jnp.ndarray:
        """Compute obstacle mask (1 = fluid, 0 = solid)"""
        # Special case for lid_driven_cavity and taylor_green - all fluid (mask = 1 everywhere)
        if self.sim_params.flow_type == 'lid_driven_cavity':
            return jnp.ones_like(self.grid.X)
        if self.sim_params.flow_type == 'taylor_green':
            return jnp.ones_like(self.grid.X)
        
        # For obstacles, use the geometry module
        if self.sim_params.obstacle_type == 'naca_airfoil':
            from obstacles.naca_airfoils import NACAParams, create_naca_mask, parse_naca_4digit, parse_naca_5digit
            
            # Parse NACA designation
            naca_str = self.sim_params.naca_airfoil.upper().replace('NACA', '').strip()
            if len(naca_str) == 4:
                m, p, t = parse_naca_4digit(naca_str)
                airfoil_type = '4-digit'
            elif len(naca_str) == 5:
                cl, p, m, t = parse_naca_5digit(naca_str)
                airfoil_type = '5-digit'
            else:
                raise ValueError(f"Unsupported NACA designation: {self.sim_params.naca_airfoil}")
            
            naca_params = NACAParams(
                airfoil_type=airfoil_type,
                designation=self.sim_params.naca_airfoil,
                chord_length=self.sim_params.naca_chord,
                angle_of_attack=self.sim_params.naca_angle,
                position_x=self.sim_params.naca_x,
                position_y=self.sim_params.naca_y
            )
            
            mask = create_naca_mask(self.grid.X, self.grid.Y, naca_params, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'cylinder':
            from solver.geometry import sdf_cylinder, smooth_mask
            phi = sdf_cylinder(self.grid.X, self.grid.Y, self.geom.center_x, self.geom.center_y, self.geom.radius)
            mask = smooth_mask(phi, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'cow':
            from obstacles.cow import create_cow_mask
            mask = create_cow_mask(self.grid.X, self.grid.Y, self.sim_params.cow_x, self.sim_params.cow_y, self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'three_cylinder_array':
            from obstacles.cylinder_array import create_three_cylinder_mask
            mask = create_three_cylinder_mask(self.grid.X, self.grid.Y,
                                            self.sim_params.cylinder_x, self.sim_params.cylinder_y,
                                            self.sim_params.eps)
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        elif self.sim_params.obstacle_type == 'custom':
            custom_mask = getattr(self.sim_params, 'custom_mask', None)
            fluid_mask = getattr(self.sim_params, 'custom_fluid_mask', None)
            inlet_mask = getattr(self.sim_params, 'custom_inlet_mask', None)
            outlet_mask = getattr(self.sim_params, 'custom_outlet_mask', None)
            
            # Check if masks match current grid dimensions
            current_shape = self.grid.X.shape
            if fluid_mask is not None and jnp.array(fluid_mask).shape != current_shape:
                print(f"Warning: fluid_mask shape {jnp.array(fluid_mask).shape} doesn't match grid {current_shape}, ignoring")
                fluid_mask = None
            if inlet_mask is not None and jnp.array(inlet_mask).shape != current_shape:
                print(f"Warning: inlet_mask shape {jnp.array(inlet_mask).shape} doesn't match grid {current_shape}, ignoring")
                inlet_mask = None
            if outlet_mask is not None and jnp.array(outlet_mask).shape != current_shape:
                print(f"Warning: outlet_mask shape {jnp.array(outlet_mask).shape} doesn't match grid {current_shape}, ignoring")
                outlet_mask = None
            if custom_mask is not None and jnp.array(custom_mask).shape != current_shape:
                print(f"Warning: custom_mask shape {jnp.array(custom_mask).shape} doesn't match grid {current_shape}, ignoring")
                custom_mask = None
            
            if custom_mask is None and (fluid_mask is not None or inlet_mask is not None or outlet_mask is not None):
                mask = jnp.zeros_like(self.grid.X)
                if fluid_mask is not None:
                    mask = jnp.where(jnp.array(fluid_mask) > 0.5, 1.0, mask)
                if inlet_mask is not None:
                    mask = jnp.where(jnp.array(inlet_mask) > 0.5, 1.0, mask)
                if outlet_mask is not None:
                    mask = jnp.where(jnp.array(outlet_mask) > 0.5, 1.0, mask)
            elif custom_mask is not None:
                mask = jnp.array(custom_mask)
            else:
                mask = jnp.ones_like(self.grid.X)
            
            # Force binary mask for LBM bounce-back to work properly
            mask = jnp.where(mask > 0.5, 1.0, 0.0)
        else:
            # Default: all fluid
            mask = jnp.ones_like(self.grid.X)
        
        return mask
    
    def set_obstacle_type(self, obstacle_type: str):
        """Change obstacle type and recompute mask"""
        self.sim_params.obstacle_type = obstacle_type
        self.mask = self._compute_mask()
        # Clear JIT cache since mask changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print(f"LBM: Obstacle type changed to {obstacle_type}")
    
    def apply_flow_type(self, flow_type: str):
        """Change flow type and reinitialize"""
        self.sim_params.flow_type = flow_type
        # Recompute mask for new flow type
        self.mask = self._compute_mask()
        self._initialize_flow()
        from .collision import equilibrium
        self.f = equilibrium(self.rho, self.u, self.v, 
                            self.lattice.get_cx(), self.lattice.get_cy(),
                            self.lattice.w, self.lattice.cs_squared)
        self.iteration = 0
        # Clear JIT cache since flow type changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print(f"LBM: Flow type changed to {flow_type}")
    
    def set_tau(self, tau: float):
        """Update tau parameter and recompile JIT functions"""
        # Ensure stability bounds
        tau_min = 0.5
        tau_max = 2.0
        self.lbm_params.tau = jnp.clip(tau, tau_min, tau_max)
        self.lbm_params.omega = 1.0 / self.lbm_params.tau
        # Update MRT relaxation rates with standard values to reduce artificial diffusion
        s_visc = self.lbm_params.omega
        s_bulk = jnp.clip(1.0, 0.5, 1.9)
        s_energy = jnp.clip(1.0, 0.5, 1.9)
        s_shear = s_visc
        self.s = jnp.array([
            s_bulk, s_energy, s_energy,
            s_shear, s_shear,
            s_shear, s_shear,
            s_shear, s_shear
        ])
        self.s = jnp.clip(self.s, 0.01, 1.99)
        # Clear JIT cache since tau changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
    
    def update_spin(self):
        """Update spin angle and rotate custom mask if spin is enabled"""
        if not self.lbm_params.enable_spin or self.lbm_params.spin_rpm == 0.0:
            return
        
        # Only update mask rotation every N steps to reduce JIT recompilation
        self.spin_step_counter += 1
        if self.spin_step_counter < self.lbm_params.spin_update_interval:
            return
        
        self.spin_step_counter = 0  # Reset counter
        
        if self.original_custom_mask is None:
            print("Warning: original_custom_mask is None, cannot spin")
            return
        
        # Update spin angle based on RPM, dt, and interval
        # RPM to degrees per second: RPM * 360 / 60 = RPM * 6
        degrees_per_second = self.lbm_params.spin_rpm * 6.0
        angle_increment = degrees_per_second * self.dt * self.lbm_params.spin_update_interval
        self.spin_angle = (self.spin_angle + angle_increment) % 360.0
        
        # Rotate the original mask
        rotated_mask = rotate(self.original_custom_mask, self.spin_angle, 
                            reshape=False, mode='nearest', order=0)
        
        # Update the custom mask in sim_params
        self.sim_params.custom_mask = jnp.array(rotated_mask)
        
        # Recompute mask with rotated custom mask
        self.mask = self._compute_mask()
        
        # Clear JIT cache since mask changed
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
    
    def apply_vorticity_confinement(self, f: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, 
                                   epsilon: float, cx: jnp.ndarray, cy: jnp.ndarray, 
                                   w: jnp.ndarray, cs_squared: float) -> jnp.ndarray:
        """Apply vorticity confinement force to enhance vortical structures
        
        Args:
            f: Distribution function (9, nx, ny)
            u: Velocity x-component (nx, ny)
            v: Velocity y-component (nx, ny)
            epsilon: Confinement coefficient
            cx: Lattice velocity x-components (9,)
            cy: Lattice velocity y-components (9,)
            w: Lattice weights (9,)
            cs_squared: Speed of sound squared
        
        Returns:
            f: Distribution function with vorticity confinement applied
        """
        if epsilon <= 0.0:
            return f
        
        # Compute vorticity (curl of velocity)
        # omega_z = dv/dx - du/dy
        du_dy = jnp.gradient(u, axis=1)
        dv_dx = jnp.gradient(v, axis=0)
        omega = dv_dx - du_dy
        
        # Compute gradient of vorticity magnitude
        omega_mag = jnp.abs(omega)
        domega_dx = jnp.gradient(omega_mag, axis=0)
        domega_dy = jnp.gradient(omega_mag, axis=1)
        
        # Normalize gradient
        grad_mag = jnp.sqrt(domega_dx**2 + domega_dy**2) + 1e-10
        nx = domega_dx / grad_mag
        ny = domega_dy / grad_mag
        
        # Compute confinement force: F = epsilon * h * (N x omega)
        # In 2D: Fx = epsilon * ny * omega, Fy = -epsilon * nx * omega
        # Grid spacing h = 1 in lattice units
        Fx = epsilon * ny * omega
        Fy = -epsilon * nx * omega
        
        # Apply force to distribution function using Guo forcing scheme
        # F_i = w_i * (3 * (c_i - u) * F + 9 * (c_i * u) * (c_i * F)) / cs^2
        for i in range(9):
            ci_dot_u = cx[i] * u + cy[i] * v
            ci_dot_F = cx[i] * Fx + cy[i] * Fy
            u_dot_F = u * Fx + v * Fy
            
            force_term = w[i] * (
                3.0 * (ci_dot_u - u_dot_F) + 
                9.0 * ci_dot_F * ci_dot_u
            ) / cs_squared
            
            f = f.at[i].add(force_term)
        
        return f
    
    def mrt_collision(self, f, rho, u, v):
        """MRT collision operator with regularization for improved stability"""
        from .collision import equilibrium
        
        # Compute equilibrium distribution
        f_eq = equilibrium(rho, u, v, 
                          self.lattice.get_cx(), self.lattice.get_cy(),
                          self.lattice.w, self.lattice.cs_squared)
        
        # Flatten f and f_eq for matrix multiplication
        shape = f.shape
        f_flat = f.reshape(9, -1)
        f_eq_flat = f_eq.reshape(9, -1)
        
        # Transform to moment space
        m = self.M @ f_flat
        m_eq = self.M @ f_eq_flat
        
        # Relaxation in moment space
        s_diag = jnp.diag(self.s)
        m_post = m - s_diag @ (m - m_eq)
        
        # Transform back to distribution space
        f_post = self.M_inv @ m_post
        f_post = f_post.reshape(shape)
        
        # Apply regularization: filter high-order non-equilibrium moments
        # This reduces numerical noise and improves stability
        f_neq = f_post - f_eq
        
        # Compute stress tensor from non-equilibrium
        cx = self.lattice.get_cx()[:, None, None]
        cy = self.lattice.get_cy()[:, None, None]
        cs_squared = self.lattice.cs_squared
        
        Pi_neq_xx = jnp.sum(f_neq * (cx**2 - cs_squared), axis=0)
        Pi_neq_yy = jnp.sum(f_neq * (cy**2 - cs_squared), axis=0)
        Pi_neq_xy = jnp.sum(f_neq * cx * cy, axis=0)
        
        # Reconstruct filtered non-equilibrium part (only keep second-order moments)
        f_neq_filtered = jnp.zeros_like(f_neq)
        for i in range(9):
            f_neq_filtered = f_neq_filtered.at[i].set(
                self.lattice.w[i] * (
                    Pi_neq_xx * (cx[i]**2 - cs_squared) / (2*cs_squared**2) +
                    Pi_neq_yy * (cy[i]**2 - cs_squared) / (2*cs_squared**2) +
                    Pi_neq_xy * (cx[i]*cy[i]) / cs_squared**2
                )
            )
        
        # Apply regularization with blending factor (0.7 = 70% regularized)
        # Balanced regularization reduces ringing without excessive diffusion
        reg_factor = 0.7
        f_final = f_eq + reg_factor * f_neq_filtered + (1 - reg_factor) * f_neq
        
        return f_final
    
    def get_step_jit(self):
        """Return JIT-compiled step function"""
        if 'step_jit' not in self._jit_cache:
            # Extract lattice data as static arrays
            cx = self.lattice.get_cx()
            cy = self.lattice.get_cy()
            w = self.lattice.w
            opposite = self.lattice.opposite
            cs_squared = self.lattice.cs_squared
            
            # Include MRT matrices if MRT is enabled
            use_mrt = self.lbm_params.use_mrt
            mrt_data = {}
            if use_mrt:
                mrt_data = {
                    'M': self.M,
                    'M_inv': self.M_inv,
                    's': self.s
                }
            
            # Create JIT function with static lattice data and grid dimensions
            self._jit_cache['step_jit'] = jax.jit(
                self._step_pure,
                static_argnames=['flow_type', 'nx', 'ny', 'use_mrt', 'vorticity_confinement',
                                'bc_left', 'bc_right', 'bc_top', 'bc_bottom', 'startup_ramp_steps',
                                'enable_startup_ramp', 'enable_two_phase', 'use_les', 'les_model']
            )
            # Store lattice data for use in step
            self._lattice_data = {
                'cx': cx,
                'cy': cy,
                'w': w,
                'opposite': opposite,
                'cs_squared': cs_squared,
                'use_mrt': use_mrt,
                'mrt_data': mrt_data,
                'outlet_pressure_left': self.lbm_params.outlet_pressure_left,
                'outlet_pressure_right': self.lbm_params.outlet_pressure_right,
                'outlet_pressure_top': self.lbm_params.outlet_pressure_top,
                'outlet_pressure_bottom': self.lbm_params.outlet_pressure_bottom
            }
        return self._jit_cache['step_jit']
    
    def update_lattice_data(self):
        """Update lattice data with current outlet pressure values"""
        if hasattr(self, '_lattice_data'):
            self._lattice_data['outlet_pressure_left'] = self.lbm_params.outlet_pressure_left
            self._lattice_data['outlet_pressure_right'] = self.lbm_params.outlet_pressure_right
            self._lattice_data['outlet_pressure_top'] = self.lbm_params.outlet_pressure_top
            self._lattice_data['outlet_pressure_bottom'] = self.lbm_params.outlet_pressure_bottom
    
    def _step_pure(self, f: jnp.ndarray, mask: jnp.ndarray, lattice_data: dict, omega: float,
                   flow_type: str, u_inlet: float, nx: int, ny: int, use_mrt: bool = False,
                   vorticity_confinement: float = 0.0, bc_left: str = None, bc_right: str = None,
                   bc_top: str = None, bc_bottom: str = None, iteration: int = 0,
                   startup_ramp_steps: int = 100, enable_startup_ramp: bool = True,
                   enable_two_phase: bool = False, G: float = -5.0, psi0: float = 1.0, 
                   rho0: float = 1.0, gravity: float = 0.0, dt: float = 1.0,
                   T: jnp.ndarray = None, enable_thermal: bool = False,
                   thermal_diffusivity: float = 0.01,
                   thermal_expansion_coeff: float = 0.003,
                   thermal_gravity: float = 1.0,
                   thermal_reference: float = 0.0,
                   thermal_inlet_temp: float = 20.0,
                   custom_inlet_mask: jnp.ndarray = None,
                   custom_outlet_mask: jnp.ndarray = None,
                   custom_inlet_angle: float = 0.0,
                   custom_inlet_velocity: float = None,
                   use_les: bool = False, les_model: str = 'dynamic_smagorinsky',
                   smagorinsky_constant: float = 0.17, dx: float = 0.1, dy: float = 0.1) -> Tuple:
        """
        Pure JAX-compatible LBM step (collision + streaming + all boundary conditions)
        
        Args:
            f: Distribution function (9, nx, ny)
            mask: Obstacle mask (nx, ny)
            lattice_data: Dictionary with lattice arrays (cx, cy, w, opposite, cs_squared, mrt_data)
            omega: Collision frequency
            flow_type: Type of flow ('von_karman', 'lid_driven_cavity', 'taylor_green')
            u_inlet: Inlet velocity for channel flows
            nx: Grid size x
            ny: Grid size y
            use_mrt: Whether to use MRT collision model
            bc_left: Boundary condition type for left ('inlet', 'outlet', 'farfield', 'wall')
            bc_right: Boundary condition type for right ('inlet', 'outlet', 'farfield', 'wall')
            bc_top: Boundary condition type for top ('inlet', 'outlet', 'farfield', 'wall')
            bc_bottom: Boundary condition type for bottom ('inlet', 'outlet', 'farfield', 'wall')
        
        Returns:
            f_final: Distribution after collision, streaming, and all boundary conditions
            rho: Density field
            u: Velocity x-component
            v: Velocity y-component
        """
        from .boundary import apply_bounce_back, apply_boundary_conditions
        from .collision import equilibrium, bgk_collision
        from .twophase import get_psi, compute_sc_force, apply_force_to_velocity
        from .operators import compute_unforced_velocity
        
        cx = lattice_data['cx']
        cy = lattice_data['cy']
        w = lattice_data['w']
        opposite = lattice_data['opposite']
        cs_squared = lattice_data['cs_squared']
        
        # Apply startup velocity ramp to prevent instability at high speeds
        if enable_startup_ramp and startup_ramp_steps > 0:
            ramp_factor = jnp.minimum(1.0, iteration / startup_ramp_steps)
            u_inlet_ramped = u_inlet * ramp_factor
        else:
            u_inlet_ramped = u_inlet
        
        # Collision
        if enable_two_phase:
            # For 2-phase flow, compute unforced velocity first
            rho, u_star, v_star = compute_unforced_velocity(f, cx, cy)
            
            # Calculate pseudopotential
            psi = get_psi(rho, psi0, rho0)
            
            # Compute Shan-Chen interaction force
            force = compute_sc_force(psi, G, w, cx, cy)
            
            # Apply force to get physical velocity
            tau = 1.0 / omega
            u, v = apply_force_to_velocity(u_star, v_star, force, rho, tau, gravity)
        else:
            # Single-phase: regular macroscopic variables
            rho, u, v = macroscopic_variables(f, cx, cy)
        
        # Apply mask to velocities: set velocity to zero inside obstacles
        u = u * mask
        v = v * mask
        
        # Clamp density to prevent numerical instability
        # Density should stay close to 1.0 (in lattice units)
        rho = jnp.clip(rho, 0.5, 2.0)
        
        # Clamp velocity to prevent numerical instability
        # Maximum velocity should be < 0.3 (Mach < 0.3 for compressibility)
        max_vel = 0.3
        vel_mag = jnp.sqrt(u**2 + v**2)
        vel_clamp_factor = jnp.minimum(1.0, max_vel / (vel_mag + 1e-10))
        u = u * vel_clamp_factor
        v = v * vel_clamp_factor
        
        # Apply vorticity confinement to macroscopic velocity before collision
        if vorticity_confinement > 0.0:
            # Scale the parameter so that slider range 0-0.01 gives appropriate effect
            # User's 0.01 should be max reasonable effect
            epsilon = vorticity_confinement * 0.01  # Scale down for reasonable effect

            # Compute vorticity
            du_dy = jnp.gradient(u, axis=1)
            dv_dx = jnp.gradient(v, axis=0)
            omega = dv_dx - du_dy

            # Compute gradient of vorticity magnitude
            omega_mag = jnp.abs(omega)
            domega_dx = jnp.gradient(omega_mag, axis=0)
            domega_dy = jnp.gradient(omega_mag, axis=1)

            # Normalize gradient
            grad_mag = jnp.sqrt(domega_dx**2 + domega_dy**2) + 1e-10
            nx = domega_dx / grad_mag
            ny = domega_dy / grad_mag

            # Compute confinement force: F = epsilon * (N × omega)
            Fx = epsilon * ny * omega
            Fy = -epsilon * nx * omega

            # Apply force directly to velocity (simpler and more stable for LBM)
            # This modifies the velocity used in equilibrium calculation
            u = u + Fx
            v = v + Fy

        # Apply thermal buoyancy when enabled
        if T is None:
            T = jnp.zeros_like(u)

        def _thermal_buoyancy(_):
            temp_anomaly = T - thermal_reference
            buoyancy = thermal_gravity * thermal_expansion_coeff * temp_anomaly
            return v + buoyancy * dt

        def _no_buoyancy(_):
            return v

        v = jax.lax.cond(enable_thermal, _thermal_buoyancy, _no_buoyancy, operand=None)

        # Use MRT or BGK collision based on flag
        if use_mrt:
            # MRT collision
            mrt_data = lattice_data['mrt_data']
            if 'M' not in mrt_data or mrt_data['M'] is None:
                # Fallback to BGK if MRT data is not available
                use_mrt = False
            else:
                M = mrt_data['M']
                M_inv = mrt_data['M_inv']
                s = mrt_data['s']

                # Compute equilibrium distribution
                f_eq = equilibrium(rho, u, v, cx, cy, w, cs_squared)

                # Flatten f and f_eq for matrix multiplication
                shape = f.shape
                f_flat = f.reshape(9, -1)
                f_eq_flat = f_eq.reshape(9, -1)

                # Transform to moment space
                m = M @ f_flat
                m_eq = M @ f_eq_flat

                # Relaxation in moment space
                s_diag = jnp.diag(s)
                m_post = m - s_diag @ (m - m_eq)

                # Transform back to distribution space
                f_post = M_inv @ m_post
                f_post = f_post.reshape(shape)

        if not use_mrt:
            # BGK collision with optional LES
            use_les = self.lbm_params.use_les
            les_model = self.lbm_params.les_model
            smagorinsky_constant = self.lbm_params.smagorinsky_constant
            dx = self.grid.dx
            dy = self.grid.dy
            
            f_post = collision_step(f, rho, u, v, cx, cy, w, cs_squared, omega,
                                   use_les=use_les, les_model=les_model,
                                   smagorinsky_constant=smagorinsky_constant,
                                   dx=dx, dy=dy)
        
        # Streaming
        f_streamed = streaming_step(f_post, cx, cy)
        
        # Apply bounce-back for obstacles (inside JIT)
        # Use under-relaxed bounce-back (eta_max=0.95) to reduce artificial diffusion
        f_bc = apply_bounce_back(f_streamed, mask, opposite, eta_max=0.95)
        
        # Apply flow-specific boundary conditions (inside JIT)
        # Use per-boundary configuration if provided, otherwise use flow_type defaults
        # Get outlet pressure values from lbm_params (passed via lattice_data)
        outlet_pressure_left = lattice_data.get('outlet_pressure_left', 0.0)
        outlet_pressure_right = lattice_data.get('outlet_pressure_right', 0.0)
        outlet_pressure_top = lattice_data.get('outlet_pressure_top', 0.0)
        outlet_pressure_bottom = lattice_data.get('outlet_pressure_bottom', 0.0)
        
        f_final, T_updated = apply_boundary_conditions(
            f_bc, mask, opposite, flow_type, u_inlet_ramped, nx, ny, cx, cy, w, cs_squared,
            bc_left=bc_left, bc_right=bc_right, bc_top=bc_top, bc_bottom=bc_bottom,
            custom_inlet_mask=custom_inlet_mask,
            custom_outlet_mask=custom_outlet_mask,
            custom_inlet_angle=custom_inlet_angle,
            custom_inlet_velocity=custom_inlet_velocity,
            T=T,
            thermal_inlet_temp=thermal_inlet_temp,
            outlet_pressure_left=outlet_pressure_left,
            outlet_pressure_right=outlet_pressure_right,
            outlet_pressure_top=outlet_pressure_top,
            outlet_pressure_bottom=outlet_pressure_bottom
        )
        
        # Compute macroscopic variables for output
        rho_new, u_new, v_new = macroscopic_variables(f_final, cx, cy)
        
        # Apply mask to final velocities to ensure zero inside obstacles
        u_new = u_new * mask
        v_new = v_new * mask
        
        return f_final, rho_new, u_new, v_new, T_updated
    
    def step(self) -> Tuple:
        """Perform one LBM step"""
        # Run complete LBM step (collision + streaming + all boundary conditions) in JIT
        custom_inlet_mask = getattr(self.sim_params, 'custom_inlet_mask', None)
        custom_outlet_mask = getattr(self.sim_params, 'custom_outlet_mask', None)
        custom_inlet_angle = getattr(self.sim_params, 'custom_inlet_angle', 0.0)
        custom_inlet_velocity = getattr(self.sim_params, 'custom_inlet_velocity', None)
        
        # Convert custom inlet velocity from physical units to lattice units if provided
        if custom_inlet_velocity is not None:
            dt = self.dt
            dx = self.grid.dx
            custom_inlet_velocity_lattice = custom_inlet_velocity * dt / dx
        else:
            custom_inlet_velocity_lattice = None
        
        f_final, rho_new, u_new, v_new, T_updated = self._step_jit(
            self.f, self.mask, self._lattice_data, self.lbm_params.omega,
            self.sim_params.flow_type, self.U_lattice,
            self.grid.nx, self.grid.ny,
            self.lbm_params.use_mrt,
            self.lbm_params.vorticity_confinement,
            self.lbm_params.bc_left,
            self.lbm_params.bc_right,
            self.lbm_params.bc_top,
            self.lbm_params.bc_bottom,
            self.iteration,
            self.lbm_params.startup_ramp_steps,
            self.lbm_params.enable_startup_ramp,
            self.lbm_params.enable_two_phase,
            self.lbm_params.G,
            self.lbm_params.psi0,
            self.lbm_params.rho0,
            self.lbm_params.gravity,
            self.dt,
            self.T,
            self.lbm_params.enable_thermal,
            self.lbm_params.thermal_diffusivity,
            self.lbm_params.thermal_expansion_coeff,
            self.lbm_params.thermal_gravity,
            self.lbm_params.thermal_reference,
            self.lbm_params.thermal_inlet_temp,
            custom_inlet_mask,
            custom_outlet_mask,
            custom_inlet_angle,
            custom_inlet_velocity_lattice,
            self.lbm_params.use_les,
            self.lbm_params.les_model,
            self.lbm_params.smagorinsky_constant,
            self.grid.dx,
            self.grid.dy
        )
        
        # Update temperature field from boundary conditions
        if T_updated is not None:
            self.T = T_updated
        
        # Compute pressure
        from .operators import compute_pressure
        p_new = compute_pressure(rho_new, self._lattice_data['cs_squared'], rho0=1.0)
        
        # Update state
        self.u_prev = self.u
        self.v_prev = self.v
        self.f = f_final
        self.u = u_new
        self.v = v_new
        self.rho = rho_new
        self.p = p_new
        self.current_pressure = p_new
        
        # Advect dye (passive scalar) using finite difference
        if self.lbm_params.enable_dye:
            from solver.operators import scalar_advection_diffusion_nonperiodic
            dx = self.grid.lx / self.grid.nx
            dy = self.grid.ly / self.grid.ny
            
            # Ensure dye field exists
            if not hasattr(self, 'c') or self.c is None:
                self.c = jnp.zeros((self.grid.nx, self.grid.ny))
                
            self.c = scalar_advection_diffusion_nonperiodic(
                self.c, self.u, self.v, self.dt, dx, dy, self.lbm_params.dye_diffusivity
            )

        # Advect thermal field if enabled and preserve solid temperature
        if self.lbm_params.enable_thermal:
            from solver.operators import scalar_advection_diffusion_nonperiodic_no_clamp
            dx = self.grid.lx / self.grid.nx
            dy = self.grid.ly / self.grid.ny
            
            # Debug: print temperature range every 100 steps
            if self.iteration % 100 == 0:
                import numpy as np
                T_np = np.array(self.T)
                print(f"Step {self.iteration}: T_min={T_np.min():.2f}, T_max={T_np.max():.2f}, T_mean={T_np.mean():.2f}")
                print(f"  enable_thermal={self.lbm_params.enable_thermal}, thermal_diffusivity={self.lbm_params.thermal_diffusivity}")
                print(f"  u_max={np.array(self.u).max():.4f}, v_max={np.array(self.v).max():.4f}")
            
            T_before = self.T.copy()
            self.T = scalar_advection_diffusion_nonperiodic_no_clamp(
                self.T, self.u, self.v, self.dt, dx, dy, self.lbm_params.thermal_diffusivity
            )
            T_after = self.T
            
            # Check if advection-diffusion actually changed the temperature
            if self.iteration % 100 == 0:
                import numpy as np
                diff = np.array(T_after - T_before)
                print(f"  T_change_max={np.abs(diff).max():.6f}")
            
            self.T = self._apply_temperature_boundary(self.T)
            
            # Reapply custom inlet temperature after advection-diffusion
            # This ensures inlet stays at prescribed temperature while allowing advection/diffusion
            if hasattr(self.sim_params, 'custom_inlet_mask') and self.sim_params.custom_inlet_mask is not None:
                custom_inlet_mask = self.sim_params.custom_inlet_mask
                import numpy as np
                if self.iteration % 100 == 0:
                    print(f"  inlet_mask_sum={np.sum(custom_inlet_mask):.0f}, inlet_mask_max={np.max(custom_inlet_mask):.2f}")
                self.T = jnp.where(custom_inlet_mask > 0.5, self.lbm_params.thermal_inlet_temp, self.T)
            
            # Debug: check if custom inlet mask is being used
            if self.iteration % 100 == 0:
                has_custom_inlet = hasattr(self.sim_params, 'custom_inlet_mask') and self.sim_params.custom_inlet_mask is not None
                print(f"  has_custom_inlet_mask={has_custom_inlet}")
        
        self.iteration += 1
        
        # Update history
        self.history['time'].append(self.iteration * self.dt)
        self.history['dt'].append(self.dt)
        
        return self.u, self.v, self.p

    def inject_dye(self, x_pos: float, y_pos: float, amount: float = 0.5):
        """Inject dye at physical coordinates"""
        # Ensure dye field exists
        if not hasattr(self, 'c') or self.c is None:
            self.c = jnp.zeros((self.grid.nx, self.grid.ny))
        
        x_clamped = max(0.0, min(x_pos, self.grid.lx))
        y_clamped = max(0.0, min(y_pos, self.grid.ly))
        
        ix = int(x_clamped / self.grid.dx)
        iy = int(y_clamped / self.grid.dy)
        
        # Inject into a 5x5 area around the target cell for smoother distribution
        radius = 2
        dx_offsets = jnp.arange(-radius, radius + 1)
        dy_offsets = jnp.arange(-radius, radius + 1)
        dx_grid, dy_grid = jnp.meshgrid(dx_offsets, dy_offsets, indexing='ij')
        
        # Compute target indices
        ix_targets = ix + dx_grid
        iy_targets = iy + dy_grid
        
        # Clamp to grid bounds
        ix_targets = jnp.clip(ix_targets, 0, self.grid.nx - 1)
        iy_targets = jnp.clip(iy_targets, 0, self.grid.ny - 1)
        
        # Compute Gaussian falloff
        distances = jnp.sqrt(dx_grid**2 + dy_grid**2)
        falloffs = jnp.exp(-distances**2 / 2.0)
        
        # Update dye concentration
        current_values = self.c[ix_targets, iy_targets]
        new_values = jnp.minimum(current_values + amount * falloffs, 1.0)
        
        # Apply updates using scatter
        self.c = self.c.at[ix_targets, iy_targets].set(new_values)
        
        print(f"Dye injected at ({x_pos:.2f}, {y_pos:.2f}) -> grid ({ix}, {iy})")

    def step_for_visualization(self, compute_divergence: bool = False,
                               compute_drag_lift: bool = False,
                               compute_diagnostics: bool = False) -> Tuple:
        """
        Step and return data for visualization (compatible with BaselineSolver)
        
        Returns:
            u, v, vort, div (divergence may be None)
        """
        # Perform step
        u, v, p = self.step()
        
        # Compute vorticity
        from solver.operators import vorticity_nonperiodic
        vort = vorticity_nonperiodic(u, v, self.grid.dx, self.grid.dy)
        
        # Compute divergence if requested
        div = None
        if compute_divergence:
            from solver.operators import divergence_nonperiodic
            div = divergence_nonperiodic(u, v, self.grid.dx, self.grid.dy)
        
        # Compute airfoil metrics if enabled (same as baseline solver)
        if self.compute_airfoil_metrics and hasattr(self.sim_params, 'flow_type') and self.sim_params.flow_type == 'von_karman':
            try:
                import numpy as np
                from solver.metrics import (
                    find_stagnation_point, find_separation_point, compute_forces_ibm,
                    get_airfoil_surface_mask, compute_CL_circulation, compute_drag_momentum_deficit
                )
                
                # Convert JAX arrays to numpy for metrics computation
                u_np = np.array(u)
                v_np = np.array(v)
                p_np = np.array(p)
                
                # Get mask
                mask_np = np.array(self.mask)
                
                # Grid parameters
                X_np = np.array(self.grid.X)
                dx = self.grid.dx
                dy = self.grid.dy
                
                # Find stagnation and separation points
                stag_x = find_stagnation_point(u_np, v_np, mask_np, p_np, X_np, dx)
                sep_x = find_separation_point(u_np, v_np, mask_np, X_np, dx, dy)
                
                # Get obstacle parameters
                chord_length = getattr(self.sim_params, 'naca_chord', 2.0)
                airfoil_x = getattr(self.sim_params, 'naca_x', 2.5)
                airfoil_y = getattr(self.sim_params, 'naca_y', 1.875)
                
                # Compute forces using circulation-based method
                cl, cd = compute_forces_ibm(u_np, v_np, vort, X_np, np.array(self.grid.Y), mask_np,
                                              dx, dy, self.flow.U_inf, chord_length, airfoil_x, airfoil_y,
                                              self.grid.lx, grid_type='collocated')
                
                # Compute pressure coefficient
                rho = 1.0
                surface = get_airfoil_surface_mask(mask_np, dx, threshold=0.1)
                p_inf = 0.0
                q_inf = 0.5 * rho * self.flow.U_inf**2
                cp = (p_np - p_inf) / q_inf
                cp_surface = np.where(surface, cp, np.inf)
                cp_min = float(np.min(cp_surface))
                
                # Compute wake deficit
                wake_x = airfoil_x + chord_length
                wake_x_idx = int(wake_x / dx)
                wake_deficit = 0.0
                if 0 <= wake_x_idx < self.grid.nx:
                    u_wake = u_np[wake_x_idx, :]
                    wake_deficit = float(self.flow.U_inf - np.mean(u_wake[mask_np[wake_x_idx, :] > 0.5]))
                
                # Store metrics in history (same format as baseline solver)
                self.history['airfoil_metrics']['stagnation_x'].append(stag_x)
                self.history['airfoil_metrics']['separation_x'].append(sep_x)
                self.history['airfoil_metrics']['CL'].append(cl)
                self.history['airfoil_metrics']['CD'].append(cd)
                self.history['airfoil_metrics']['Cp_min'].append(cp_min)
                self.history['airfoil_metrics']['wake_deficit'].append(wake_deficit)
                self.history['airfoil_metrics']['strouhal'].append(0.0)  # Initialize with .0, updated when stable
                self.history['airfoil_metrics']['time'].append(self.iteration * self.dt)
                
                # Store drag and lift for compatibility
                self.history['drag'].append(float(cd))
                self.history['lift'].append(float(cl))
                
                # Check for vortex shedding stability (same as baseline solver)
                if len(self.history['airfoil_metrics']['CL']) >= 30 and self.iteration % 25 == 0:
                    from solver.metrics import detect_vortex_shedding_stability, compute_time_averaged_coefficients
                    
                    cl_history = np.array(self.history['airfoil_metrics']['CL'])
                    cd_history = np.array(self.history['airfoil_metrics']['CD'])
                    times = np.array(self.history['time'])
                    
                    is_stable, stable_start, strouhal = detect_vortex_shedding_stability(
                        cl_history, times, self.flow.U_inf, chord_length
                    )
                    
                    # Store stability state
                    if 'stability_state' not in self.history:
                        self.history['stability_state'] = []
                    
                    self.history['stability_state'].append({
                        'iteration': self.iteration,
                        'is_stable': is_stable,
                        'stable_start': stable_start,
                        'strouhal': strouhal
                    })
                    
                    # If stable, compute time-averaged coefficients
                    if is_stable and stable_start > 0:
                        avg_cl, avg_cd, num_samples = compute_time_averaged_coefficients(
                            cl_history, cd_history, stable_start
                        )
                        
                        # Store time-averaged values
                        if 'time_averaged' not in self.history:
                            self.history['time_averaged'] = {
                                'CL': [], 'CD': [], 'stable_start': [], 'strouhal': [], 'num_samples': []
                            }
                        
                        self.history['time_averaged']['CL'].append(avg_cl)
                        self.history['time_averaged']['CD'].append(avg_cd)
                        self.history['time_averaged']['stable_start'].append(stable_start)
                        self.history['time_averaged']['strouhal'].append(strouhal)
                        self.history['time_averaged']['num_samples'].append(num_samples)
                        
                        # Update recent Strouhal values in airfoil metrics
                        if len(self.history['airfoil_metrics']['strouhal']) > 0:
                            num_updates = min(10, len(self.history['airfoil_metrics']['strouhal']))
                            for i in range(num_updates):
                                self.history['airfoil_metrics']['strouhal'][-(i+1)] = strouhal
                        
            except Exception as e:
                print(f"LBM Error computing airfoil metrics: {e}")
                # Append zeros to maintain history consistency
                for key in self.history['airfoil_metrics']:
                    self.history['airfoil_metrics'][key].append(0.0)
                self.history['drag'].append(0.0)
                self.history['lift'].append(0.0)
        
        return u, v, vort, div
    
    def update_flow_parameters(self):
        """Update LBM parameters when flow parameters change (Re, U, nu)"""
        print("LBM: Updating flow parameters...")
        
        # Recalculate LBM parameters based on new flow parameters
        cs_squared = 1.0 / 3.0
        
        # Choose target lattice Reynolds number for aggressive diffusion reduction
        # Much higher Re_lattice = much less diffusion, push stability limits
        # For vortex shedding, aim for Re_lattice > 1000 for strong vortices
        target_Re_lattice = min(5000.0, self.flow.Re * 10.0)  # Much more aggressive scaling
        
        # Calculate lattice velocity based on physical velocity
        # Scale physical velocity to lattice units while maintaining stability
        physical_velocity = self.flow.U_inf
        
        # Calculate characteristic length scale (chord length or obstacle diameter)
        if hasattr(self.sim_params, 'naca_chord'):
            L_char = self.sim_params.naca_chord
        else:
            L_char = self.geom.radius * 2  # Diameter for cylinder
        
        # Calculate lattice velocity scaling with aggressive approach for less diffusion
        dx = self.grid.lx / self.grid.nx
        
        # Target much higher lattice velocity for less diffusion
        # Push to higher Mach number for better flow physics
        target_U_lattice = 0.25 * jnp.sqrt(cs_squared)  # Mach ~0.25
        
        # Scale physical velocity relative to a reference (e.g., 2.0 m/s)
        reference_velocity = 2.0  # Reference physical velocity
        velocity_scale_factor = physical_velocity / reference_velocity
        
        # Apply scaling to target lattice velocity
        self.U_lattice = target_U_lattice * velocity_scale_factor
        
        # Apply more relaxed stability constraints (Mach < 0.35 for aggressive flow)
        max_U_lattice = 0.35 * jnp.sqrt(cs_squared)
        min_U_lattice = 0.08  # Higher minimum for better flow
        self.U_lattice = jnp.clip(self.U_lattice, min_U_lattice, max_U_lattice)
        
        # Use grid characteristic length as L_lattice
        L_lattice = min(self.grid.nx, self.grid.ny)
        
        # Calculate required lattice viscosity
        nu_lattice = self.U_lattice * L_lattice / target_Re_lattice
        
        # Convert to tau
        tau_target = 0.5 + nu_lattice / cs_squared
        
        # Ensure stability bounds
        tau_min = 0.5  # Theoretical stability limit
        tau_max = 2.0   # Upper bound for stability
        self.lbm_params.tau = jnp.clip(tau_target, tau_min, tau_max)
        self.lbm_params.omega = 1.0 / self.lbm_params.tau
        
        print(f"LBM: Updated parameters - Re={self.flow.Re:.1f}, tau={self.lbm_params.tau:.4f}, "
              f"U_lattice={self.U_lattice:.4f}, omega={self.lbm_params.omega:.4f}")
        
        # Recompile JIT functions with new parameters
        self.recompile_jit()

    def recompile_jit(self):
        """Recompile JIT functions (called when parameters change)"""
        self._jit_cache = {}
        self._step_jit = self.get_step_jit()
        print("LBM: JIT cache cleared and recompiled")
