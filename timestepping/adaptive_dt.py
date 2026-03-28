"""
Adaptive timestep controller for CFD simulation
"""
import jax
import jax.numpy as jnp
import numpy as np


class AdaptiveDtController:
    """Handles adaptive timestep control with safety limits"""
    
    def __init__(self, flow_type: str = 'von_karman', dt_min: float = 1e-6, dt_max: float = 0.005):
        self.flow_type = flow_type
        self.dt_min = dt_min
        self.dt_max = dt_max
        self._dt_adjust_counter = 0
        self._dt_overshoot_count = 0
        self._cfl_history = []
        self._last_dt = None
        self._emergency_reductions = 0
        
    def get_initial_dt(self, U_inf: float, dx: float, dy: float) -> float:
        """Estimate a safe initial timestep based on grid and flow"""
        min_spacing = min(dx, dy)
        
        # MUCH MORE CONSERVATIVE estimates for von Karman
        if self.flow_type == 'von_karman':
            # Use CFL = 0.1 for safety at startup (NOT 0.3!)
            estimated_dt = 0.1 * min_spacing / U_inf
        elif self.flow_type == 'taylor_green':
            estimated_dt = min_spacing / (4.0 * U_inf)
        elif self.flow_type == 'backward_step':
            estimated_dt = min_spacing / (6.0 * U_inf)
        elif self.flow_type == 'lid_driven_cavity':
            estimated_dt = min_spacing / (6.0 * U_inf)
        else:
            estimated_dt = min_spacing / (5.0 * U_inf)
        
        # Clamp to bounds
        estimated_dt = max(self.dt_min, min(estimated_dt, self.dt_max))
        
        return float(estimated_dt)
    
    def update_adaptive_dt(self, u, v, current_dt, dx, dy, U_inf, max_cfl=0.3):
        """Robust adaptive timestep update with explosion prevention"""
        
        # Initialize counters
        self._dt_adjust_counter += 1
        self._last_dt = current_dt
        
        # Only adjust every 50 iterations for stability (much more conservative)
        if self._dt_adjust_counter % 50 != 0:
            return current_dt
        
        # Calculate current CFL
        try:
            # Convert to numpy for safety
            u_np = np.array(u)
            v_np = np.array(v)
            max_vel = float(np.max(np.sqrt(u_np**2 + v_np**2)))
            
            if max_vel < 1e-10:
                max_vel = 1e-10
                
            cfl_x = max_vel * current_dt / dx
            cfl_y = max_vel * current_dt / dy
            current_cfl = max(cfl_x, cfl_y)
        except Exception as e:
            print(f"CFL calculation error: {e}")
            return current_dt
        
        # Store CFL history
        self._cfl_history.append(current_cfl)
        if len(self._cfl_history) > 10:
            self._cfl_history.pop(0)
        
        # Calculate average CFL for smoothing
        avg_cfl = np.mean(self._cfl_history) if self._cfl_history else current_cfl
        
        # Flow-type specific parameters (MUCH MORE CONSERVATIVE)
        if self.flow_type == 'von_karman':
            safe_cfl = 0.15          # Target CFL (very conservative)
            max_allowed_cfl = 0.25   # Hard limit (was 0.35)
            dt_growth_limit = 1.01   # Only 1% growth max (was 1.05)
            dt_reduction_factor = 0.85  # Reduce by 15% when needed
            
        elif self.flow_type == 'backward_step':
            safe_cfl = 0.2
            max_allowed_cfl = 0.3
            dt_growth_limit = 1.02
            dt_reduction_factor = 0.88
            
        elif self.flow_type == 'lid_driven_cavity':
            safe_cfl = 0.25
            max_allowed_cfl = 0.35
            dt_growth_limit = 1.03
            dt_reduction_factor = 0.9
            
        elif self.flow_type == 'channel_flow':
            safe_cfl = 0.3
            max_allowed_cfl = 0.4
            dt_growth_limit = 1.05
            dt_reduction_factor = 0.92
            
        else:  # taylor_green
            safe_cfl = 0.4
            max_allowed_cfl = 0.5
            dt_growth_limit = 1.08
            dt_reduction_factor = 0.95
        
        new_dt = current_dt
        
        # Emergency: CFL too high - immediate reduction
        if current_cfl > max_allowed_cfl:
            # Emergency reduction
            new_dt = current_dt * 0.5
            self._dt_overshoot_count += 1
            self._emergency_reductions += 1
            
            print(f"⚠️ EMERGENCY: CFL={current_cfl:.3f} > {max_allowed_cfl:.3f}, "
                  f"dt: {current_dt:.6f} → {new_dt:.6f}")
            
            # After 5 emergency reductions, switch to fixed dt mode
            if self._emergency_reductions > 5:
                print("⚠️⚠️ CRITICAL: Too many emergency reductions!")
                print("Switching to fixed dt mode for stability...")
                return current_dt * 0.5  # Signal to switch to fixed dt
        
        elif current_cfl > safe_cfl:
            # CFL too high - reduce dt
            # Calculate reduction factor based on how far above safe CFL
            reduction_needed = safe_cfl / (current_cfl + 1e-10)
            reduction_factor = max(0.7, min(0.95, reduction_needed))
            new_dt = current_dt * reduction_factor
            self._dt_overshoot_count = max(0, self._dt_overshoot_count - 1)
            
            if reduction_factor < 0.95:
                print(f"⚠️ dt reduced: {current_dt:.6f} → {new_dt:.6f} "
                      f"(CFL={current_cfl:.3f} > {safe_cfl:.3f})")
            
        elif current_cfl < safe_cfl * 0.5 and self._dt_overshoot_count == 0:
            # CFL very low - allow VERY cautious dt growth
            # Only increase if we haven't had recent overshoots
            growth_factor = min(dt_growth_limit, 1 + (safe_cfl * 0.05))
            new_dt = current_dt * growth_factor
            self._dt_overshoot_count = max(0, self._dt_overshoot_count - 1)
            
            if growth_factor > 1.01:
                print(f"✓ dt increased: {current_dt:.6f} → {new_dt:.6f} "
                      f"(CFL={current_cfl:.3f})")
            
        else:
            # CFL in good range - minimal adjustment
            new_dt = current_dt
            self._dt_overshoot_count = max(0, self._dt_overshoot_count - 1)
        
        # Apply flow-type specific maximum dt
        if self.flow_type == 'von_karman':
            # Calculate physically reasonable max dt
            # At Re=150, shedding period ~5-6 seconds
            shedding_period = 6.0
            # Need at least 50 steps per shedding period for accuracy
            max_phys_dt = shedding_period / 100  # 100 steps per period
            dt_max = min(self.dt_max, max_phys_dt)
        else:
            dt_max = self.dt_max
        
        # Apply bounds
        new_dt = max(self.dt_min, min(new_dt, dt_max))
        
        # Only update if change is significant (>1%)
        if abs(new_dt - current_dt) / current_dt > 0.01:
            return float(new_dt)
        else:
            return current_dt
    
    def check_stability(self, u, v, U_inf):
        """Check if simulation is numerically stable"""
        try:
            # Convert to numpy for safety
            u_np = np.array(u)
            v_np = np.array(v)
            
            # Check for NaN or Inf
            if not np.all(np.isfinite(u_np)) or not np.all(np.isfinite(v_np)):
                return False
            
            # Check for velocity explosion (should be < 50x U_inf for stability)
            max_vel = float(np.max(np.sqrt(u_np**2 + v_np**2)))
            if max_vel > 50 * U_inf:
                print(f"⚠️ Velocity too high: {max_vel:.2f} > {50*U_inf:.2f}")
                return False
            
            # Check for kinetic energy explosion
            ke = 0.5 * np.mean(u_np**2 + v_np**2)
            if ke > 100 * (U_inf**2):
                print(f"⚠️ KE too high: {ke:.2f} > {100*U_inf**2:.2f}")
                return False
            
            return True
        except:
            return False
    
    def reset_counters(self):
        """Reset counters when flow is reset"""
        self._dt_adjust_counter = 0
        self._dt_overshoot_count = 0
        self._cfl_history = []
        self._emergency_reductions = 0


# Wrapper functions for compatibility with the main solver
def update_adaptive_dt(solver_instance):
    """Wrapper function to update adaptive dt"""
    if not solver_instance.sim_params.adaptive_dt:
        return
    
    if solver_instance.adaptive_controller is not None:
        new_dt = solver_instance.adaptive_controller.update_adaptive_dt(
            solver_instance.u, solver_instance.v, solver_instance.dt,
            solver_instance.grid.dx, solver_instance.grid.dy,
            solver_instance.flow.U_inf, solver_instance.sim_params.max_cfl
        )
        
        if new_dt != solver_instance.dt:
            old_dt = solver_instance.dt
            solver_instance.dt = new_dt
            # Recompile JIT functions with new dt
            solver_instance._step_jit = jax.jit(solver_instance._step)
            
            # Only print if significant change
            if abs(new_dt - old_dt) / old_dt > 0.05:
                print(f"dt changed: {old_dt:.6f} → {new_dt:.6f}")


def set_adaptive_dt(solver_instance, max_cfl=None, dt_min=None, dt_max=None):
    """Switch to adaptive timestep mode"""
    solver_instance.sim_params.adaptive_dt = True
    
    # Update controller parameters if provided
    if solver_instance.adaptive_controller is not None:
        if max_cfl is not None:
            solver_instance.sim_params.max_cfl = max_cfl
        if dt_min is not None:
            solver_instance.adaptive_controller.dt_min = dt_min
        if dt_max is not None:
            solver_instance.adaptive_controller.dt_max = dt_max
        
        # Reset counters
        solver_instance.adaptive_controller.reset_counters()
        
        # Reinitialize dt based on current flow
        solver_instance.dt = solver_instance.adaptive_controller.get_initial_dt(
            solver_instance.flow.U_inf, solver_instance.grid.dx, solver_instance.grid.dy
        )
        
        # Recompile JIT functions
        solver_instance._step_jit = jax.jit(solver_instance._step)
        
        print(f"Switched to adaptive mode")
        print(f"  Starting dt = {solver_instance.dt:.6f}")
        print(f"  Max CFL = {solver_instance.sim_params.max_cfl}")
        print(f"  dt range: [{solver_instance.adaptive_controller.dt_min:.6f}, "
              f"{solver_instance.adaptive_controller.dt_max:.6f}]")
    else:
        print("Warning: AdaptiveDtController not available")