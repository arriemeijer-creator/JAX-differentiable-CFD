"""
State save/load functionality for AeroJAX solver.

This module provides serialization and deserialization of simulation states,
allowing users to save and restore complete simulation configurations.

Key design decision: Pressure is NOT saved/restored because it's a derived field
computed from velocity divergence. Loading restores velocity fields and lets
the solver recompute pressure naturally on the next timestep.
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any
import jax.numpy as jnp
import numpy as np

from .params import (
    GridParams, FlowParams, FlowConstraints, GeometryParams,
    SimulationParams, SimState
)


class StateIO:
    """Handle saving and loading of simulation states."""
    
    @staticmethod
    def save_state(solver, filepath: str, include_metadata: bool = True) -> None:
        """
        Save complete simulation state to file.
        
        Args:
            solver: BaselineSolver or LBMSolver instance to save
            filepath: Path to save file (.pkl recommended)
            include_metadata: Whether to include additional metadata (history, profiling data)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert JAX arrays to numpy for serialization
        def to_numpy(obj):
            if isinstance(obj, jnp.ndarray):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {k: to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_numpy(item) for item in obj)
            else:
                return obj
        
        # Check solver type and extract state accordingly
        if hasattr(solver, 'state'):
            # BaselineSolver has a state attribute
            sim_state_data = {
                'u': to_numpy(solver.state.u),
                'v': to_numpy(solver.state.v),
                'c': to_numpy(solver.state.c),
                'u_prev': to_numpy(solver.state.u_prev),
                'v_prev': to_numpy(solver.state.v_prev),
                'dt': float(solver.state.dt),
                'iteration': int(solver.state.iteration),
                'grid_type': solver.state.grid_type,
                'integral': float(solver.state.integral),
                'prev_error': float(solver.state.prev_error),
            }
            solver_type = 'baseline'
        else:
            # LBMSolver stores state directly as attributes
            sim_state_data = {
                'u': to_numpy(solver.u),
                'v': to_numpy(solver.v),
                'c': to_numpy(solver.c),
                'u_prev': to_numpy(solver.u_prev),
                'v_prev': to_numpy(solver.v_prev),
                'dt': float(solver.dt),
                'iteration': int(solver.iteration) if hasattr(solver, 'iteration') else 0,
                'grid_type': solver.sim_params.grid_type if hasattr(solver, 'sim_params') else 'collocated',
                'integral': 0.0,  # LBM doesn't use PID controller
                'prev_error': 0.0,
            }
            solver_type = 'lbm'
        
        # Prepare state dictionary
        state_dict = {
            'solver_type': solver_type,
            # Core simulation state
            'sim_state': sim_state_data,
            
            # Parameters
            'grid_params': {
                'nx': solver.grid.nx,
                'ny': solver.grid.ny,
                'lx': float(solver.grid.lx),
                'ly': float(solver.grid.ly),
            },
            'flow_params': {
                'U_inf': float(solver.flow.U_inf),
                'nu': float(solver.flow.nu),
                'Re': float(solver.flow.Re),
                'L_char': float(solver.flow.L_char),
                'constraints': {
                    'lock_U': solver.flow.constraints.lock_U,
                    'lock_nu': solver.flow.constraints.lock_nu,
                    'lock_Re': solver.flow.constraints.lock_Re,
                    'lock_L': solver.flow.constraints.lock_L,
                }
            },
            'geometry_params': {
                'center_x': float(solver.geom.center_x),
                'center_y': float(solver.geom.center_y),
                'radius': float(solver.geom.radius),
            },
            'simulation_params': {
                'eps_multiplier': float(solver.sim_params.eps_multiplier),
                'auto_eps_multiplier': solver.sim_params.auto_eps_multiplier,
                'eps': float(solver.sim_params.eps),
                'solver_type': solver.sim_params.solver_type,
                'advection_scheme': solver.sim_params.advection_scheme,
                'limiter': solver.sim_params.limiter,
                'weno_epsilon': solver.sim_params.weno_epsilon,
                'max_cfl': solver.sim_params.max_cfl,
                'adaptive_dt': solver.sim_params.adaptive_dt,
                'fixed_dt': float(solver.sim_params.fixed_dt),
                'pressure_solver': solver.sim_params.pressure_solver,
                'sor_omega': solver.sim_params.sor_omega,
                'pressure_max_iter': solver.sim_params.pressure_max_iter,
                'pressure_tolerance': solver.sim_params.pressure_tolerance,
                'multigrid_levels': solver.sim_params.multigrid_levels,
                'multigrid_v_cycles': solver.sim_params.multigrid_v_cycles,
                'flow_type': solver.sim_params.flow_type,
                'dt_min': float(solver.sim_params.dt_min),
                'dt_max': float(solver.sim_params.dt_max),
                'grid_type': solver.sim_params.grid_type,
                'obstacle_type': solver.sim_params.obstacle_type,
                'naca_airfoil': solver.sim_params.naca_airfoil,
                'naca_chord': float(solver.sim_params.naca_chord),
                'naca_angle': float(solver.sim_params.naca_angle),
                'naca_x': float(solver.sim_params.naca_x),
                'naca_y': float(solver.sim_params.naca_y),
                'cow_x': float(solver.sim_params.cow_x),
                'cow_y': float(solver.sim_params.cow_y),
                'use_les': solver.sim_params.use_les,
                'les_model': solver.sim_params.les_model,
                'smagorinsky_constant': float(solver.sim_params.smagorinsky_constant),
                'use_scalar': solver.sim_params.use_scalar,
                'scalar_diffusivity': float(solver.sim_params.scalar_diffusivity),
                'nu_h': float(solver.sim_params.nu_h),
                'brinkman_eta': float(solver.sim_params.brinkman_eta),
            },
            
            # Fields
            'mask': to_numpy(solver.mask),
            'sdf': to_numpy(solver.sdf) if hasattr(solver, 'sdf') and solver.sdf is not None else None,
            
            # Solver state (may not exist on LBMSolver)
            'slip_walls': getattr(solver, 'slip_walls', True),
            'nu_hyper_ratio': float(getattr(solver, 'nu_hyper_ratio', 0.0)),
            'enable_scalar_update': getattr(solver, 'enable_scalar_update', True),
            'compute_airfoil_metrics': getattr(solver, 'compute_airfoil_metrics', False),
            'metrics_frame_skip': getattr(solver, 'metrics_frame_skip', 100),
        }
        
        # Optional metadata
        if include_metadata:
            state_dict['metadata'] = {
                'history': to_numpy(solver.history),
                'iteration': solver.iteration,
            }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"State saved to {filepath}")
        print(f"  Iteration: {state_dict['sim_state']['iteration']}")
        print(f"  dt: {state_dict['sim_state']['dt']:.6f}")
        print(f"  Flow type: {state_dict['simulation_params']['flow_type']}")
        print(f"  Re: {state_dict['flow_params']['Re']:.1f}")
    
    @staticmethod
    def load_state(filepath: str) -> Dict[str, Any]:
        """
        Load simulation state from file.
        
        Args:
            filepath: Path to saved state file
            
        Returns:
            Dictionary containing all saved state components
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        print(f"State loaded from {filepath}")
        print(f"  Iteration: {state_dict['sim_state']['iteration']}")
        print(f"  dt: {state_dict['sim_state']['dt']:.6f}")
        print(f"  Flow type: {state_dict['simulation_params']['flow_type']}")
        print(f"  Re: {state_dict['flow_params']['Re']:.1f}")
        
        return state_dict
    
    @staticmethod
    def restore_solver(solver, state_dict: Dict[str, Any]) -> None:
        """
        Restore solver state from loaded state dictionary.
        
        This method:
        1. Restores all parameters
        2. Restores velocity and dye fields
        3. Resets pressure to zero (will be recomputed on next step)
        4. Clears JIT cache to force recompilation
        
        Args:
            solver: BaselineSolver or LBMSolver instance to restore
            state_dict: State dictionary from load_state()
        """
        import jax
        
        # Restore parameters
        sim_state = state_dict['sim_state']
        grid_params = state_dict['grid_params']
        flow_params = state_dict['flow_params']
        geometry_params = state_dict['geometry_params']
        simulation_params = state_dict['simulation_params']
        solver_type = state_dict.get('solver_type', 'baseline')
        
        # Update grid parameters
        solver.grid.nx = grid_params['nx']
        solver.grid.ny = grid_params['ny']
        solver.grid.lx = grid_params['lx']
        solver.grid.ly = grid_params['ly']
        solver.grid.dx = solver.grid.lx / solver.grid.nx
        solver.grid.dy = solver.grid.ly / solver.grid.ny
        solver.grid.x = jnp.linspace(0, solver.grid.lx, solver.grid.nx)
        solver.grid.y = jnp.linspace(0, solver.grid.ly, solver.grid.ny)
        solver.grid.X, solver.grid.Y = jnp.meshgrid(solver.grid.x, solver.grid.y, indexing='ij')
        
        # Update staggered grid coordinates if available
        if hasattr(solver.grid, 'x_u'):
            solver.grid.x_u = jnp.linspace(0, solver.grid.lx, solver.grid.nx + 1)
            solver.grid.y_v = jnp.linspace(0, solver.grid.ly, solver.grid.ny + 1)
            solver.grid.X_u, _ = jnp.meshgrid(solver.grid.x_u, solver.grid.y, indexing='ij')
            _, solver.grid.Y_v = jnp.meshgrid(solver.grid.x, solver.grid.y_v, indexing='ij')
        
        # Update flow parameters
        solver.flow.U_inf = flow_params['U_inf']
        solver.flow.nu = flow_params['nu']
        solver.flow.Re = flow_params['Re']
        solver.flow.L_char = flow_params['L_char']
        solver.flow.constraints.lock_U = flow_params['constraints']['lock_U']
        solver.flow.constraints.lock_nu = flow_params['constraints']['lock_nu']
        solver.flow.constraints.lock_Re = flow_params['constraints']['lock_Re']
        solver.flow.constraints.lock_L = flow_params['constraints']['lock_L']
        
        # Update geometry parameters
        solver.geom.center_x = jnp.array(geometry_params['center_x'])
        solver.geom.center_y = jnp.array(geometry_params['center_y'])
        solver.geom.radius = jnp.array(geometry_params['radius'])
        
        # Update simulation parameters
        for key, value in simulation_params.items():
            if hasattr(solver.sim_params, key):
                setattr(solver.sim_params, key, value)
        
        # Restore fields (convert back to JAX arrays)
        solver.u = jnp.array(sim_state['u'])
        solver.v = jnp.array(sim_state['v'])
        solver.c = jnp.array(sim_state['c'])
        solver.u_prev = jnp.array(sim_state['u_prev'])
        solver.v_prev = jnp.array(sim_state['v_prev'])
        solver.mask = jnp.array(state_dict['mask'])
        solver.sdf = jnp.array(state_dict['sdf']) if state_dict['sdf'] is not None else None
        
        # CRITICAL: Reset pressure to zero - will be recomputed on next step
        # This prevents pressure explosion issues
        solver.current_pressure = jnp.zeros((solver.grid.nx, solver.grid.ny))
        if hasattr(solver, 'p'):
            solver.p = jnp.zeros((solver.grid.nx, solver.grid.ny))
        
        # Restore solver state
        solver.dt = sim_state['dt']
        solver.iteration = sim_state['iteration']
        
        # Restore BaselineSolver-specific state
        if solver_type == 'baseline':
            solver.slip_walls = state_dict['slip_walls']
            solver.nu_hyper_ratio = state_dict['nu_hyper_ratio']
            solver.enable_scalar_update = state_dict['enable_scalar_update']
            solver.compute_airfoil_metrics = state_dict['compute_airfoil_metrics']
            solver.metrics_frame_skip = state_dict['metrics_frame_skip']
            
            # Restore PID controller state if adaptive dt is enabled
            if solver.dt_controller is not None:
                solver.dt_controller.integral = sim_state['integral']
                solver.dt_controller.prev_error = sim_state['prev_error']
            
            # Update SimState object
            solver.state = SimState(
                u=solver.u,
                v=solver.v,
                p=solver.current_pressure,  # Zero pressure - will be recomputed
                u_prev=solver.u_prev,
                v_prev=solver.v_prev,
                c=solver.c,
                dt=solver.dt,
                iteration=solver.iteration,
                grid_type=sim_state['grid_type'],
                integral=sim_state['integral'],
                prev_error=sim_state['prev_error']
            )
        
        # Restore history if available
        if 'metadata' in state_dict:
            solver.history = state_dict['metadata']['history']
            solver.iteration = state_dict['metadata']['iteration']
        
        # CRITICAL: Clear JIT cache to force recompilation with new state
        jax.clear_caches()
        
        # Re-initialize step JIT with new state
        try:
            solver._step_jit = solver.get_step_jit()
            print("Successfully recompiled step function after state load")
        except Exception as e:
            print(f"Warning: Failed to recompile step function: {e}")
            print("The simulation may still work, but performance may be degraded")
        
        print(f"Solver state restored successfully (solver type: {solver_type})")
        print("  Pressure reset to zero (will be recomputed on next step)")
        print("  JIT cache cleared and step function recompiled")
