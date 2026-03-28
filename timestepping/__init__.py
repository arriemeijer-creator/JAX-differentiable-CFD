"""
Timestepping Module for Differential CFD-ML Framework
=====================================================

This module provides adaptive timestepping controllers for the CFD solver.

Components:
- AdaptiveDtController: CFL-based adaptive timestepping with flow-type specific safety limits
- update_adaptive_dt: Wrapper function for grid-aware timestep updates
- set_adaptive_dt: Wrapper function for switching to adaptive mode

The adaptive timestepping system automatically adjusts timestep size based on:
- Current CFL number
- Flow type specific safety limits
- Grid spacing changes
- Numerical stability monitoring
"""

from .adaptive_dt import AdaptiveDtController, update_adaptive_dt, set_adaptive_dt

__all__ = [
    'AdaptiveDtController',
    'update_adaptive_dt', 
    'set_adaptive_dt'
]
