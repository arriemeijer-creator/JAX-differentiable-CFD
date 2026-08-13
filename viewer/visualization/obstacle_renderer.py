"""
Obstacle renderer for visualization
"""

import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt6 import sip
import logging

logger = logging.getLogger(__name__)

class ObstacleRenderer:
    """Handles rendering of obstacles (cylinder, NACA airfoils)"""

    # Dict mapping obstacle types to their handler methods
    OBSTACLE_HANDLERS = {
        'naca_airfoil': '_draw_naca_outline',
        'cow': '_draw_cow_outline',
        'three_cylinder_array': '_draw_cylinder_array_outline',
        'custom': '_draw_custom_outline',
        'cylinder': '_draw_cylinder_outline',
        'solid_wall': '_draw_solid_wall_outline',
        'urban_map': '_draw_urban_map_outline',
        'tesla_valve': '_draw_tesla_valve_outline',
    }

    def __init__(self, vel_outline, div_outline, vort_outline, scalar_outline, pressure_outline):
        self.vel_outline = vel_outline
        self.div_outline = div_outline  # May be None
        self.vort_outline = vort_outline
        self.scalar_outline = scalar_outline
        self.pressure_outline = pressure_outline
        self.naca_available = self._check_naca_availability()
        self.show_outlines = True  # Default to showing outlines
    
    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False
    
    def update_obstacle_outlines(self, solver, force_update=False):
        """Update obstacle outlines based on current solver geometry"""
        # Rate limit updates to prevent error cascades (unless force_update is True)
        if not force_update:
            if not hasattr(self, '_last_naca_update_time'):
                self._last_naca_update_time = 0
                self._naca_error_count = 0
                self._last_naca_error_designation = None
            
            import time
            current_time = time.time()
            if current_time - self._last_naca_update_time < 0.05:  # Reduced to 50ms for smoother slider updates
                return
            
            self._last_naca_update_time = current_time
        
        if not hasattr(solver, 'sim_params') or solver is None:
            return
        
        try:
            # Clear all outlines first to prevent duplicates when grid changes
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF
            empty_path = QPainterPath()
            
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(empty_path)
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(empty_path)
            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(empty_path)
            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(empty_path)
            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(empty_path)
            
            # Check flow type and obstacle type
            if solver.sim_params.flow_type != 'von_karman':
                return
            
            # Check solver type - don't draw outlines for LBM solver
            solver_type = getattr(solver.sim_params, 'solver_type', 'navier_stokes')
            if solver_type == 'lattice_boltzmann':
                return
            
            # Get obstacle parameters
            obstacle_type = getattr(solver.sim_params, 'obstacle_type', 'cylinder')

            # Use dict mapping to get handler method
            handler_method_name = self.OBSTACLE_HANDLERS.get(obstacle_type, '_draw_cylinder_outline')
            handler = getattr(self, handler_method_name, None)

            if handler:
                handler(solver)
            else:
                logger.warning(f"No handler found for obstacle type: {obstacle_type}, using default cylinder")
                self._draw_cylinder_outline(solver)
        except Exception as e:
            logger.error(f"Error in update_obstacle_outlines: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cylinder_outline(self, solver):
        """Draw cylinder outline"""
        try:
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF
            
            # Get cylinder parameters
            center_x = float(solver.geom.center_x.item()) if hasattr(solver.geom.center_x, 'item') else float(solver.geom.center_x)
            center_y = float(solver.geom.center_y.item()) if hasattr(solver.geom.center_y, 'item') else float(solver.geom.center_y)
            radius = float(solver.geom.radius.item()) if hasattr(solver.geom.radius, 'item') else float(solver.geom.radius)
            
            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(theta)
            y_circle = center_y + radius * np.sin(theta)
            
            # Create QPainterPath from points
            path = QPainterPath()
            if len(x_circle) > 0:
                path.moveTo(QPointF(x_circle[0], y_circle[0]))
                for x, y in zip(x_circle[1:], y_circle[1:]):
                    path.lineTo(QPointF(x, y))
                path.closeSubpath()
            
            # Check if outline items exist and are properly connected to plots
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)

        except Exception as e:
            print(f"Error drawing cylinder outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cylinder_array_outline(self, solver):
        """Draw three-cylinder array outline"""
        try:
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF
            
            # Get cylinder array parameters from sim_params
            cylinder_x = getattr(solver.sim_params, 'cylinder_x', 5.0)
            cylinder_y = getattr(solver.sim_params, 'cylinder_y', solver.grid.ly / 2.0)
            cylinder_diameter = getattr(solver.sim_params, 'cylinder_diameter', 0.5)
            cylinder_spacing = getattr(solver.sim_params, 'cylinder_spacing', 0.5)
            radius = cylinder_diameter / 2.0
            spacing = cylinder_spacing  # Use dynamic spacing from sim_params
            
            # Create circle points for each of the 3 cylinders
            theta = np.linspace(0, 2*np.pi, 100)
            
            # Combine all 3 circles into one polygon
            all_points = []
            for i in range(3):
                center_x_i = cylinder_x + i * spacing
                x_circle = center_x_i + radius * np.cos(theta)
                y_circle = cylinder_y + radius * np.sin(theta)
                all_points.extend([QPointF(x, y) for x, y in zip(x_circle, y_circle)])
            
            # Create QPainterPath from points
            path = QPainterPath()
            if len(all_points) > 0:
                path.moveTo(all_points[0])
                for pt in all_points[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
            
            # Check if outline items exist and are properly connected to plots
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)

        except Exception as e:
            print(f"Error drawing cylinder array outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_cow_outline(self, solver):
        """Draw cow outline using cow.py SDF with matplotlib contour extraction"""
        try:
            from PyQt6.QtGui import QPolygonF, QPen, QBrush, QColor
            from PyQt6.QtCore import QPointF, Qt
            from obstacles.cow import sdf_cow_side
            import jax
            import numpy as np
            from matplotlib import pyplot as plt
            
            # Compute cow position relative to grid bounds
            # Use the same positioning as mask generator
            cow_x = getattr(solver.sim_params, 'cow_x', solver.grid.lx * 0.25)  # 25% of domain width default
            cow_y = getattr(solver.sim_params, 'cow_y', solver.grid.ly * 0.35)  # 35% of domain height default
            
            # Compute scale factor based on grid dimensions relative to reference (20x3.75)
            ref_lx = 20.0
            ref_ly = 3.75
            scale_x = solver.grid.lx / ref_lx
            scale_y = solver.grid.ly / ref_ly
            cow_scale = (scale_x + scale_y) / 2.0  # Average of x and y scaling
            
            # Create a fine grid to sample the SDF
            nx_fine = 400
            ny_fine = 100
            lx_fine = solver.grid.lx
            ly_fine = solver.grid.ly
            
            x_fine = jnp.linspace(0, lx_fine, nx_fine)
            y_fine = jnp.linspace(0, ly_fine, ny_fine)
            X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine, indexing='ij')
            
            # Compute SDF on fine grid with cow_x, cow_y, and scale parameters
            sdf = sdf_cow_side(X_fine, Y_fine, cow_x, cow_y, cow_scale)
            sdf_np = np.array(sdf)
            
            # Transpose to match matplotlib's expectation (ny, nx)
            sdf_np = sdf_np.T
            
            # Extract contour at SDF = 0 using matplotlib
            contours = plt.contour(x_fine, y_fine, sdf_np, levels=[0])
            
            # Collect all contour points using allsegs
            all_points = []
            for seg in contours.allsegs[0]:  # allsegs[0] corresponds to level 0
                for x, y in zip(seg[:, 0], seg[:, 1]):
                    all_points.append(QPointF(float(x), float(y)))
            
            plt.close()  # Close the matplotlib figure
            
            # Create QPainterPath from contour points
            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            if len(all_points) > 0:
                path.moveTo(all_points[0])
                for pt in all_points[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
            
            # Draw using existing outline items (same pattern as cylinder)
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)

        except Exception as e:
            print(f"Error drawing cow outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_naca_outline(self, solver):
        """Draw NACA airfoil outline"""
        if not self.naca_available:
            return
        
        try:
            sim = solver.sim_params
            
            # Get NACA parameters
            designation = sim.naca_airfoil
            chord = sim.naca_chord
            angle = sim.naca_angle
            pos_x = sim.naca_x
            pos_y = sim.naca_y
            
            # Extract digits from designation
            digits = ''.join(filter(str.isdigit, designation))
            
            if len(digits) == 4:
                # 4-digit airfoil
                from obstacles.naca_airfoils import generate_naca_4digit, parse_naca_4digit
                
                try:
                    m, p, t = parse_naca_4digit(designation)
                    # Validate parameters to prevent NaN
                    if p < 0 or p >= 1:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    if t <= 0 or t > 0.5:
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid thickness t={t} for 4-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    
                    # Special handling for symmetric airfoils (p=0)
                    if p == 0.0:
                        # For symmetric airfoils, upper and lower surfaces are just thickness distribution
                        x_norm = np.linspace(0, 1, 100)
                        
                        # Thickness distribution (same as in naca_airfoils.py)
                        yt = 5 * t * (0.2969 * np.sqrt(np.abs(x_norm)) - 0.1260 * x_norm - 
                                      0.3516 * x_norm**2 + 0.2843 * x_norm**3 - 0.1015 * x_norm**4)
                        
                        # For symmetric airfoil: yc = 0, theta = 0
                        xu = x_norm
                        yu = yt  # Upper surface
                        xl = x_norm  
                        yl = -yt  # Lower surface (negative)
                    else:
                        # Use regular NACA generation for cambered airfoils
                        x_norm = np.linspace(0, 1, 100)
                        xu, yu, xl, yl = generate_naca_4digit(jnp.array(x_norm), m, p, t)
                        
                        # Convert to numpy
                        xu, yu = np.array(xu), np.array(yu)
                        xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 4-digit airfoil: {e}")
                    return
                
            elif len(digits) == 5:
                # 5-digit airfoil
                from obstacles.naca_airfoils import generate_naca_5digit, parse_naca_5digit
                
                try:
                    cl, p, m, t = parse_naca_5digit(designation)
                    # Validate parameters to prevent NaN
                    # p=0 is valid for reflexed 5-digit airfoils (e.g., NACA 23012)
                    if p < 0 or p > 1:
                        # Only print error once per designation
                        if self._last_naca_error_designation != designation:
                            print(f"NACA Error: Invalid camber position p={p} for 5-digit airfoil {designation}")
                            self._last_naca_error_designation = designation
                        return
                    x_norm = np.linspace(0, 1, 100)
                    xu, yu, xl, yl = generate_naca_5digit(jnp.array(x_norm), cl, p, m, t)
                    
                    # Convert to numpy
                    xu, yu = np.array(xu), np.array(yu)
                    xl, yl = np.array(xl), np.array(yl)
                except Exception as e:
                    print(f"NACA Error: Failed to generate 5-digit airfoil: {e}")
                    return
            
            # Check for NaN values
            if np.any(np.isnan(xu)) or np.any(np.isnan(yu)) or np.any(np.isnan(xl)) or np.any(np.isnan(yl)):
                # Only print error once per designation
                if self._last_naca_error_designation != designation:
                    print(f"NACA Error: NaN values detected in generated coordinates for {designation}")
                    print(f"  Parameters: digits={digits}, len={len(digits)}")
                    if len(digits) == 4:
                        print(f"  4-digit: m={m}, p={p}, t={t}")
                    elif len(digits) == 5:
                        print(f"  5-digit: cl={cl}, p={p}, m={m}, t={t}")
                    self._last_naca_error_designation = designation
                return
            
            # Scale
            xu, yu = xu * chord, yu * chord
            xl, yl = xl * chord, yl * chord
            
            # Rotate (positive angle - flip Y for screen coordinates where Y increases downward)
            angle_rad = np.radians(angle)
            xu_rot = xu * np.cos(angle_rad) + yu * np.sin(angle_rad)  # Flip Y
            yu_rot = -xu * np.sin(angle_rad) + yu * np.cos(angle_rad)  # Flip Y
            xl_rot = xl * np.cos(angle_rad) + yl * np.sin(angle_rad)  # Flip Y
            yl_rot = -xl * np.sin(angle_rad) + yl * np.cos(angle_rad)  # Flip Y
            
            # Translate (position is leading edge)
            xu_final = xu_rot + pos_x
            yu_final = yu_rot + pos_y
            xl_final = xl_rot + pos_x
            yl_final = yl_rot + pos_y
            
            # Combine upper and lower surfaces
            x_outline = np.concatenate([xu_final, xl_final[::-1], [xu_final[0]]])
            y_outline = np.concatenate([yu_final, yl_final[::-1], [yu_final[0]]])
            
            # Final check for NaN values
            if np.any(np.isnan(x_outline)) or np.any(np.isnan(y_outline)):
                print("NACA Error: NaN values in final outline coordinates")
                return
            
            # Check if outline items are still valid before setting data
            try:
                from PyQt6.QtGui import QPainterPath
                from PyQt6.QtCore import QPointF
                path = QPainterPath()
                if len(x_outline) > 0:
                    path.moveTo(QPointF(x_outline[0], y_outline[0]))
                    for x, y in zip(x_outline[1:], y_outline[1:]):
                        path.lineTo(QPointF(x, y))
                    path.closeSubpath()
                
                if (self.vel_outline is not None and
                    hasattr(self.vel_outline, 'setPath') and
                    not sip.isdeleted(self.vel_outline)):
                    self.vel_outline.setPath(path)

                if (self.div_outline is not None and
                    hasattr(self.div_outline, 'setPath') and
                    not sip.isdeleted(self.div_outline)):
                    self.div_outline.setPath(path)

                if (self.vort_outline is not None and
                    hasattr(self.vort_outline, 'setPath') and
                    not sip.isdeleted(self.vort_outline)):
                    self.vort_outline.setPath(path)

                if (self.scalar_outline is not None and
                    hasattr(self.scalar_outline, 'setPath') and
                    not sip.isdeleted(self.scalar_outline)):
                    self.scalar_outline.setPath(path)

                if (self.pressure_outline is not None and
                    hasattr(self.pressure_outline, 'setPath') and
                    not sip.isdeleted(self.pressure_outline)):
                    self.pressure_outline.setPath(path)
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    print("Warning: Outline plot items deleted during cleanup")
                else:
                    raise
            except Exception as e:
                print(f"Error drawing NACA outline: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error drawing NACA outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_custom_outline(self, solver):
        """Draw custom obstacle outline using PNG SDF field or custom mask with matplotlib contour extraction"""
        try:
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF
            import numpy as np
            import jax.numpy as jnp
            
            # Try to import matplotlib with error handling for DLL loading issues
            try:
                from matplotlib import pyplot as plt
            except ImportError as e:
                print(f"Warning: Matplotlib import failed ({e}), skipping custom obstacle outline drawing")
                return
            
            # Check for PNG SDF field first
            sdf_field = getattr(solver.sim_params, 'sdf_field', None)
            if sdf_field is not None:
                # PNG mask case - use the SDF field directly
                # Convert to numpy if needed
                if hasattr(sdf_field, 'shape'):
                    sdf_np = np.array(sdf_field)
                else:
                    sdf_np = sdf_field
                
                # Get grid coordinates
                X = np.array(solver.grid.X)
                Y = np.array(solver.grid.Y)
                
                # Debug: print shapes
                print(f"DEBUG PNG outline: X shape={X.shape}, Y shape={Y.shape}, SDF shape={sdf_np.shape}")
                
                # Ensure arrays are compatible for matplotlib
                # Matplotlib expects x, y to define the coordinate system
                # and Z to have shape (ny, nx) for contour plotting
                if X.shape == Y.shape and X.shape == sdf_np.shape:
                    # Grid is already in correct format
                    x_coords = X[:, 0]  # Extract x coordinates (first column)
                    y_coords = Y[0, :]  # Extract y coordinates (first row)
                    Z_for_contour = sdf_np.T  # Transpose for matplotlib (ny, nx)
                else:
                    # Extract coordinate vectors from meshgrid
                    x_coords = np.array(solver.grid.x)
                    y_coords = np.array(solver.grid.y)
                    Z_for_contour = sdf_np.T
                
                print(f"DEBUG PNG outline: x_coords shape={x_coords.shape}, y_coords shape={y_coords.shape}, Z shape={Z_for_contour.shape}")
                
                # Extract contour at SDF = 0 (the boundary between solid and fluid)
                try:
                    contours = plt.contour(x_coords, y_coords, Z_for_contour, levels=[0])
                except Exception as e:
                    print(f"Error in PNG contour extraction: {e}")
                    # Try with transposed coordinates as fallback
                    try:
                        contours = plt.contour(y_coords, x_coords, sdf_np, levels=[0])
                    except Exception as e2:
                        print(f"Fallback contour also failed: {e2}")
                        return
                
            else:
                # Fallback to freeform drawing mask
                custom_mask = getattr(solver.sim_params, 'custom_mask', None)
                if custom_mask is None:
                    return
                
                # Get obstacle center position from sliders
                center_x = getattr(solver.sim_params, 'custom_x', solver.grid.lx * 0.25)
                center_y = getattr(solver.sim_params, 'custom_y', solver.grid.ly * 0.5)
                
                # Scale the custom obstacle to fit in the domain while preserving aspect ratio
                # Use the smaller dimension to determine scale, so the drawing fits
                mask_height, mask_width = custom_mask.shape
                
                # Calculate scale to fit in domain (use 60% of the smaller dimension)
                domain_min_dim = min(solver.grid.lx, solver.grid.ly)
                scale = domain_min_dim * 0.6
                
                # Use same scale for both dimensions to preserve aspect ratio
                scale_x = scale
                scale_y = scale
                
                # Calculate offset to center the obstacle at the specified position
                # offset is the center position (matches solver)
                offset_x = center_x
                offset_y = center_y
                
                # Create a fine grid to sample the SDF
                nx_fine = 400
                ny_fine = 100
                lx_fine = solver.grid.lx
                ly_fine = solver.grid.ly
                
                x_fine = jnp.linspace(0, lx_fine, nx_fine)
                y_fine = jnp.linspace(0, ly_fine, ny_fine)
                X_fine, Y_fine = jnp.meshgrid(x_fine, y_fine, indexing='ij')
                
                # Compute the custom SDF/mask on the fine grid
                from obstacles.freeform_drawer import create_freeform_mask_smooth
                # Use a sharp transition (same as in freeform_drawer.py) to prevent visualization artifacts
                mask_fine = create_freeform_mask_smooth(X_fine, Y_fine, custom_mask,
                                                       scale_x=scale_x, scale_y=scale_y,
                                                       offset_x=offset_x, offset_y=offset_y,
                                                       smooth_width=0.001)
                
                # Convert to numpy and transpose for matplotlib
                mask_np = np.array(mask_fine).T
                
                # Extract contour at mask = 0.5 (the boundary between solid and fluid)
                contours = plt.contour(x_fine, y_fine, mask_np, levels=[0.5])
            
            # Collect all contour points - handle multiple disconnected regions
            path = QPainterPath()
            
            # Process each contour segment as a separate subpath to avoid connecting separate regions
            if hasattr(contours, 'allsegs') and len(contours.allsegs) > 0:
                for seg in contours.allsegs[0]:  # allsegs[0] corresponds to level 0
                    if len(seg) > 0:
                        # Start a new subpath for each disconnected region
                        path.moveTo(QPointF(float(seg[0, 0]), float(seg[0, 1])))
                        for i in range(1, len(seg)):
                            path.lineTo(QPointF(float(seg[i, 0]), float(seg[i, 1])))
                        path.closeSubpath()
            
            plt.close()  # Close the matplotlib figure
            
            # Draw using existing outline items
            if (self.vel_outline is not None and
                hasattr(self.vel_outline, 'setPath') and
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)

            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)
                
        except Exception as e:
            print(f"Error drawing custom obstacle outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_solid_wall_outline(self, solver):
        """Draw solid wall outline"""
        try:
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF

            # Get wall parameters from sim_params
            import jax.numpy as jnp
            y_min = float(jnp.min(solver.grid.Y))
            y_max = float(jnp.max(solver.grid.Y))
            y_range = y_max - y_min

            wall_x = solver.grid.lx * getattr(solver.sim_params, 'solid_wall_x', 0.25)
            wall_thickness = solver.grid.lx * 0.02  # 2% of domain width
            wall_y_start = y_min + y_range * getattr(solver.sim_params, 'solid_wall_y_bottom', 0.0)
            wall_y_end = y_min + y_range * getattr(solver.sim_params, 'solid_wall_y_top', 0.5)
            
            # Create thin wall outline points (rectangle)
            points = [
                QPointF(wall_x - wall_thickness/2, wall_y_start),  # Bottom-left corner
                QPointF(wall_x + wall_thickness/2, wall_y_start),  # Bottom-right corner
                QPointF(wall_x + wall_thickness/2, wall_y_end),    # Top-right corner
                QPointF(wall_x - wall_thickness/2, wall_y_end),    # Top-left corner
                QPointF(wall_x - wall_thickness/2, wall_y_start)   # Close the polygon
            ]
            
            # Create QPainterPath from points
            path = QPainterPath()
            if len(points) > 0:
                path.moveTo(points[0])
                for pt in points[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
            
            # Draw using existing outline items (same pattern as cylinder)
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)

        except Exception as e:
            print(f"Error drawing solid wall outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_urban_map_outline(self, solver):
        """Draw urban map outline using SDF field with matplotlib contour extraction"""
        try:
            from PyQt6.QtGui import QPainterPath, QPainter
            from PyQt6.QtCore import QPointF
            from matplotlib import pyplot as plt
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.contour import QuadContourSet
            
            import numpy as np
            
            # Check if SDF field exists
            sdf_field = getattr(solver.sim_params, 'sdf_field', None)
            if sdf_field is None:
                print("No SDF field found for urban map outline")
                return
            
            # Ensure sdf_field is numpy array
            if not isinstance(sdf_field, np.ndarray):
                print(f"DEBUG: sdf_field is {type(sdf_field)}, converting to numpy array")
                # Handle tuple case (from individual_sdfs return)
                if isinstance(sdf_field, tuple):
                    sdf_field = np.array(sdf_field)
                    print(f"DEBUG: Converted tuple to numpy array with shape {sdf_field.shape}")
                elif len(sdf_field.shape) == 2:
                    # Combined SDF case (2D array)
                    pass
                elif len(sdf_field.shape) == 3:
                    # List of individual SDFs case
                    individual_sdfs = sdf_field
                    print(f"DEBUG: Using {len(individual_sdfs)} individual SDFs from list")
                else:
                    print(f"DEBUG: Unexpected sdf_field shape: {sdf_field.shape}")
            
            print(f"Drawing urban map outline from SDF shape: {sdf_field.shape}")
            if hasattr(sdf_field, 'array'):
                sdf_field = np.array(sdf_field)
            
            # Debug SDF field statistics
            sdf_min = np.min(sdf_field)
            sdf_max = np.max(sdf_field)
            sdf_mean = np.mean(sdf_field)
            negative_count = np.sum(sdf_field < 0)
            print(f"SDF stats: min={sdf_min:.3f}, max={sdf_max:.3f}, mean={sdf_mean:.3f}, negative_cells={negative_count}")
            
            # Check for cached polygon contours (extracted once during loading)
            cached_polygons = getattr(solver.sim_params, 'urban_map_polygons', [])
            
            if cached_polygons and len(cached_polygons) > 0:
                print(f"DEBUG: Using {len(cached_polygons)} cached building polygons")
                all_polygons = cached_polygons
            else:
                # Fallback: check for individual building SDFs and extract contours
                individual_sdfs = getattr(solver.sim_params, 'individual_sdfs', [])
                
                if individual_sdfs and len(individual_sdfs) > 0:
                    print(f"DEBUG: Extracting contours from {len(individual_sdfs)} individual building SDFs")
                    all_polygons = []
                    
                    for i, building_sdf in enumerate(individual_sdfs):
                        if building_sdf is None or not isinstance(building_sdf, np.ndarray):
                            continue
                        
                        # Create separate figure for each building
                        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
                        fig_ind.patch.set_facecolor('none')
                        ax_ind.set_facecolor('none')
                        ax_ind.set_aspect('equal')
                        
                        nx, ny = building_sdf.shape
                        x = np.linspace(0, solver.grid.lx, nx)
                        y = np.linspace(0, solver.grid.ly, ny)
                        
                        # Extract contour for this building only with higher resolution and smoothing
                        building_contours = ax_ind.contour(x, y, building_sdf.T, levels=[0], linewidths=2, colors='red')
                        # Apply cubic spline interpolation to smooth the contour vertices
                        if hasattr(building_contours, 'collections'):
                            for collection in building_contours.collections:
                                for path in collection.get_paths():
                                    vertices = path.vertices
                                    if len(vertices) > 3:
                                        # Use scipy's spline interpolation for smoothing
                                        try:
                                            from scipy.interpolate import splprep, splev
                                            # Parameterize the path
                                            tck, u = splprep([vertices[:, 0], vertices[:, 1]], s=0.1, per=True)
                                            # Evaluate with more points for smoother curve
                                            u_new = np.linspace(0, 1, len(vertices) * 2)
                                            x_smooth, y_smooth = splev(u_new, tck)
                                            # Replace the path vertices with smoothed ones
                                            path.vertices = np.column_stack([x_smooth, y_smooth])
                                        except ImportError:
                                            # Fallback: simple moving average if scipy not available
                                            pass
                                        except Exception as e:
                                            print(f"Warning: Spline smoothing failed: {e}")
                                            pass
                        
                        # Extract contour coordinates for this building
                        if hasattr(building_contours, 'collections'):
                            for collection in building_contours.collections:
                                for path in collection.get_paths():
                                    vertices = []
                                    for vertex in path.vertices:
                                        px = vertex[0]
                                        py = vertex[1]
                                        vertices.append((px, py))
                                    
                                    if len(vertices) >= 3:
                                        all_polygons.append(vertices)
                                        print(f"DEBUG: Building {i}: extracted contour with {len(vertices)} vertices")
                        else:
                            for path in building_contours.get_paths():
                                vertices = []
                                for vertex in path.vertices:
                                    px = vertex[0]
                                    py = vertex[1]
                                    vertices.append((px, py))
                                
                                if len(vertices) >= 3:
                                    all_polygons.append(vertices)
                                    print(f"DEBUG: Building {i}: extracted contour with {len(vertices)} vertices")
                        
                        plt.close(fig_ind)
                else:
                    print(f"DEBUG: No individual SDFs found, using combined SDF")
                
                # Create matplotlib figure for contour extraction
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                ax.set_aspect('equal')
                
                # Extract contour at SDF = 0 (building boundaries)
                nx, ny = sdf_field.shape
                x = np.linspace(0, solver.grid.lx, nx)
                y = np.linspace(0, solver.grid.ly, ny)
                
                # Create contour at zero level (building boundaries) with smoothing
                contours = ax.contour(x, y, sdf_field.T, levels=[0], linewidths=2, colors='red')
                print(f"Contour extraction completed, contours object: {type(contours)}")
                
                # Apply spline smoothing to combined contours as well
                if hasattr(contours, 'collections'):
                    for collection in contours.collections:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 3:
                                try:
                                    from scipy.interpolate import splprep, splev
                                    tck, u = splprep([vertices[:, 0], vertices[:, 1]], s=0.1, per=True)
                                    u_new = np.linspace(0, 1, len(vertices) * 2)
                                    x_smooth, y_smooth = splev(u_new, tck)
                                    path.vertices = np.column_stack([x_smooth, y_smooth])
                                except Exception as e:
                                    pass
                
                # Extract contour coordinates
                all_polygons = []
                # Handle different matplotlib versions
                if hasattr(contours, 'collections'):
                    # Newer matplotlib versions
                    for collection in contours.collections:
                        for path in collection.get_paths():
                            vertices = []
                            for vertex in path.vertices:
                                # Convert from matplotlib coordinates to display coordinates
                                px = vertex[0]  # x coordinate
                                py = vertex[1]  # y coordinate (no flip needed)
                                vertices.append((px, py))
                            
                            if len(vertices) >= 3:
                                all_polygons.append(vertices)
                else:
                    # Older matplotlib versions
                    for path in contours.get_paths():
                        vertices = []
                        for vertex in path.vertices:
                            # Convert from matplotlib coordinates to display coordinates
                            px = vertex[0]  # x coordinate
                            py = vertex[1]  # y coordinate (no flip needed)
                            vertices.append((px, py))
                        
                        if len(vertices) >= 3:
                            all_polygons.append(vertices)
                
                plt.close(fig)
            
            # Build QPainterPath with separate subpaths for each building
            # Each moveTo() starts a new disconnected subpath
            print(f"DEBUG: Found {len(all_polygons)} contour polygons")
            from PyQt6.QtCore import QPointF
            from PyQt6.QtGui import QPainterPath
            
            path = QPainterPath()
            
            for i, polygon_points in enumerate(all_polygons):
                x_coords = [p[0] for p in polygon_points]
                y_coords = [p[1] for p in polygon_points]
                print(f"DEBUG: Polygon {i}: {len(polygon_points)} vertices, bounds: x=[{min(x_coords):.1f}, {max(x_coords):.1f}], y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
                
                # Use direct coordinates like cylinder - no transformations
                screen_points = []
                for px, py in polygon_points:
                    screen_points.append(QPointF(px, py))
                
                print(f"DEBUG: Raw polygon coordinates: x_range=[{min(x_coords):.1f}, {max(x_coords):.1f}], y_range=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
                
                # Get actual plot bounds from solver
                plot_x_min, plot_x_max = 0.0, solver.grid.lx
                plot_y_min, plot_y_max = 0.0, solver.grid.ly
                if min(x_coords) < plot_x_min or max(x_coords) > plot_x_max:
                    print(f"WARNING: Polygon {i} x-coordinates outside plot bounds!")
                if min(y_coords) < plot_y_min or max(y_coords) > plot_y_max:
                    print(f"WARNING: Polygon {i} y-coordinates outside plot bounds!")
                
                # Add as separate subpath: moveTo for first point, lineTo for rest
                if len(screen_points) > 0:
                    path.moveTo(screen_points[0])
                    for pt in screen_points[1:]:
                        path.lineTo(pt)
                    path.closeSubpath()
            
            # Draw using existing outline items
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                print(f"DEBUG: Setting path with {len(all_polygons)} polygons to vel_outline")
                
                # Debug: Check plot aspect ratio vs data aspect ratio
                data_aspect_ratio = solver.grid.lx / solver.grid.ly
                print(f"DEBUG: Data aspect ratio: {data_aspect_ratio:.2f} (width/height)")
                
                # Use identical approach to cylinder - simple direct coordinates
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)
                
                # Debug: Check if path bounds match expected
                path_bounds = path.boundingRect()
                path_aspect_ratio = path_bounds.width() / path_bounds.height()
                print(f"DEBUG: Path bounding rect: x={path_bounds.x():.1f}, y={path_bounds.y():.1f}, w={path_bounds.width():.1f}, h={path_bounds.height():.1f}")
                print(f"DEBUG: Path aspect ratio: {path_aspect_ratio:.2f}")
                print(f"DEBUG: Outline item position: x={self.vel_outline.x():.1f}, y={self.vel_outline.y():.1f}")
                print(f"DEBUG: Outline item scale: x={self.vel_outline.scale():.1f}, y={self.vel_outline.scale():.1f}")
                print(f"DEBUG: Outline item rotation: {self.vel_outline.rotation():.1f}")
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)
            
            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)
            
            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)
            
            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)
            
            print(f"Drew {len(all_polygons)} building outlines from urban map")
            
        except Exception as e:
            print(f"Error drawing urban map outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_tesla_valve_outline(self, solver):
        """Draw Tesla valve outline by drawing individual diagonal branches separately"""
        try:
            from PyQt6.QtGui import QPainterPath
            from PyQt6.QtCore import QPointF
            import numpy as np
            import math
            
            # Get Tesla valve parameters from sim_params
            num_stages = getattr(solver.sim_params, 'tesla_valve_stages', 3)
            stage_length = getattr(solver.sim_params, 'tesla_valve_stage_length', 1.5)
            main_width = getattr(solver.sim_params, 'tesla_valve_main_width', 0.4)
            branch_width = getattr(solver.sim_params, 'tesla_valve_branch_width', 0.2)
            branch_angle = getattr(solver.sim_params, 'tesla_valve_branch_angle', 35.0)
            diagonal_length = getattr(solver.sim_params, 'tesla_valve_diagonal_length', 0.4)
            
            # Ensure branch_angle is always in degrees for consistent cache key
            if isinstance(branch_angle, float) and branch_angle > 2 * math.pi:
                # Already in degrees
                pass
            elif isinstance(branch_angle, float) and branch_angle <= 2 * math.pi:
                # Convert from radians to degrees
                branch_angle = math.degrees(branch_angle)
            
            # Get position from sim_params (same as mask generator)
            valve_x = getattr(solver.sim_params, 'tesla_valve_x', solver.grid.lx * 0.25)
            valve_y = getattr(solver.sim_params, 'tesla_valve_y', solver.grid.ly * 0.5)
            
            # Create a cache key to avoid regenerating outline every frame
            cache_key = (num_stages, stage_length, main_width, branch_width, 
                         branch_angle, diagonal_length, valve_x, valve_y,
                         solver.grid.lx, solver.grid.ly)
            
            # Check if we have cached outline
            if not hasattr(self, '_tesla_valve_outline_cache'):
                self._tesla_valve_outline_cache = {}
            
            if cache_key in self._tesla_valve_outline_cache:
                path = self._tesla_valve_outline_cache[cache_key]
            else:
                # Generate outline only when parameters change
                path = self._generate_tesla_valve_outline(solver, num_stages, stage_length, 
                                                        main_width, branch_width, branch_angle,
                                                        diagonal_length, valve_x, valve_y)
                self._tesla_valve_outline_cache[cache_key] = path
            
            # Draw using existing outline items (same pattern as cylinder)
            if (self.vel_outline is not None and 
                hasattr(self.vel_outline, 'setPath') and 
                not sip.isdeleted(self.vel_outline)):
                self.vel_outline.setPath(path)
                self.vel_outline.setVisible(self.show_outlines)
            
            if (self.div_outline is not None and
                hasattr(self.div_outline, 'setPath') and
                not sip.isdeleted(self.div_outline)):
                self.div_outline.setPath(path)

            if (self.vort_outline is not None and
                hasattr(self.vort_outline, 'setPath') and
                not sip.isdeleted(self.vort_outline)):
                self.vort_outline.setPath(path)

            if (self.scalar_outline is not None and
                hasattr(self.scalar_outline, 'setPath') and
                not sip.isdeleted(self.scalar_outline)):
                self.scalar_outline.setPath(path)

            if (self.pressure_outline is not None and
                hasattr(self.pressure_outline, 'setPath') and
                not sip.isdeleted(self.pressure_outline)):
                self.pressure_outline.setPath(path)

        except Exception as e:
            print(f"Error drawing Tesla valve outline: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_tesla_valve_outline(self, solver, num_stages, stage_length, main_width, 
                                     branch_width, branch_angle, diagonal_length, valve_x, valve_y):
        """Generate Tesla valve outline using rotated rectangles matching the mask"""
        from PyQt6.QtGui import QPainterPath
        from PyQt6.QtCore import QPointF
        import numpy as np
        import math
        
        def draw_rotated_rectangle(path, cx, cy, width, height, angle):
            """Draw a rotated rectangle as part of the path"""
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Rectangle corners relative to center
            corners = [
                (-width/2, -height/2),
                (width/2, -height/2),
                (width/2, height/2),
                (-width/2, height/2)
            ]
            
            # Rotate and translate corners
            rotated_points = []
            for dx, dy in corners:
                x = cx + dx * cos_a - dy * sin_a
                y = cy + dx * sin_a + dy * cos_a
                rotated_points.append(QPointF(x, y))
            
            # Add rectangle to path
            path.moveTo(rotated_points[0])
            for point in rotated_points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        
        # Create QPainterPath
        path = QPainterPath()
        
        # Calculate total width and stage positions
        total_width = num_stages * stage_length
        
        for i in range(num_stages):
            # Calculate stage position using same logic as mask generation
            stage_start = valve_x - total_width/2 + i * stage_length
            
            # Diagonal branch parameters (same as mask generation)
            branch_start = stage_start + stage_length * 0.3
            # Fixed base point at branch_start, extend outward when diagonal_length increases
            branch_base_x = branch_start
            branch_len = diagonal_length
            
            # Top diagonal branch - starts from top boundary
            # Position rectangle so its base edge starts at branch_base_x
            # The rectangle center is offset by half the length along the diagonal direction
            top_y = valve_y + solver.grid.ly/2
            top_center_x = branch_base_x + (branch_len/2) * math.cos(math.radians(branch_angle))
            top_center_y = -(branch_len/2) * math.sin(math.radians(branch_angle))  # Negative because top branch goes down
            draw_rotated_rectangle(path, top_center_x, top_y + top_center_y, branch_len, branch_width, branch_angle)
            
            # Bottom diagonal branch - starts from bottom boundary  
            # Position rectangle so its base edge starts at branch_base_x
            # The rectangle center is offset by half the length along the diagonal direction
            bottom_y = valve_y - solver.grid.ly/2
            bottom_center_x = branch_base_x + (branch_len/2) * math.cos(math.radians(branch_angle))
            bottom_center_y = (branch_len/2) * math.sin(math.radians(branch_angle))  # Positive because bottom branch goes up
            draw_rotated_rectangle(path, bottom_center_x, bottom_y + bottom_center_y, branch_len, branch_width, -branch_angle)
        
        return path


