import sys
import numpy as np
import jax.numpy as jnp
import jax
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox, QMenuBar, QCheckBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
import pyqtgraph as pg
from dataclasses import dataclass
from PIL import Image
import io

try:
    from baseline_clean import (GridParams, FlowParams, GeometryParams, SimulationParams, 
                           CavityGeometryParams, BaselineSolver, compute_forces)
    print("Successfully imported BaselineSolver")
except ImportError as e:
    print(f"Failed to import BaselineSolver: {e}")
    sys.exit(1)

class BaselineViewer(QMainWindow):
    
    def __init__(self, solver: BaselineSolver):
        super().__init__()
        self.solver = solver
        self.is_recording = False
        self.recorded_frames = []
        
        # Pre-compute scale factors for performance
        self.scale_x = self.solver.grid.lx / self.solver.grid.nx
        self.scale_y = self.solver.grid.ly / self.solver.grid.ny
        
        # Info label
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("color: white; background-color: black; padding: 5px;")
        
        # Initialize all attributes to prevent attribute errors
        # Pre-allocate NumPy arrays for history data (better performance)
        self.max_history = 200
        self.time_data = np.zeros(self.max_history)
        self.drag_data = np.zeros(self.max_history)
        self.lift_data = np.zeros(self.max_history)
        self.ke_data = np.zeros(self.max_history)
        self.enst_data = np.zeros(self.max_history)
        self.history_idx = 0
        
        # Initialize image items to prevent attribute errors
        self.vel_img = None
        self.vort_img = None
        self.stream_img = None
        self.pressure_img = None
        
        # Initialize plot items
        self.vel_plot = None
        self.vort_plot = None
        self.stream_plot = None
        self.pressure_plot = None
        self.drag_plot = None
        self.energy_plot = None
        
        # Initialize curves
        self.drag_curve = None
        self.lift_curve = None
        self.ke_curve = None
        self.enst_curve = None
        
        # Performance settings
        self.show_velocity = True
        self.show_vorticity = True
        self.show_streamlines = True
        self.show_pressure = True
        self.show_energy = True
        self.show_forces = True
        self.update_counter = 0
        
        # Create plot widget first
        self.plot_widget = pg.GraphicsLayoutWidget()
        
        # Setup plots
        self.setup_plots()
        
        # Then setup controls
        self.create_control_buttons()
        
        self.setWindowTitle("Baseline Navier-Stokes Solver - Vortex Shedding")
        self.setGeometry(100, 100, 1600, 900)  # Increased size for all plots
        
        self.plot_widget.setBackground('w')
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)  # 30 FPS - optimized for CFD visualization
        
    def create_control_buttons(self):
        """Create control panel with multiple rows"""
        # Create main control widget
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        
        # Row 1: Simulation controls
        row1_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        
        self.start_btn.clicked.connect(self.start_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.reset_btn.clicked.connect(self.reset_simulation)
        
        self.pause_btn.setEnabled(False)
        
        row1_layout.addWidget(self.start_btn)
        row1_layout.addWidget(self.pause_btn)
        row1_layout.addWidget(self.reset_btn)
        row1_layout.addWidget(QLabel("|"))
        
        # Reynolds number control
        re_label = QLabel("Re:")
        self.re_input = QSpinBox()
        self.re_input.setRange(10, 1000)
        self.re_input.setValue(int(self.solver.flow.Re))
        self.re_input.setSingleStep(10)
        self.re_input.setSuffix(" ")
        self.apply_re_btn = QPushButton("Apply Re")
        self.apply_re_btn.clicked.connect(self.apply_reynolds)
        
        row1_layout.addWidget(re_label)
        row1_layout.addWidget(self.re_input)
        row1_layout.addWidget(self.apply_re_btn)
        
        # Row 2: Flow and scheme controls
        row2_layout = QHBoxLayout()
        
        flow_label = QLabel("Flow:")
        self.flow_combo = QComboBox()
        self.flow_combo.addItems(["von_karman", "lid_driven_cavity", "channel_flow", "backward_step", "taylor_green"])
        self.flow_combo.setCurrentText(self.solver.sim_params.flow_type)
        self.flow_combo.currentTextChanged.connect(self.update_grid_options)
        self.apply_flow_btn = QPushButton("Apply Flow")
        self.apply_flow_btn.clicked.connect(self.apply_flow_type)
        
        # Grid size controls
        grid_label = QLabel("Grid:")
        self.grid_combo = QComboBox()
        self.update_grid_options()  # Populate with current flow type options
        self.apply_grid_btn = QPushButton("Apply Grid")
        self.apply_grid_btn.clicked.connect(self.apply_grid_size)
        
        scheme_label = QLabel("Advection:")
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["upwind", "maccormack", "jos_stam", "quick", "weno5", "tvd", "rk3", "spectral"])
        self.scheme_combo.setCurrentText(self.solver.sim_params.advection_scheme)
        self.apply_scheme_btn = QPushButton("Apply Scheme")
        self.apply_scheme_btn.clicked.connect(self.apply_advection_scheme)
        
        row2_layout.addWidget(flow_label)
        row2_layout.addWidget(self.flow_combo)
        row2_layout.addWidget(self.apply_flow_btn)
        row2_layout.addWidget(QLabel("|"))
        row2_layout.addWidget(grid_label)
        row2_layout.addWidget(self.grid_combo)
        row2_layout.addWidget(self.apply_grid_btn)
        row2_layout.addWidget(QLabel("|"))
        row2_layout.addWidget(scheme_label)
        row2_layout.addWidget(self.scheme_combo)
        row2_layout.addWidget(self.apply_scheme_btn)
        
        # Row 3: Pressure solver and dt controls
        row3_layout = QHBoxLayout()
        
        pressure_label = QLabel("Pressure:")
        self.pressure_combo = QComboBox()
        self.pressure_combo.addItems(["jacobi", "fft", "adi", "sor", "gauss_seidel_rb", "cg", "multigrid"])
        self.pressure_combo.setCurrentText(self.solver.sim_params.pressure_solver)
        self.apply_pressure_btn = QPushButton("Apply Pressure")
        self.apply_pressure_btn.clicked.connect(self.apply_pressure_solver)
        
        dt_label = QLabel("dt:")
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.0001, 0.01)
        self.dt_spinbox.setValue(self.solver.dt)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.0001)
        self.dt_spinbox.valueChanged.connect(self.apply_dt)  # Apply dt automatically when value changes
        self.apply_dt_btn = QPushButton("Apply dt")
        self.apply_dt_btn.clicked.connect(self.apply_dt)
        
        # Adaptive dt toggle
        self.adaptive_dt_checkbox = QCheckBox("Adaptive dt")
        self.adaptive_dt_checkbox.setChecked(self.solver.sim_params.adaptive_dt)
        self.adaptive_dt_checkbox.stateChanged.connect(self.toggle_adaptive_dt)
        
        # Enable/disable dt controls based on adaptive mode
        if self.solver.sim_params.adaptive_dt:
            self.dt_spinbox.setEnabled(False)
            self.apply_dt_btn.setEnabled(False)
        
        self.cfl_label = QLabel("CFL: 0.00")
        
        row3_layout.addWidget(pressure_label)
        row3_layout.addWidget(self.pressure_combo)
        row3_layout.addWidget(self.apply_pressure_btn)
        row3_layout.addWidget(QLabel("|"))
        row3_layout.addWidget(dt_label)
        row3_layout.addWidget(self.dt_spinbox)
        row3_layout.addWidget(self.apply_dt_btn)
        row3_layout.addWidget(self.adaptive_dt_checkbox)
        row3_layout.addWidget(self.cfl_label)
        
        # Row 4: Visualization and export controls
        row4_layout = QHBoxLayout()
        
        # Individual plot toggles
        self.show_velocity_checkbox = QCheckBox("Velocity")
        self.show_velocity_checkbox.setChecked(True)
        self.show_velocity_checkbox.stateChanged.connect(self.toggle_velocity)
        
        self.show_vorticity_checkbox = QCheckBox("Vorticity")
        self.show_vorticity_checkbox.setChecked(True)
        self.show_vorticity_checkbox.stateChanged.connect(self.toggle_vorticity)
        
        self.show_streamlines_checkbox = QCheckBox("Streamlines")
        self.show_streamlines_checkbox.setChecked(True)
        self.show_streamlines_checkbox.stateChanged.connect(self.toggle_streamlines)
        
        self.show_pressure_checkbox = QCheckBox("Pressure")
        self.show_pressure_checkbox.setChecked(True)
        self.show_pressure_checkbox.stateChanged.connect(self.toggle_pressure)
        
        self.show_energy_checkbox = QCheckBox("Energy")
        self.show_energy_checkbox.setChecked(True)
        self.show_energy_checkbox.stateChanged.connect(self.toggle_energy)
        
        self.show_forces_checkbox = QCheckBox("Drag/Lift")
        self.show_forces_checkbox.setChecked(True)
        self.show_forces_checkbox.stateChanged.connect(self.toggle_forces)
        
        colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        # Use all available pyqtgraph colormaps
        available_colormaps = [
            # Sequential colormaps
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
            # CET colormaps (sequential)
            'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7',
            'CET-D1', 'CET-D2', 'CET-D3', 'CET-D4', 'CET-D6', 'CET-D7', 'CET-D8', 'CET-D9', 'CET-D10', 'CET-D11', 'CET-D12', 'CET-D13',
            'CET-L1', 'CET-L2', 'CET-L3', 'CET-L4', 'CET-L5', 'CET-L6', 'CET-L7', 'CET-L8', 'CET-L9', 'CET-L10', 'CET-L11', 'CET-L12', 'CET-L13', 'CET-L14', 'CET-L15', 'CET-L16', 'CET-L17', 'CET-L18', 'CET-L19',
            # Diverging colormaps
            'CET-CBC1', 'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2', 'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
            'CET-I1', 'CET-I2', 'CET-I3',
            'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4',
            # Additional options
            'PAL-relaxed', 'PAL-relaxed_bright'
        ]
        self.colormap_combo.addItems(available_colormaps)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        
        # Recording controls
        self.record_btn = QPushButton("Record Video")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.save_btn = QPushButton("Save Video")
        self.save_btn.clicked.connect(self.save_video)
        self.save_btn.setEnabled(False)
        
        row4_layout.addWidget(self.show_velocity_checkbox)
        row4_layout.addWidget(self.show_vorticity_checkbox)
        row4_layout.addWidget(self.show_streamlines_checkbox)
        row4_layout.addWidget(self.show_pressure_checkbox)
        row4_layout.addWidget(self.show_energy_checkbox)
        row4_layout.addWidget(self.show_forces_checkbox)
        row4_layout.addWidget(QLabel("|"))
        row4_layout.addWidget(colormap_label)
        row4_layout.addWidget(self.colormap_combo)
        row4_layout.addWidget(QLabel("|"))
        row4_layout.addWidget(self.export_btn)
        row4_layout.addWidget(self.record_btn)
        row4_layout.addWidget(self.save_btn)
        
        # Add all rows to main layout
        control_layout.addLayout(row1_layout)
        control_layout.addLayout(row2_layout)
        control_layout.addLayout(row3_layout)
        control_layout.addLayout(row4_layout)
        
        control_widget.setLayout(control_layout)
        
        # Create toolbar with controls and info
        toolbar = QWidget()
        toolbar_layout = QVBoxLayout()
        toolbar_layout.addWidget(control_widget)
        toolbar_layout.addWidget(self.info_label)
        toolbar.setLayout(toolbar_layout)
        
        # Add toolbar as main widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.plot_widget)
        
        # Ensure plot widget is visible and properly sized
        self.plot_widget.show()
        self.plot_widget.setMinimumSize(800, 600)
        
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def apply_dt(self):
        """Apply new timestep in real-time"""
        new_dt = self.dt_spinbox.value()
        
        # Switch to fixed dt mode without reset
        self.solver.set_fixed_dt(new_dt)
        
        # Update spinbox to match actual dt
        self.dt_spinbox.setValue(self.solver.dt)
        
        # Disable adaptive dt checkbox when fixed dt is set
        self.adaptive_dt_checkbox.setChecked(False)
        
        print(f"dt changed to {new_dt:.6f} (in-simulation)")
        print(f"No reset required - dt applied immediately")
    
    def toggle_velocity(self, state):
        """Toggle velocity magnitude visualization"""
        self.show_velocity = (state == 2)  # Qt.Checked = 2
        if self.vel_plot is not None:
            self.vel_plot.setVisible(self.show_velocity)
        print(f"Velocity magnitude {'enabled' if self.show_velocity else 'disabled'}")
    
    def toggle_vorticity(self, state):
        """Toggle vorticity visualization"""
        self.show_vorticity = (state == 2)  # Qt.Checked = 2
        if self.vort_plot is not None:
            self.vort_plot.setVisible(self.show_vorticity)
        print(f"Vorticity {'enabled' if self.show_vorticity else 'disabled'}")
    
    def toggle_streamlines(self, state):
        """Toggle streamlines visualization"""
        self.show_streamlines = (state == 2)  # Qt.Checked = 2
        if self.stream_plot is not None:
            self.stream_plot.setVisible(self.show_streamlines)
        print(f"Streamlines {'enabled' if self.show_streamlines else 'disabled'}")
    
    def toggle_pressure(self, state):
        """Toggle pressure visualization"""
        self.show_pressure = (state == 2)  # Qt.Checked = 2
        if self.pressure_plot is not None:
            self.pressure_plot.setVisible(self.show_pressure)
        print(f"Pressure field {'enabled' if self.show_pressure else 'disabled'}")
    
    def toggle_energy(self, state):
        """Toggle energy plot visualization"""
        self.show_energy = (state == 2)  # Qt.Checked = 2
        if self.energy_plot is not None:
            self.energy_plot.setVisible(self.show_energy)
        print(f"Energy plot {'enabled' if self.show_energy else 'disabled'}")
    
    def toggle_forces(self, state):
        """Toggle drag/lift plot visualization"""
        self.show_forces = (state == 2)  # Qt.Checked = 2
        if self.drag_plot is not None:
            self.drag_plot.setVisible(self.show_forces)
        print(f"Drag/Lift plot {'enabled' if self.show_forces else 'disabled'}")
    
    def toggle_expensive_viz(self, state):
        """Toggle expensive visualizations for performance"""
        is_enabled = (state == 2)  # Qt.Checked = 2
        self.expensive_viz_enabled = is_enabled
        print(f"Advanced visualizations {'enabled' if is_enabled else 'disabled'}")
    
    def toggle_adaptive_dt(self, state):
        """Toggle between fixed and adaptive dt modes"""
        is_adaptive = (state == 2)  # Qt.Checked = 2
        
        self.timer.stop()
        
        if is_adaptive:
            # Switch to adaptive mode
            self.solver.set_adaptive_dt()
            self.dt_spinbox.setEnabled(False)
            self.apply_dt_btn.setEnabled(False)
            print("Switched to adaptive dt mode")
        else:
            # Switch to fixed mode
            current_dt = self.solver.dt
            self.solver.set_fixed_dt(current_dt)
            self.dt_spinbox.setEnabled(True)
            self.apply_dt_btn.setEnabled(True)
            self.dt_spinbox.setValue(self.solver.dt)
            print(f"Switched to fixed dt mode: {self.solver.dt:.6f}")
        
        self.reset_simulation()
    
    def change_colormap(self, colormap_name):
        """Change colormap for all plots"""
        try:
            colormap = pg.colormap.get(colormap_name)
            lut = colormap.getLookupTable()
            
            # Determine colormap type for appropriate assignment
            sequential_colormaps = [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
                'CET-C1', 'CET-C2', 'CET-C3', 'CET-C4', 'CET-C5', 'CET-C6', 'CET-C7',
                'CET-D1', 'CET-D2', 'CET-D3', 'CET-D4', 'CET-D6', 'CET-D7', 'CET-D8', 'CET-D9', 'CET-D10', 'CET-D11', 'CET-D12', 'CET-D13',
                'CET-L1', 'CET-L2', 'CET-L3', 'CET-L4', 'CET-L5', 'CET-L6', 'CET-L7', 'CET-L8', 'CET-L9', 'CET-L10', 'CET-L11', 'CET-L12', 'CET-L13', 'CET-L14', 'CET-L15', 'CET-L16', 'CET-L17', 'CET-L18', 'CET-L19',
                'PAL-relaxed', 'PAL-relaxed_bright'
            ]
            
            diverging_colormaps = [
                'CET-CBC1', 'CET-CBC2', 'CET-CBD1', 'CET-CBL1', 'CET-CBL2', 'CET-CBTC1', 'CET-CBTC2', 'CET-CBTD1', 'CET-CBTL1', 'CET-CBTL2',
                'CET-I1', 'CET-I2', 'CET-I3',
                'CET-R1', 'CET-R2', 'CET-R3', 'CET-R4'
            ]
            
            # Update velocity plot (always update with selected colormap)
            if hasattr(self, 'vel_img') and self.vel_img is not None:
                self.vel_img.setLookupTable(lut)
            
            # Update vorticity plot (always update with selected colormap)
            if hasattr(self, 'vort_img') and self.vort_img is not None:
                self.vort_img.setLookupTable(lut)
            
            # Update streamlines and pressure (sequential colormaps only)
            if colormap_name in sequential_colormaps:
                if hasattr(self, 'stream_img') and self.stream_img is not None:
                    self.stream_img.setLookupTable(lut)
                    
                if hasattr(self, 'pressure_img') and self.pressure_img is not None:
                    self.pressure_img.setLookupTable(lut)
            
            print(f"Colormap changed to {colormap_name}")
            
        except Exception as e:
            print(f"Error changing colormap: {e}")
    
    def export_data(self):
        """Export simulation data to files"""
        try:
            import datetime
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get current data
            u_np = np.array(self.solver.u)
            v_np = np.array(self.solver.v)
            
            # Get vorticity from cache if available, otherwise compute
            if hasattr(self.solver, 'current_pressure') and self.solver.current_pressure is not None:
                # Use cached vorticity if we have it (computed during step)
                vort_np = np.array(self.solver._vorticity(self.solver.u, self.solver.v, 
                                                              self.solver.grid.dx, self.solver.grid.dy))
            else:
                vort_np = np.zeros_like(u_np)  # Fallback
            
            # Get pressure from cache
            if hasattr(self.solver, 'current_pressure') and self.solver.current_pressure is not None:
                p_np = np.array(self.solver.current_pressure)
            else:
                p_np = np.zeros_like(u_np)  # Fallback
            
            # Export to CSV files
            np.savetxt(f'velocity_u_{timestamp}.csv', u_np, delimiter=',')
            np.savetxt(f'velocity_v_{timestamp}.csv', v_np, delimiter=',')
            np.savetxt(f'vorticity_{timestamp}.csv', vort_np, delimiter=',')
            np.savetxt(f'pressure_{timestamp}.csv', p_np, delimiter=',')
            
            # Export history data (convert from NumPy circular buffer)
            if self.history_idx > 0:
                # Get valid data from circular buffer
                n_points = min(self.history_idx, self.max_history)
                if self.history_idx <= self.max_history:
                    time_data = self.time_data[:n_points]
                    ke_data = self.ke_data[:n_points]
                    enst_data = self.enst_data[:n_points]
                    drag_data = self.drag_data[:n_points]
                    lift_data = self.lift_data[:n_points]
                else:
                    # Handle wrap-around
                    start_idx = self.history_idx % self.max_history
                    time_data = np.concatenate([self.time_data[start_idx:], self.time_data[:start_idx]])
                    ke_data = np.concatenate([self.ke_data[start_idx:], self.ke_data[:start_idx]])
                    enst_data = np.concatenate([self.enst_data[start_idx:], self.enst_data[:start_idx]])
                    drag_data = np.concatenate([self.drag_data[start_idx:], self.drag_data[:start_idx]])
                    lift_data = np.concatenate([self.lift_data[start_idx:], self.lift_data[:start_idx]])
                
                history_data = np.column_stack([time_data, ke_data, enst_data, drag_data, lift_data])
                np.savetxt(f'history_{timestamp}.csv', history_data, delimiter=',',
                          header='time,ke,enstrophy,drag,lift')
            
            # Export grid information
            grid_info = {
                'nx': self.solver.grid.nx,
                'ny': self.solver.grid.ny,
                'lx': self.solver.grid.lx,
                'ly': self.solver.grid.ly,
                'dx': self.solver.grid.dx,
                'dy': self.solver.grid.dy,
                'flow_type': self.solver.sim_params.flow_type,
                'Re': self.solver.flow.Re,
                'U_inf': self.solver.flow.U_inf,
                'nu': self.solver.flow.nu,
                'dt': self.solver.dt,
                'iteration': self.solver.iteration
            }
            
            # Export grid info as JSON
            import json
            with open(f'grid_info_{timestamp}.json', 'w') as f:
                json.dump(grid_info, f, indent=2)
            
            print(f"Data exported successfully with timestamp {timestamp}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def compute_streamlines_vectorized(self, u, v):
        """Vectorized streamline computation (much faster)"""
        try:
            nx, ny = u.shape
            dx = self.solver.grid.dx
            dy = self.solver.grid.dy
            
            # Validate grid parameters match input arrays
            if nx != self.solver.grid.nx or ny != self.solver.grid.ny:
                print(f"Grid mismatch in streamlines: input {u.shape} vs solver grid {self.solver.grid.nx}x{self.solver.grid.ny}")
                return np.zeros_like(u)
            
            # Use cumulative sum for vectorized integration
            # ψ[i,j] = ∫ u dy - ∫ v dx (approximated)
            
            # Create coordinate grids
            X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
            
            # Vectorized integration
            psi = np.zeros((nx, ny))
            
            # Integrate in x-direction (cumulative sum along axis=0)
            psi = np.cumsum(-v * dx, axis=0)
            
            # Integrate in y-direction (cumulative sum along axis=1)
            psi += np.cumsum(u * dy, axis=1)
            
            # Normalize
            psi = (psi - psi.min()) / (psi.max() - psi.min() + 1e-10)
            
            return psi
            
        except Exception as e:
            print(f"Error computing streamlines: {e}")
            return np.zeros_like(u)
    
    def apply_reynolds(self):
        new_re = self.re_input.value()
        
        self.timer.stop()
        
        self.solver.flow.Re = float(new_re)
        self.solver.flow.nu = self.solver.flow.U_inf * 2.0 * float(self.solver.geom.radius) / new_re
        
        self.solver._step_jit = jax.jit(self.solver._step)
        
        self.reset_simulation()
        
        print(f"Reynolds number changed to {new_re}")
        print(f"New kinematic viscosity: {self.solver.flow.nu:.6f}")
        print(f"JIT functions recompiled with new viscosity")
        
    def compute_pressure(self):
        """Compute pressure field"""
        try:
            from baseline_clean import divergence
            
            # Use the selected pressure solver
            if self.solver.sim_params.pressure_solver == 'jacobi':
                from pressure_solvers.jacobi_solver import poisson_jacobi
                if poisson_jacobi is not None:
                    p = poisson_jacobi(self.solver.u, self.solver.v, 
                                      jnp.ones_like(self.solver.u), 
                                      self.solver.grid.dx, self.solver.grid.dy, self.solver.dt)
                else:
                    return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            elif self.solver.sim_params.pressure_solver == 'fft':
                from pressure_solvers.fft_solver import poisson_fft
                if poisson_fft is not None:
                    p = poisson_fft(self.solver.u, self.solver.v, 
                                  jnp.ones_like(self.solver.u), 
                                  self.solver.grid.dx, self.solver.grid.dy, self.solver.dt)
                else:
                    return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            else:
                # Fallback to simple pressure field
                return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            
            return np.array(p)
            
        except Exception as e:
            print(f"Error computing pressure: {e}")
            return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
    
    def get_pressure_from_solver(self):
        """Get pressure field from solver without recomputation"""
        try:
            # Use cached pressure from solver (no recomputation!)
            pressure = self.solver.get_cached_pressure()
            if pressure is not None:
                return np.array(pressure)
            
            # Fallback: compute pressure if not cached
            from baseline_clean import divergence
            
            # Use the selected pressure solver
            if self.solver.sim_params.pressure_solver == 'jacobi':
                from pressure_solvers.jacobi_solver import poisson_jacobi
                if poisson_jacobi is not None:
                    p = poisson_jacobi(self.solver.u, self.solver.v, 
                                      self.solver.current_mask, 
                                      self.solver.grid.dx, self.solver.grid.dy, self.solver.dt,
                                      max_iter=self.solver.sim_params.pressure_max_iter)
                else:
                    return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            elif self.solver.sim_params.pressure_solver == 'fft':
                from pressure_solvers.fft_solver import poisson_fft
                if poisson_fft is not None:
                    p = poisson_fft(self.solver.u, self.solver.v, 
                                  self.solver.current_mask, 
                                  self.solver.grid.dx, self.solver.grid.dy, self.solver.dt)
                else:
                    return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            else:
                # For other solvers, return zeros for now
                return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
            
            return np.array(p)
        except Exception as e:
            print(f"Pressure computation error: {e}")
            return np.zeros((self.solver.grid.nx, self.solver.grid.ny))
    
    def update_grid_options(self):
        """Update grid size options based on selected flow type"""
        flow_type = self.flow_combo.currentText()
        
        # Define grid options for each flow type
        grid_options = {
            'von_karman': [
                '256x48 (Coarse)',
                '512x96 (Medium)', 
                '1024x192 (Fine)',
                '2048x384 (Ultra Fine)'
            ],
            'lid_driven_cavity': [
                '64x64 (Coarse)',
                '128x128 (Medium)',
                '256x256 (Fine)',
                '512x512 (Ultra Fine)'
            ],
            'channel_flow': [
                '128x32 (Coarse)',
                '256x64 (Medium)',
                '512x128 (Fine)',
                '1024x256 (Ultra Fine)'
            ],
            'backward_step': [
                '256x64 (Coarse)',
                '512x128 (Medium)',
                '1024x256 (Fine)',
                '2048x512 (Ultra Fine)'
            ],
            'taylor_green': [
                '64x64 (Coarse)',
                '128x128 (Medium)',
                '256x256 (Fine)',
                '512x512 (Ultra Fine)'
            ]
        }
        
        # Store current selection
        current_text = self.grid_combo.currentText() if hasattr(self, 'grid_combo') else None
        
        # Update dropdown
        self.grid_combo.clear()
        self.grid_combo.addItems(grid_options.get(flow_type, grid_options['von_karman']))
        
        # Set selection based on current solver grid
        self.set_current_grid_selection()
        
        # Update adaptive controller flow type if active
        if (hasattr(self.solver, 'adaptive_controller') and 
            self.solver.adaptive_controller is not None and 
            self.solver.adaptive_controller.flow_type != flow_type):
            
            print(f"Updating adaptive controller flow type to: {flow_type}")
            self.solver.adaptive_controller.flow_type = flow_type
            self.solver.adaptive_controller.reset_counters()

    def set_current_grid_selection(self):
        """Set grid dropdown selection based on current solver grid"""
        current_nx, current_ny = self.solver.grid.nx, self.solver.grid.ny
        
        # Find matching grid option
        for i in range(self.grid_combo.count()):
            text = self.grid_combo.itemText(i)
            # Extract nx, ny from selection like "512x96 (Medium)"
            import re
            match = re.search(r'(\d+)x(\d+)', text)
            if match:
                nx, ny = int(match.group(1)), int(match.group(2))
                if nx == current_nx and ny == current_ny:
                    self.grid_combo.setCurrentIndex(i)
                    return
        
        # If no match found, select medium quality by default
        self.grid_combo.setCurrentIndex(1)

    def update_adaptive_controller_for_grid_change(self):
        """Update adaptive dt controller when grid changes"""
        if (hasattr(self.solver, 'adaptive_controller') and 
            self.solver.adaptive_controller is not None and 
            self.solver.sim_params.adaptive_dt):
            
            # Reset counters for new grid
            self.solver.adaptive_controller.reset_counters()
            
            # Recalculate initial dt for new grid spacing
            old_dt = self.solver.dt
            self.solver.dt = self.solver.adaptive_controller.get_initial_dt(
                self.solver.flow.U_inf, self.solver.grid.dx, self.solver.grid.dy
            )
            
            print(f"Adaptive dt updated for new grid:")
            print(f"  Grid spacing: dx={self.solver.grid.dx:.4f}, dy={self.solver.grid.dy:.4f}")
            print(f"  dt: {old_dt:.6f} → {self.solver.dt:.6f}")
            
            # Recompile JIT with new dt
            self.solver._step_jit = jax.jit(self.solver._step)

    def get_grid_params_from_selection(self, grid_selection: str, flow_type: str):
        """Convert grid selection string to GridParams"""
        # Extract nx, ny from selection like "512x96 (Medium)"
        import re
        match = re.search(r'(\d+)x(\d+)', grid_selection)
        if not match:
            raise ValueError(f"Invalid grid selection: {grid_selection}")
        
        nx, ny = int(match.group(1)), int(match.group(2))
        
        # Define domain parameters for each flow type
        domains = {
            'von_karman': (20.0, 4.5),
            'lid_driven_cavity': (1.0, 1.0),
            'channel_flow': (4.0, 1.0),
            'backward_step': (10.0, 1.0),
            'taylor_green': (2*jnp.pi, 2*jnp.pi)
        }
        
        lx, ly = domains.get(flow_type, (20.0, 4.5))
        
        return GridParams(nx=nx, ny=ny, lx=lx, ly=ly)

    def apply_grid_size(self):
        """Apply only grid size change without changing flow type"""
        selected_grid = self.grid_combo.currentText()
        current_flow = self.solver.sim_params.flow_type
        
        self.timer.stop()
        
        # Store current grid info
        old_nx, old_ny = self.solver.grid.nx, self.solver.grid.ny
        
        # Get grid parameters from selection
        new_grid = self.get_grid_params_from_selection(selected_grid, current_flow)
        
        # Update solver with new grid (keep same flow type)
        self.solver.grid = new_grid
        
        # Recreate grid coordinates
        x = jnp.linspace(0, self.solver.grid.lx, self.solver.grid.nx)
        y = jnp.linspace(0, self.solver.grid.ly, self.solver.grid.ny)
        self.solver.grid.X, self.solver.grid.Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Reinitialize the flow (same flow type, new grid)
        if current_flow == 'lid_driven_cavity':
            self.solver._initialize_cavity_flow()
        elif current_flow == 'channel_flow':
            self.solver._initialize_channel_flow()
        elif current_flow == 'backward_step':
            self.solver._initialize_backward_step_flow()
        elif current_flow == 'taylor_green':
            self.solver._initialize_taylor_green_flow()
        else:  # von_karman
            self.solver._initialize_von_karman_flow()
        
        # Recompile JIT functions
        self.solver._step_jit = jax.jit(self.solver._step)
        
        # Reset history
        self.solver.history = {'time': [], 'ke': [], 'enstrophy': [], 'drag': [], 'lift': [], 'dt': []}
        self.solver.iteration = 0
        
        # Update adaptive dt controller for new grid
        self.update_adaptive_controller_for_grid_change()
        
        # Check if grid size changed
        new_nx, new_ny = self.solver.grid.nx, self.solver.grid.ny
        if old_nx != new_nx or old_ny != new_ny:
            # Update plots for new grid size
            self.update_plots_for_new_grid()
        
        self.reset_simulation()
        
        print(f"Grid size changed to {self.solver.grid.nx}x{self.solver.grid.ny} ({self.solver.grid.lx}x{self.solver.grid.ly})")
        print(f"Flow type: {current_flow}")
        print(f"Grid reinitialized and JIT functions recompiled")

    def apply_flow_type(self):
        selected_flow = self.flow_combo.currentText()
        selected_grid = self.grid_combo.currentText()
        
        self.timer.stop()
        
        # Store current grid info
        old_nx, old_ny = self.solver.grid.nx, self.solver.grid.ny
        
        # Get grid parameters from selection
        new_grid = self.get_grid_params_from_selection(selected_grid, selected_flow)
        
        # Use the solver's apply_flow_type method first to handle flow-specific setup
        self.solver.apply_flow_type(selected_flow)
        
        # Then override with custom grid selection
        self.solver.grid = new_grid
        
        # Recreate grid coordinates with custom grid
        x = jnp.linspace(0, self.solver.grid.lx, self.solver.grid.nx)
        y = jnp.linspace(0, self.solver.grid.ly, self.solver.grid.ny)
        self.solver.grid.X, self.solver.grid.Y = jnp.meshgrid(x, y, indexing='ij')
        
        # Reinitialize the flow with custom grid
        if selected_flow == 'lid_driven_cavity':
            self.solver._initialize_cavity_flow()
        elif selected_flow == 'channel_flow':
            self.solver._initialize_channel_flow()
        elif selected_flow == 'backward_step':
            self.solver._initialize_backward_step_flow()
        elif selected_flow == 'taylor_green':
            self.solver._initialize_taylor_green_flow()
        else:  # von_karman
            self.solver._initialize_von_karman_flow()
        
        # Recompute mask for new grid
        self.solver.mask = self.solver._compute_mask()
        
        # Clear JAX cache and recompile JIT functions
        jax.clear_caches()
        self.solver._step_jit = jax.jit(self.solver._step)
        
        # Reset history
        self.solver.history = {'time': [], 'ke': [], 'enstrophy': [], 'drag': [], 'lift': [], 'dt': []}
        self.solver.iteration = 0
        
        # Update adaptive dt controller for new grid
        self.update_adaptive_controller_for_grid_change()
        
        # Check if grid size changed
        new_nx, new_ny = self.solver.grid.nx, self.solver.grid.ny
        if old_nx != new_nx or old_ny != new_ny:
            # Clear cached streamlines when grid changes
            if hasattr(self, 'cached_streamlines'):
                self.cached_streamlines = None
            # Update plots for new grid size
            self.update_plots_for_new_grid()
        
        self.reset_simulation()
        
        print(f"Flow type changed to {selected_flow}")
        print(f"Grid updated to {self.solver.grid.nx}x{self.solver.grid.ny} ({self.solver.grid.lx}x{self.solver.grid.ly})")
        print(f"Flow reinitialized and JIT functions recompiled")
    
    def update_plots_for_new_grid(self):
        """Update plot items when grid size changes"""
        nx, ny = self.solver.grid.nx, self.solver.grid.ny
        lx, ly = self.solver.grid.lx, self.solver.grid.ly
        
        print(f"Updating plots for new grid: {nx}x{ny}, domain: {lx}x{ly}")
        
        # Clear cached streamlines to prevent shape mismatch
        if hasattr(self, 'cached_streamlines'):
            self.cached_streamlines = None
        
        # Clear and recreate plots completely
        self.plot_widget.clear()
        
        # Recreate plots
        self.vel_plot = self.plot_widget.addPlot(title="Velocity Magnitude", row=0, col=0)
        self.vort_plot = self.plot_widget.addPlot(title="Vorticity", row=0, col=1)
        self.stream_plot = self.plot_widget.addPlot(title="Streamlines", row=1, col=0)
        self.pressure_plot = self.plot_widget.addPlot(title="Pressure", row=1, col=1)
        
        # Configure plots
        for plot in [self.vel_plot, self.vort_plot, self.stream_plot, self.pressure_plot]:
            plot.setAspectLocked(True)
            plot.hideButtons()
            plot.enableAutoRange()  # Enable auto-fit
            plot.setAutoVisible(y=True)  # Auto-scale based on visible data
        
        # Create image items
        self.vel_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()
        self.stream_img = pg.ImageItem()
        self.pressure_img = pg.ImageItem()
        
        self.vel_plot.addItem(self.vel_img)
        self.vort_plot.addItem(self.vort_img)
        self.stream_plot.addItem(self.stream_img)
        self.pressure_plot.addItem(self.pressure_img)
        
        # Set initial colormaps
        plasma_lut = pg.colormap.get('plasma').getLookupTable()
        # Use available colormaps
        try:
            rdbu_lut = pg.colormap.get('RdBu').getLookupTable()
        except:
            rdbu_lut = pg.colormap.get('plasma').getLookupTable()  # Fallback to plasma
        
        viridis_lut = pg.colormap.get('viridis').getLookupTable()
        inferno_lut = pg.colormap.get('inferno').getLookupTable()
        
        self.vel_img.setLookupTable(plasma_lut)
        self.vort_img.setLookupTable(rdbu_lut)
        self.stream_img.setLookupTable(viridis_lut)
        self.pressure_img.setLookupTable(inferno_lut)
        
        # No colorbars - removed to fix plot sizing issues
        
        # Live plots for drag/lift
        self.drag_plot = self.plot_widget.addPlot(title="Drag & Lift", row=2, col=0, colspan=1)
        self.drag_curve = self.drag_plot.plot(pen='r', name='Drag')
        self.lift_curve = self.drag_plot.plot(pen='b', name='Lift')
        self.drag_plot.addLegend()
        self.drag_plot.setLabel('left', 'Force')
        self.drag_plot.setLabel('bottom', 'Time')
        self.drag_plot.enableAutoRange()  # Enable auto-fit
        self.drag_plot.setAutoVisible(y=True)  # Auto-scale based on visible data
        
        # KE/Enstrophy plot
        self.energy_plot = self.plot_widget.addPlot(title="Energy", row=2, col=1, colspan=1)
        self.ke_curve = self.energy_plot.plot(pen='g', name='KE')
        self.enst_curve = self.energy_plot.plot(pen='m', name='Enstrophy')
        self.energy_plot.addLegend()
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.setLabel('bottom', 'Time')
        self.energy_plot.enableAutoRange()  # Enable auto-fit
        self.energy_plot.setAutoVisible(y=True)  # Auto-scale based on visible data
        
        # Initialize plot data (recreate as NumPy arrays)
        self.max_history = 200
        self.time_data = np.zeros(self.max_history)
        self.drag_data = np.zeros(self.max_history)
        self.lift_data = np.zeros(self.max_history)
        self.ke_data = np.zeros(self.max_history)
        self.enst_data = np.zeros(self.max_history)
        self.history_idx = 0
        
        # Update plot ranges for ALL field plots
        self.vel_plot.setXRange(0, lx)
        self.vel_plot.setYRange(0, ly)
        self.vort_plot.setXRange(0, lx)
        self.vort_plot.setYRange(0, ly)
        self.stream_plot.setXRange(0, lx)
        self.stream_plot.setYRange(0, ly)
        self.pressure_plot.setXRange(0, lx)
        self.pressure_plot.setYRange(0, ly)
        
        print(f"Successfully updated plots for new grid")
        
    def apply_advection_scheme(self):
        selected_scheme = self.scheme_combo.currentText()
        
        self.timer.stop()
        
        self.solver.apply_advection_scheme(selected_scheme)
        
        self.reset_simulation()
        
        print(f"Advection scheme changed to {selected_scheme}")
        print(f"JIT functions recompiled with new scheme")
        
    def apply_pressure_solver(self):
        selected_solver = self.pressure_combo.currentText()
        
        self.timer.stop()
        
        self.solver.apply_pressure_solver(selected_solver)
        
        self.reset_simulation()
        
        print(f"Pressure solver changed to {selected_solver}")
        print(f"JIT functions recompiled with new pressure solver")
        
    def start_simulation(self):
        self.timer.start(33)  # 30 FPS - realistic for CFD simulation
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        
    def pause_simulation(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
    def reset_simulation(self):
        self.timer.stop()
        
        # Reinitialize based on current flow type
        if self.solver.sim_params.flow_type == 'lid_driven_cavity':
            self.solver._initialize_cavity_flow()
        elif self.solver.sim_params.flow_type == 'channel_flow':
            self.solver._initialize_channel_flow()
        elif self.solver.sim_params.flow_type == 'backward_step':
            self.solver._initialize_backward_step_flow()
        elif self.solver.sim_params.flow_type == 'taylor_green':
            self.solver._initialize_taylor_green_flow()
        else:  # von_karman
            self.solver._initialize_von_karman_flow()
        
        self.recorded_frames = []
        self.is_recording = False
        self.record_btn.setText("Record Video")
        self.save_btn.setEnabled(False)
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
        # Clear and reset image items completely
        self.vel_img.clear()
        self.vort_img.clear()
        
        # Recreate image items to ensure correct dimensions
        nx, ny = self.solver.grid.nx, self.solver.grid.ny
        lx, ly = self.solver.grid.lx, self.solver.grid.ly
        
        # Remove old items
        self.vel_plot.removeItem(self.vel_img)
        self.vort_plot.removeItem(self.vort_img)
        
        # Create new image items
        self.vel_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()
        
        self.vel_img.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        self.vort_img.setLookupTable(pg.colormap.get('plasma').getLookupTable())
        
        # Update scaling
        scale_x = lx/nx
        self.vel_img.setScale(scale_x)
        self.vel_img.setPos(0, 0)
        
        self.vort_img.setScale(scale_x)
        self.vort_img.setPos(0, 0)
        
        # Add items back to plots
        self.vel_plot.addItem(self.vel_img)
        self.vort_plot.addItem(self.vort_img)
        
        print(f"Reset simulation with perturbation strength 0.05")
        print(f"Image items recreated for {nx}x{ny} grid")
        
    def update(self):
        """Optimized update with reduced computational load"""
        try:
            # Check if widget is still valid
            if not hasattr(self, 'plot_widget') or self.plot_widget is None:
                return
            
            # Step solver once with optimized computation flags
            compute_vorticity = self.show_vorticity or self.show_pressure
            compute_energy = self.show_energy
            compute_drag_lift = self.show_forces
            
            u, v, vort, ke, enst, drag, lift = self.solver.step(
                compute_vorticity=compute_vorticity,
                compute_energy=compute_energy, 
                compute_drag_lift=compute_drag_lift
            )
            
            # Convert to numpy arrays only when needed for visible plots
            u_np = v_np = vort_np = None
            vel_mag = None
            
            # Check if velocity fields match current grid shape
            current_nx, current_ny = self.solver.grid.nx, self.solver.grid.ny
            if u.shape != (current_nx, current_ny):
                print(f"Velocity shape mismatch: u.shape {u.shape} vs grid {(current_nx, current_ny)}")
                return  # Skip this frame, wait for solver to catch up
            
            if self.show_velocity:
                u_np = np.array(u)
                v_np = np.array(v)
                vel_mag = np.sqrt(u_np**2 + v_np**2)
            
            if self.show_vorticity:
                if vort_np is None:  # Only convert if not already done for velocity
                    vort_np = np.array(vort)
            
            # Check shapes only if velocity plot is visible
            if self.show_velocity and vel_mag is not None:
                current_nx, current_ny = self.solver.grid.nx, self.solver.grid.ny
                if vel_mag.shape != (current_nx, current_ny):
                    return
            
            # OPTIMIZATION 1: Alternate plot updates per frame
            # Initialize frame counter if needed
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            
            # Update only 2 plots per frame, others less frequently
            if self._frame_counter % 2 == 0:
                # Even frames: velocity, vorticity, and streamlines
                if self.show_velocity and self.vel_img is not None:
                    self.vel_img.setImage(vel_mag, levels=(0, 2))
                if self.show_vorticity and self.vort_img is not None:
                    self.vort_img.setImage(vort_np, levels=(-5, 5))
                
                # Streamlines - cached and less frequent (move to even frames)
                if self.show_streamlines and self._frame_counter % 100 == 0 and u_np is not None:
                    try:
                        if self.stream_img is not None:
                            if not hasattr(self, 'cached_streamlines'):
                                self.cached_streamlines = None
                            # Only recompute if not cached or flow changed significantly
                            print(f"Recomputing streamlines at frame {self._frame_counter}")
                            streamlines = self.compute_streamlines_vectorized(u_np, v_np)
                            print(f"Streamlines computed, shape: {streamlines.shape}, range: [{streamlines.min():.3f}, {streamlines.max():.3f}]")
                            self.cached_streamlines = streamlines
                            self.stream_img.setImage(streamlines)
                            print(f"Streamlines set on image item")
                    except Exception as e:
                        print(f"Streamlines error: {e}")
                elif self.show_streamlines and hasattr(self, 'cached_streamlines') and self.cached_streamlines is not None and self.stream_img is not None:
                    # Check if cached streamlines match current grid shape
                    if self.cached_streamlines.shape != u_np.shape:
                        print(f"Streamlines shape mismatch: cached {self.cached_streamlines.shape} vs current {u_np.shape}")
                        self.cached_streamlines = None  # Clear cache
                        return  # Skip this frame, recompute next time
                    
                    # Additional safety check: verify cached streamlines match solver grid
                    if (self.cached_streamlines.shape[0] != self.solver.grid.nx or 
                        self.cached_streamlines.shape[1] != self.solver.grid.ny):
                        print(f"Streamlines grid mismatch: cached {self.cached_streamlines.shape} vs solver grid {self.solver.grid.nx}x{self.solver.grid.ny}")
                        self.cached_streamlines = None  # Clear cache
                        return  # Skip this frame, recompute next time
                    
                    # Use cached streamlines most of the time
                    if self._frame_counter % 20 == 0:  # Debug every 20 frames
                        print(f"Using cached streamlines at frame {self._frame_counter}, shape: {self.cached_streamlines.shape}")
                    self.stream_img.setImage(self.cached_streamlines)
            else:
                # Odd frames: pressure only
                if self.show_pressure and self.pressure_img is not None:
                    try:
                        pressure = self.get_pressure_from_solver()
                        self.pressure_img.setImage(pressure, levels=(-1, 1))
                    except Exception as e:
                        print(f"Pressure error: {e}")
            
            self._frame_counter += 1
            
            # OPTIMIZATION 3: Update live plots with reduced frequency
            if self._frame_counter % 10 == 0:  # Update plots every 10 frames (further reduced)
                current_time = self.solver.iteration * self.solver.dt
                
                # Only update history data if plots are visible
                if self.show_forces or self.show_energy:
                    # Use NumPy array indexing (much faster than append)
                    idx = self.history_idx % self.max_history
                    self.time_data[idx] = current_time
                    self.drag_data[idx] = float(drag)
                    self.lift_data[idx] = float(lift)
                    self.ke_data[idx] = float(ke)
                    self.enst_data[idx] = float(enst)
                    self.history_idx += 1
                
                # Update curves only if plots are visible
                if self.show_forces:
                    n_points = min(self.history_idx, self.max_history)
                    if n_points > 0:
                        # Correct circular buffer slicing
                        if self.history_idx <= self.max_history:
                            # Simple case: no wrap-around
                            time_slice = self.time_data[:n_points]
                            drag_slice = self.drag_data[:n_points]
                            lift_slice = self.lift_data[:n_points]
                        else:
                            # Wrap-around case: get last N points from circular buffer
                            start_idx = self.history_idx % self.max_history
                            time_slice = np.concatenate([
                                self.time_data[start_idx:],
                                self.time_data[:start_idx]
                            ])
                            drag_slice = np.concatenate([
                                self.drag_data[start_idx:],
                                self.drag_data[:start_idx]
                            ])
                            lift_slice = np.concatenate([
                                self.lift_data[start_idx:],
                                self.lift_data[:start_idx]
                            ])
                        
                        self.drag_curve.setData(time_slice, drag_slice)
                        self.lift_curve.setData(time_slice, lift_slice)
                
                if self.show_energy:
                    n_points = min(self.history_idx, self.max_history)
                    if n_points > 0:
                        # Correct circular buffer slicing
                        if self.history_idx <= self.max_history:
                            # Simple case: no wrap-around
                            time_slice = self.time_data[:n_points]
                            ke_slice = self.ke_data[:n_points]
                            enst_slice = self.enst_data[:n_points]
                        else:
                            # Wrap-around case: get last N points from circular buffer
                            start_idx = self.history_idx % self.max_history
                            time_slice = np.concatenate([
                                self.time_data[start_idx:],
                                self.time_data[:start_idx]
                            ])
                            ke_slice = np.concatenate([
                                self.ke_data[start_idx:],
                                self.ke_data[:start_idx]
                            ])
                            enst_slice = np.concatenate([
                                self.enst_data[start_idx:],
                                self.enst_data[:start_idx]
                            ])
                        
                        self.ke_curve.setData(time_slice, ke_slice)
                        self.enst_curve.setData(time_slice, enst_slice)
            
            # Update CFL display (reduced frequency)
            if hasattr(self, 'cfl_label') and self.cfl_label is not None and self._frame_counter % 10 == 0:
                try:
                    current_cfl = float(self.solver.check_cfl_condition())
                    self.cfl_label.setText(f"CFL: {current_cfl:.3f}")
                    
                    if current_cfl > 0.8:
                        self.cfl_label.setStyleSheet("color: red; font-weight: bold;")
                    elif current_cfl > 0.5:
                        self.cfl_label.setStyleSheet("color: orange; font-weight: bold;")
                    else:
                        self.cfl_label.setStyleSheet("color: green;")
                except Exception:
                    self.cfl_label.setText("CFL: N/A")
                    self.cfl_label.setStyleSheet("color: gray;")
            
            # Update info label (cheap)
            step = self.solver.iteration
            time = step * self.solver.dt
            if hasattr(self, 'info_label') and self.info_label is not None:
                self.info_label.setText(
                    f"Step: {step:6d} | Time: {time:.2f}s | "
                    f"KE: {ke:.4f} | Drag: {drag:.4f} | Lift: {lift:+.4f} | dt: {self.solver.dt:.4f}"
                )
            
            # Capture frame for recording (only if recording)
            if self.is_recording:
                self.capture_frame()
            
        except Exception as e:
            import traceback
            print(f"Update error: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            if hasattr(self, 'info_label') and self.info_label is not None:
                self.info_label.setText(f"Error: {str(e)[:50]}...")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            self.record_btn.setText("Stop Recording")
            self.save_btn.setEnabled(False)
            print("Started recording...")
        else:
            self.is_recording = False
            self.record_btn.setText("Record Video")
            self.save_btn.setEnabled(True)
            print(f"Stopped recording. Captured {len(self.recorded_frames)} frames.")
            
    def save_video(self):
        if not self.recorded_frames:
            print("No frames to save!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "GIF Files (*.gif);;All Files (*)"
        )
        
        if filename:
            if not filename.endswith('.gif'):
                filename += '.gif'
                
            print(f"Saving {len(self.recorded_frames)} frames to {filename}...")
            
            self.recorded_frames[0].save(
                filename,
                save_all=True,
                append_images=self.recorded_frames[1:],
                duration=50,
                loop=0
            )
            print(f"Video saved to {filename}")
            
    def capture_frame(self):
        """Capture frame for video recording - optimized"""
        if self.is_recording:
            # Reduce capture frequency to prevent stuttering
            if not hasattr(self, '_capture_frame_counter'):
                self._capture_frame_counter = 0
            self._capture_frame_counter += 1
            
            # Only capture every 3rd frame for smooth recording
            if self._capture_frame_counter % 3 != 0:
                return
            
            # Use the more efficient ImageExporter method
            try:
                exporter = pg.exporters.ImageExporter(self.plot_widget)
                frame = exporter.export(toBytes=True)
                self.recorded_frames.append(frame)
            except Exception as e:
                print(f"Frame capture error: {e}")
                # Fallback to grab method if exporter fails
                pixmap = self.plot_widget.grab()
                qimage = pixmap.toImage()
                w, h = qimage.width(), qimage.height()
                ptr = qimage.bits()
                ptr.setsize(h * w * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
                rgb_arr = arr[:, :, :3]
                pil_image = Image.fromarray(rgb_arr, 'RGB')
                self.recorded_frames.append(pil_image)
        
    def setup_plots(self):
        """Setup all visualization plots"""
        nx, ny = self.solver.grid.nx, self.solver.grid.ny
        lx, ly = self.solver.grid.lx, self.solver.grid.ly
        
        # Main visualization plots
        self.vel_plot = self.plot_widget.addPlot(title="Velocity Magnitude", row=0, col=0)
        self.vel_plot.setLabel('left', 'y')
        self.vel_plot.setLabel('bottom', 'x')
        self.vel_plot.setAspectLocked(True)
        self.vel_plot.setFixedWidth(600)  # Fixed width for equal columns
        
        self.vort_plot = self.plot_widget.addPlot(title="Vorticity", row=0, col=1)
        self.vort_plot.setLabel('left', 'y')
        self.vort_plot.setLabel('bottom', 'x')
        self.vort_plot.setAspectLocked(True)
        self.vort_plot.setFixedWidth(600)  # Fixed width for equal columns
        
        self.stream_plot = self.plot_widget.addPlot(title="Streamlines", row=1, col=0)
        self.stream_plot.setLabel('left', 'y')
        self.stream_plot.setLabel('bottom', 'x')
        self.stream_plot.setAspectLocked(True)
        self.stream_plot.setFixedWidth(600)  # Fixed width for equal columns
        
        self.pressure_plot = self.plot_widget.addPlot(title="Pressure", row=1, col=1)
        self.pressure_plot.setLabel('left', 'y')
        self.pressure_plot.setLabel('bottom', 'x')
        self.pressure_plot.setAspectLocked(True)
        self.pressure_plot.setFixedWidth(600)  # Fixed width for equal columns
        
        # Set initial plot ranges for all field plots
        self.vel_plot.setXRange(0, lx)
        self.vel_plot.setYRange(0, ly)
        self.vort_plot.setXRange(0, lx)
        self.vort_plot.setYRange(0, ly)
        self.stream_plot.setXRange(0, lx)
        self.stream_plot.setYRange(0, ly)
        self.pressure_plot.setXRange(0, lx)
        self.pressure_plot.setYRange(0, ly)
        
        # Create image items
        self.vel_img = pg.ImageItem()
        self.vort_img = pg.ImageItem()
        self.stream_img = pg.ImageItem()
        self.pressure_img = pg.ImageItem()
        
        # Set initial colormaps
        try:
            plasma_lut = pg.colormap.get('plasma').getLookupTable()
            rdbu_lut = pg.colormap.get('RdBu').getLookupTable()
            viridis_lut = pg.colormap.get('viridis').getLookupTable()
            inferno_lut = pg.colormap.get('inferno').getLookupTable()
        except:
            plasma_lut = pg.colormap.get('plasma').getLookupTable()
            rdbu_lut = pg.colormap.get('plasma').getLookupTable()
            viridis_lut = pg.colormap.get('viridis').getLookupTable()
            inferno_lut = pg.colormap.get('inferno').getLookupTable()
        
        self.vel_img.setLookupTable(plasma_lut)
        self.vort_img.setLookupTable(rdbu_lut)
        self.stream_img.setLookupTable(viridis_lut)
        self.pressure_img.setLookupTable(inferno_lut)
        
        # Add image items to plots
        self.vel_plot.addItem(self.vel_img)
        self.vort_plot.addItem(self.vort_img)
        self.stream_plot.addItem(self.stream_img)
        self.pressure_plot.addItem(self.pressure_img)
        
        # No colorbars - removed to fix plot sizing issues
        
        # Set scaling and positioning
        scale_x = lx/nx
        scale_y = ly/ny
        
        self.vel_img.setScale(scale_x)
        self.vel_img.setPos(0, 0)
        
        self.vort_img.setScale(scale_x)
        self.vort_img.setPos(0, 0)
        
        self.stream_img.setScale(scale_x)
        self.stream_img.setPos(0, 0)
        
        self.pressure_img.setScale(scale_x)
        self.pressure_img.setPos(0, 0)
        
        # Set plot ranges
        self.vel_plot.setXRange(0, lx)
        self.vel_plot.setYRange(0, ly)
        self.vort_plot.setXRange(0, lx)
        self.vort_plot.setYRange(0, ly)
        self.stream_plot.setXRange(0, lx)
        self.stream_plot.setYRange(0, ly)
        self.pressure_plot.setXRange(0, lx)
        self.pressure_plot.setYRange(0, ly)
        
        # Configure all plots
        for plot in [self.vel_plot, self.vort_plot, self.stream_plot, self.pressure_plot]:
            plot.setAspectLocked(True)
            plot.hideButtons()
            plot.enableAutoRange(False)
            plot.setAutoVisible(y=False)
        
        # Live plots for drag/lift
        self.drag_plot = self.plot_widget.addPlot(title="Drag & Lift", row=2, col=0, colspan=1)
        self.drag_curve = self.drag_plot.plot(pen='r', name='Drag')
        self.lift_curve = self.drag_plot.plot(pen='b', name='Lift')
        self.drag_plot.addLegend()
        self.drag_plot.setLabel('left', 'Force')
        self.drag_plot.setLabel('bottom', 'Time')
        self.drag_plot.enableAutoRange()
        self.drag_plot.setAutoVisible(y=True)
        
        # KE/Enstrophy plot
        self.energy_plot = self.plot_widget.addPlot(title="Energy", row=2, col=1, colspan=1)
        self.ke_curve = self.energy_plot.plot(pen='g', name='KE')
        self.enst_curve = self.energy_plot.plot(pen='m', name='Enstrophy')
        self.energy_plot.addLegend()
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.setLabel('bottom', 'Time')
        self.energy_plot.enableAutoRange()
        self.energy_plot.setAutoVisible(y=True)
        
        # Initialize plot data (recreate as NumPy arrays)
        self.max_history = 200
        self.time_data = np.zeros(self.max_history)
        self.drag_data = np.zeros(self.max_history)
        self.lift_data = np.zeros(self.max_history)
        self.ke_data = np.zeros(self.max_history)
        self.enst_data = np.zeros(self.max_history)
        self.history_idx = 0
        
    def recreate_image_items(self):
        """Recreate image items with current grid dimensions"""
        nx, ny = self.solver.grid.nx, self.solver.grid.ny
        lx, ly = self.solver.grid.lx, self.solver.grid.ly
        
        try:
            # Remove old items
            self.vel_plot.removeItem(self.vel_img)
            self.vort_plot.removeItem(self.vort_img)
            self.stream_plot.removeItem(self.stream_img)
            self.pressure_plot.removeItem(self.pressure_img)
            
            # Create new image items
            self.vel_img = pg.ImageItem()
            self.vort_img = pg.ImageItem()
            self.stream_img = pg.ImageItem()
            self.pressure_img = pg.ImageItem()
            
            # Set colormaps
            plasma_lut = pg.colormap.get('plasma').getLookupTable()
            # Use available colormaps
            try:
                rdbu_lut = pg.colormap.get('RdBu').getLookupTable()
            except:
                rdbu_lut = pg.colormap.get('plasma').getLookupTable()  # Fallback to plasma
            
            viridis_lut = pg.colormap.get('viridis').getLookupTable()
            inferno_lut = pg.colormap.get('inferno').getLookupTable()
            
            self.vel_img.setLookupTable(plasma_lut)
            self.vort_img.setLookupTable(rdbu_lut)
            self.stream_img.setLookupTable(viridis_lut)
            self.pressure_img.setLookupTable(inferno_lut)
            
            # Update scaling
            scale_x = lx/nx
            self.vel_img.setScale(scale_x)
            self.vel_img.setPos(0, 0)
            
            self.vort_img.setScale(scale_x)
            self.vort_img.setPos(0, 0)
            
            self.stream_img.setScale(scale_x)
            self.stream_img.setPos(0, 0)
            
            self.pressure_img.setScale(scale_x)
            self.pressure_img.setPos(0, 0)
            
            # Add items back to plots
            self.vel_plot.addItem(self.vel_img)
            self.vort_plot.addItem(self.vort_img)
            self.stream_plot.addItem(self.stream_img)
            self.pressure_plot.addItem(self.pressure_img)
            
            # No colorbars to reconnect - removed to fix plot sizing issues
            
            print(f"Image items recreated for {nx}x{ny} grid")
        except Exception as e:
            print(f"Error recreating image items: {e}")
            
    def closeEvent(self, event):
        print("Visualization stopped by user")
        self.timer.stop()
        
        # Clean up visualization objects to prevent errors
        try:
            if hasattr(self, 'vel_img') and self.vel_img is not None:
                self.vel_img.clear()
            if hasattr(self, 'vort_img') and self.vort_img is not None:
                self.vort_img.clear()
            if hasattr(self, 'stream_img') and self.stream_img is not None:
                self.stream_img.clear()
            if hasattr(self, 'pressure_img') and self.pressure_img is not None:
                self.pressure_img.clear()
        except:
            pass  # Ignore cleanup errors
        
        # Set objects to None
        self.vel_img = None
        self.vort_img = None
        self.stream_img = None
        self.pressure_img = None
        self.plot_widget = None
        
        super().closeEvent(event)

def run_pyqtgraph_visualization(solver: BaselineSolver):
    print("Starting baseline visualization...")
    print(f"Physical domain: X=[{solver.grid.lx:.1f}, {solver.grid.lx:.1f}]m")
    print(f"Physical domain: Y=[0.0, {solver.grid.ly:.1f}]m")
    print(f"Grid resolution: {solver.grid.nx}x{solver.grid.ny}")
    print(f"Reynolds number: {solver.flow.Re:.1f}")
    print(f"Mode: Baseline")
    print("(Close window to stop simulation)")
    
    app = QApplication(sys.argv)
    viewer = BaselineViewer(solver)
    viewer.show()
    sys.exit(app.exec())

def main():
    print("=" * 60)
    print("Baseline Viewer - Clean Navier-Stokes Solver")
    print("=" * 60)
    print("Optimized for vortex shedding visualization")
    
    grid = GridParams(nx=512, ny=96, lx=20.0, ly=4.5)
    flow = FlowParams(Re=300.0, U_inf=1.0)
    geom = GeometryParams(center_x=jnp.array(2.5), center_y=jnp.array(2.25), radius=jnp.array(0.18))
    sim_params = SimulationParams(Cs=0.17, eps=0.05)
    
    solver = BaselineSolver(grid, flow, geom, sim_params, dt=0.002)
    
    X, Y = solver.grid.X, solver.grid.Y
    print(f"\nScaling Information:")
    print(f"  Physical domain: X=[{float(X[0, 0]):.1f}, {float(X[-1, 0]):.1f}]m")
    print(f"  Physical domain: Y=[{float(Y[0, 0]):.1f}, {float(Y[0, -1]):.1f}]m")
    print(f"  Grid spacing: dx={solver.grid.dx:.4f}m, dy={solver.grid.dy:.4f}m")
    print(f"  Total domain size: {(float(X[-1, 0]) - float(X[0, 0])):.1f}m x {(float(Y[0, -1]) - float(Y[0, 0])):.1f}m")
    
    print(f"\nInitial perturbation already added by BaselineSolver")
    print(f"Ready for vortex shedding visualization")
    
    run_pyqtgraph_visualization(solver)

if __name__ == "__main__":
    main()
