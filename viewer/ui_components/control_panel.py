"""
Main control panel orchestrating all UI components.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QSpinBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QSlider, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt
from .top_console import TopConsole
from .obstacle_controls import ObstacleControls
from .time_controls import TimeControls
from .dye_controls import DyeControls
from .visualization_controls import VisualizationControls
from .neural_operator_training import NeuralOperatorTraining
from .collapsible_groupbox import CollapsibleGroupBox


class ControlPanel(QWidget):
    """Main control panel with top console and sidebar for advanced controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the complete control panel UI: top console + scrollable sidebar"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # ========== TOP CONSOLE (horizontal bar with buttons only) ==========
        self.top_console = TopConsole(self)
        main_layout.addWidget(self.top_console)

        # ========== SCROLLABLE SIDEBAR (all groupboxes at same level) ==========
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Sidebar content widget
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # ========== GRID SIZE GROUP ==========
        grid_group = CollapsibleGroupBox("Grid Size", start_collapsed=True)
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid:"))
        self.grid_x_spinbox = QSpinBox()
        self.grid_x_spinbox.setRange(64, 4096)
        self.grid_x_spinbox.setValue(512)
        self.grid_x_spinbox.setSingleStep(64)
        self.grid_x_spinbox.setMaximumWidth(90)
        grid_layout.addWidget(self.grid_x_spinbox)
        grid_layout.addWidget(QLabel("×"))
        self.grid_y_spinbox = QSpinBox()
        self.grid_y_spinbox.setRange(32, 2048)
        self.grid_y_spinbox.setValue(192)  # Increased from 128 for proper circulation contour margins
        self.grid_y_spinbox.setSingleStep(32)
        self.grid_y_spinbox.setMaximumWidth(90)
        grid_layout.addWidget(self.grid_y_spinbox)
        self.apply_grid_btn = QPushButton("Apply")
        self.apply_grid_btn.setMaximumWidth(60)
        grid_layout.addWidget(self.apply_grid_btn)
        grid_layout.addStretch()
        grid_group.setLayout(grid_layout)
        sidebar_layout.addWidget(grid_group)
        
        # ========== GRID TYPE GROUP ==========
        grid_type_group = CollapsibleGroupBox("Grid Type", start_collapsed=True)
        grid_type_layout = QHBoxLayout()
        grid_type_layout.addWidget(QLabel("Type:"))
        self.grid_type_combo = QComboBox()
        self.grid_type_combo.addItem("Collocated", "collocated")
        self.grid_type_combo.addItem("MAC (Staggered)", "mac")
        self.grid_type_combo.setCurrentIndex(0)  # Default to collocated
        self.grid_type_combo.setMaximumWidth(150)
        grid_type_layout.addWidget(self.grid_type_combo)
        self.apply_grid_type_btn = QPushButton("Apply")
        self.apply_grid_type_btn.setMaximumWidth(60)
        grid_type_layout.addWidget(self.apply_grid_type_btn)
        grid_type_layout.addStretch()
        grid_type_group.setLayout(grid_type_layout)
        sidebar_layout.addWidget(grid_type_group)

        # ========== SOLVER TYPE GROUP ==========
        solver_type_group = CollapsibleGroupBox("Solver Type", start_collapsed=True)
        solver_type_layout = QVBoxLayout()
        
        # Solver type selection row
        solver_row = QHBoxLayout()
        solver_row.addWidget(QLabel("Method:"))
        self.solver_type_combo = QComboBox()
        self.solver_type_combo.addItem("Navier-Stokes", "navier_stokes")
        self.solver_type_combo.addItem("Lattice Boltzmann", "lattice_boltzmann")
        self.solver_type_combo.setCurrentIndex(0)  # Default to Navier-Stokes
        self.solver_type_combo.setMaximumWidth(180)
        self.solver_type_combo.currentIndexChanged.connect(self.on_solver_type_index_changed)
        solver_row.addWidget(self.solver_type_combo)
        self.apply_solver_type_btn = QPushButton("Apply")
        self.apply_solver_type_btn.setMaximumWidth(60)
        solver_row.addWidget(self.apply_solver_type_btn)
        solver_row.addStretch()
        solver_type_layout.addLayout(solver_row)
        
        # LBM Tau parameter row (initially hidden)
        self.lbm_tau_widget = QWidget()
        lbm_tau_row = QHBoxLayout(self.lbm_tau_widget)
        lbm_tau_row.setContentsMargins(0, 0, 0, 0)
        lbm_tau_row.addWidget(QLabel("Tau:"))
        self.tau_slider = QSlider(Qt.Orientation.Horizontal)
        self.tau_slider.setRange(51, 200)  # Range for tau: 0.51 to 2.00 (multiply by 0.01)
        self.tau_slider.setValue(60)  # Default tau = 0.60
        self.tau_slider.setMaximumWidth(350)
        lbm_tau_row.addWidget(self.tau_slider)
        self.tau_spinbox = QDoubleSpinBox()
        self.tau_spinbox.setRange(0.51, 2.00)
        self.tau_spinbox.setSingleStep(0.001)
        self.tau_spinbox.setDecimals(3)
        self.tau_spinbox.setValue(0.60)
        self.tau_spinbox.setMaximumWidth(80)
        lbm_tau_row.addWidget(self.tau_spinbox)
        self.apply_tau_btn = QPushButton("Apply")
        self.apply_tau_btn.setMaximumWidth(50)
        lbm_tau_row.addWidget(self.apply_tau_btn)
        lbm_tau_row.addStretch()
        solver_type_layout.addWidget(self.lbm_tau_widget)
        self.lbm_tau_widget.setVisible(False)  # Initially hidden
        self.tau_slider.valueChanged.connect(self._update_tau_from_slider)
        self.tau_spinbox.valueChanged.connect(self._update_tau_from_spinbox)
        
        # LBM Outlet type row (initially hidden)
        self.lbm_outlet_widget = QWidget()
        lbm_outlet_row = QHBoxLayout(self.lbm_outlet_widget)
        lbm_outlet_row.setContentsMargins(0, 0, 0, 0)
        lbm_outlet_row.addWidget(QLabel("Outlet:"))
        self.outlet_type_combo = QComboBox()
        self.outlet_type_combo.addItem("Convective (CBC)", "convective")
        self.outlet_type_combo.addItem("Extrapolation", "extrapolation")
        self.outlet_type_combo.setCurrentIndex(0)  # Default to convective
        self.outlet_type_combo.setMaximumWidth(150)
        lbm_outlet_row.addWidget(self.outlet_type_combo)
        self.apply_outlet_btn = QPushButton("Apply")
        self.apply_outlet_btn.setMaximumWidth(50)
        self.apply_outlet_btn.clicked.connect(self.on_outlet_type_changed)
        lbm_outlet_row.addWidget(self.apply_outlet_btn)
        lbm_outlet_row.addStretch()
        solver_type_layout.addWidget(self.lbm_outlet_widget)
        self.lbm_outlet_widget.setVisible(False)  # Initially hidden
        
        # LBM BC mode row (initially hidden)
        self.lbm_bc_mode_widget = QWidget()
        lbm_bc_mode_row = QHBoxLayout(self.lbm_bc_mode_widget)
        lbm_bc_mode_row.setContentsMargins(0, 0, 0, 0)
        lbm_bc_mode_row.addWidget(QLabel("BC Mode:"))
        self.bc_mode_combo = QComboBox()
        self.bc_mode_combo.addItem("Supply (L→R)", "supply")
        self.bc_mode_combo.addItem("Extract (R→L)", "extract")
        self.bc_mode_combo.setCurrentIndex(0)  # Default to supply
        self.bc_mode_combo.setMaximumWidth(150)
        lbm_bc_mode_row.addWidget(self.bc_mode_combo)
        self.apply_bc_mode_btn = QPushButton("Apply")
        self.apply_bc_mode_btn.setMaximumWidth(50)
        self.apply_bc_mode_btn.clicked.connect(self.on_bc_mode_changed)
        lbm_bc_mode_row.addWidget(self.apply_bc_mode_btn)
        lbm_bc_mode_row.addStretch()
        solver_type_layout.addWidget(self.lbm_bc_mode_widget)
        self.lbm_bc_mode_widget.setVisible(False)  # Initially hidden
        
        solver_type_group.setLayout(solver_type_layout)
        sidebar_layout.addWidget(solver_type_group)

        # ========== REYNOLDS NUMBER GROUP ==========
        re_group = CollapsibleGroupBox("Reynolds Number", start_collapsed=True)
        from PyQt6.QtWidgets import QGridLayout
        re_layout = QGridLayout()
        re_layout.setSpacing(5)
        re_layout.setColumnStretch(3, 1)  # Stretch last column

        # Row 0: U input
        re_layout.addWidget(QLabel("U (m/s):"), 0, 0)
        self.u_input = QDoubleSpinBox()
        self.u_input.setRange(0.01, 100.0)
        self.u_input.setSingleStep(0.1)
        self.u_input.setValue(0.5)
        self.u_input.setMaximumWidth(110)
        re_layout.addWidget(self.u_input, 0, 1)
        self.lock_u_cb = QCheckBox("Lock")
        self.lock_u_cb.setChecked(False)
        re_layout.addWidget(self.lock_u_cb, 0, 2)

        # Row 1: ν input
        re_layout.addWidget(QLabel("ν (m²/s):"), 1, 0)
        self.nu_input = QDoubleSpinBox()
        self.nu_input.setRange(1e-6, 1.0)
        self.nu_input.setSingleStep(1e-4)
        self.nu_input.setDecimals(6)
        self.nu_input.setValue(0.001667)
        self.nu_input.setMaximumWidth(110)
        re_layout.addWidget(self.nu_input, 1, 1)
        self.lock_nu_cb = QCheckBox("Lock")
        self.lock_nu_cb.setChecked(True)
        re_layout.addWidget(self.lock_nu_cb, 1, 2)

        # Row 2: Re input
        re_layout.addWidget(QLabel("Re:"), 2, 0)
        self.re_input = QDoubleSpinBox()
        self.re_input.setRange(1.0, 100000.0)
        self.re_input.setSingleStep(1.0)
        self.re_input.setValue(2000.0)
        self.re_input.setMaximumWidth(110)
        re_layout.addWidget(self.re_input, 2, 1)
        self.lock_re_cb = QCheckBox("Lock")
        self.lock_re_cb.setChecked(True)
        re_layout.addWidget(self.lock_re_cb, 2, 2)

        # Row 3: Apply button
        self.apply_re_btn = QPushButton("Apply")
        self.apply_re_btn.setMaximumWidth(60)
        re_layout.addWidget(self.apply_re_btn, 3, 0, 1, 2)  # Span 2 columns

        re_group.setLayout(re_layout)
        sidebar_layout.addWidget(re_group)

        # ========== FLOW TYPE GROUP ==========
        flow_group = CollapsibleGroupBox("Flow Type", start_collapsed=True)
        flow_layout = QVBoxLayout()
        flow_row = QHBoxLayout()
        flow_row.addWidget(QLabel("Flow:"))
        self.flow_combo = QComboBox()
        self.flow_combo.addItems(["von_karman", "lid_driven_cavity", "taylor_green", "kelvin_helmholtz"])
        self.flow_combo.setMaximumWidth(150)
        self.flow_combo.currentTextChanged.connect(self.on_flow_type_changed)
        flow_row.addWidget(self.flow_combo)
        flow_row.addStretch()
        flow_layout.addLayout(flow_row)
        
        # LDC Benchmark Re selection (shown only for lid_driven_cavity)
        self.ldc_re_widget = QWidget()
        self.ldc_re_row = QHBoxLayout(self.ldc_re_widget)
        self.ldc_re_row.setContentsMargins(0, 0, 0, 0)
        self.ldc_re_row.addWidget(QLabel("LDC Benchmark Re:"))
        self.ldc_re_button_group = QButtonGroup(self)
        self.ldc_re_radios = {}
        ldc_re_values = [100, 400, 1000, 3200, 5000, 7500]
        for Re in ldc_re_values:
            radio = QRadioButton(str(Re))
            self.ldc_re_radios[Re] = radio
            self.ldc_re_button_group.addButton(radio, Re)
            self.ldc_re_row.addWidget(radio)
            if Re == 1000:  # Default selection
                radio.setChecked(True)
        # Connect button group signal outside the loop (only once)
        self.ldc_re_button_group.idClicked.connect(self.on_ldc_re_selected)
        self.ldc_re_row.addStretch()
        flow_layout.addWidget(self.ldc_re_widget)
        self.ldc_re_widget.setVisible(False)  # Initially hidden
        
        # Add separator line
        from PyQt6.QtWidgets import QFrame
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        flow_layout.addWidget(separator)
        
        # Add KH parameters directly to flow type group
        self.kh_widget = QWidget()
        kh_layout = QGridLayout(self.kh_widget)
        kh_layout.setSpacing(5)
        kh_layout.setColumnStretch(3, 1)  # Stretch last column

        # Row 0: Shear Velocity (U0)
        kh_layout.addWidget(QLabel("Shear Velocity (U₀):"), 0, 0)
        self.kh_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.kh_strength_slider.setRange(1, 50)  # 0.1 to 5.0
        self.kh_strength_slider.setValue(10)  # Default 1.0
        self.kh_strength_slider.setMaximumWidth(100)
        kh_layout.addWidget(self.kh_strength_slider, 0, 1)
        self.kh_strength_label = QLabel("1.0")
        self.kh_strength_label.setMinimumWidth(30)
        kh_layout.addWidget(self.kh_strength_label, 0, 2)
        self.apply_kh_strength_btn = QPushButton("Apply")
        self.apply_kh_strength_btn.setMaximumWidth(50)
        kh_layout.addWidget(self.apply_kh_strength_btn, 0, 3)

        # Row 1: Perturbation Amplitude
        kh_layout.addWidget(QLabel("Perturbation:"), 1, 0)
        self.kh_perturbation_slider = QSlider(Qt.Orientation.Horizontal)
        self.kh_perturbation_slider.setRange(1, 100)  # 0.001 to 0.1
        self.kh_perturbation_slider.setValue(10)  # Default 0.01
        self.kh_perturbation_slider.setMaximumWidth(100)
        kh_layout.addWidget(self.kh_perturbation_slider, 1, 1)
        self.kh_perturbation_label = QLabel("0.01")
        self.kh_perturbation_label.setMinimumWidth(30)
        kh_layout.addWidget(self.kh_perturbation_label, 1, 2)
        self.apply_kh_perturbation_btn = QPushButton("Apply")
        self.apply_kh_perturbation_btn.setMaximumWidth(50)
        kh_layout.addWidget(self.apply_kh_perturbation_btn, 1, 3)

        # Row 2: Shear Layer Thickness
        kh_layout.addWidget(QLabel("Shear Thickness:"), 2, 0)
        self.kh_thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self.kh_thickness_slider.setRange(5, 50)  # 0.05 to 0.5
        self.kh_thickness_slider.setValue(10)  # Default 0.1
        self.kh_thickness_slider.setMaximumWidth(100)
        kh_layout.addWidget(self.kh_thickness_slider, 2, 1)
        self.kh_thickness_label = QLabel("0.10")
        self.kh_thickness_label.setMinimumWidth(30)
        kh_layout.addWidget(self.kh_thickness_label, 2, 2)
        self.apply_kh_thickness_btn = QPushButton("Apply")
        self.apply_kh_thickness_btn.setMaximumWidth(50)
        kh_layout.addWidget(self.apply_kh_thickness_btn, 2, 3)

        # Initially hide KH parameters widget
        self.kh_widget.setVisible(False)

        # Connect KH slider signals after they are created
        self.kh_strength_slider.valueChanged.connect(self._update_kh_strength_label)
        self.kh_perturbation_slider.valueChanged.connect(self._update_kh_perturbation_label)
        self.kh_thickness_slider.valueChanged.connect(self._update_kh_thickness_label)
        self.apply_kh_strength_btn.clicked.connect(self.on_kh_strength_changed)
        self.apply_kh_perturbation_btn.clicked.connect(self.on_kh_perturbation_changed)
        self.apply_kh_thickness_btn.clicked.connect(self.on_kh_thickness_changed)

        flow_layout.addWidget(self.kh_widget)
        
        flow_group.setLayout(flow_layout)
        sidebar_layout.addWidget(flow_group)

        # ========== PRECISION GROUP ==========
        precision_group = CollapsibleGroupBox("Precision", start_collapsed=True)
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "float64"])
        self.precision_combo.setMaximumWidth(100)
        precision_layout.addWidget(self.precision_combo)
        self.apply_precision_btn = QPushButton("Apply")
        self.apply_precision_btn.setMaximumWidth(60)
        precision_layout.addWidget(self.apply_precision_btn)
        precision_layout.addStretch()
        precision_group.setLayout(precision_layout)
        sidebar_layout.addWidget(precision_group)

        # ========== SOLVER PARAMETERS GROUP ==========
        solver_group = CollapsibleGroupBox("Solver Parameters", start_collapsed=True)
        solver_layout = QGridLayout()
        solver_layout.setSpacing(5)
        solver_layout.setColumnStretch(4, 1)  # Stretch last column

        # Row 0: MG V-cycles
        solver_layout.addWidget(QLabel("MG V-cycles:"), 0, 0)
        self.vcycles_slider = QSlider(Qt.Orientation.Horizontal)
        self.vcycles_slider.setRange(1, 10)
        self.vcycles_slider.setValue(5)  # Default V-cycles
        self.vcycles_slider.setMaximumWidth(100)
        solver_layout.addWidget(self.vcycles_slider, 0, 1)
        self.vcycles_label = QLabel("5")
        self.vcycles_label.setMinimumWidth(20)
        solver_layout.addWidget(self.vcycles_label, 0, 2)
        self.apply_vcycles_btn = QPushButton("Apply")
        self.apply_vcycles_btn.setMaximumWidth(50)
        solver_layout.addWidget(self.apply_vcycles_btn, 0, 3)
        self.vcycles_slider.valueChanged.connect(self._update_vcycles_label)

        # Row 1: Hyper ν
        solver_layout.addWidget(QLabel("Hyper ν:"), 1, 0)
        self.hyper_viscosity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hyper_viscosity_slider.setRange(0, 100)
        self.hyper_viscosity_slider.setValue(0)
        self.hyper_viscosity_slider.setMaximumWidth(100)
        solver_layout.addWidget(self.hyper_viscosity_slider, 1, 1)
        self.hyper_viscosity_spinbox = QDoubleSpinBox()
        self.hyper_viscosity_spinbox.setRange(0.0, 0.05)
        self.hyper_viscosity_spinbox.setSingleStep(0.0001)
        self.hyper_viscosity_spinbox.setDecimals(4)
        self.hyper_viscosity_spinbox.setValue(0.0)
        self.hyper_viscosity_spinbox.setMaximumWidth(80)
        solver_layout.addWidget(self.hyper_viscosity_spinbox, 1, 2)
        self.apply_hyper_viscosity_btn = QPushButton("Apply")
        self.apply_hyper_viscosity_btn.setMaximumWidth(50)
        solver_layout.addWidget(self.apply_hyper_viscosity_btn, 1, 3)
        self.hyper_viscosity_slider.valueChanged.connect(self._update_hyper_viscosity_label)
        self.hyper_viscosity_spinbox.valueChanged.connect(self._update_hyper_viscosity_spinbox)

        # Row 2: Fast mode (RK2)
        solver_layout.addWidget(QLabel("Fast Mode (RK2):"), 2, 0)
        self.fast_mode_checkbox = QCheckBox("Enable")
        self.fast_mode_checkbox.setChecked(False)
        self.fast_mode_checkbox.setToolTip("RK2 for speed (Real-Time Interaction)\nRK3 for accuracy")
        solver_layout.addWidget(self.fast_mode_checkbox, 2, 1, 1, 2)

        # Row 3: LES controls
        solver_layout.addWidget(QLabel("LES:"), 3, 0)
        self.les_checkbox = QCheckBox("Enable")
        self.les_checkbox.setChecked(False)
        self.les_checkbox.stateChanged.connect(self._on_les_checkbox_changed)
        solver_layout.addWidget(self.les_checkbox, 3, 1)
        self.les_model_combo = QComboBox()
        self.les_model_combo.addItems(["dynamic_smagorinsky", "smagorinsky"])
        self.les_model_combo.setMaximumWidth(150)
        self.les_model_combo.setEnabled(False)
        solver_layout.addWidget(self.les_model_combo, 3, 2)
        self.apply_les_btn = QPushButton("Apply")
        self.apply_les_btn.setMaximumWidth(50)
        self.apply_les_btn.setEnabled(False)
        solver_layout.addWidget(self.apply_les_btn, 3, 3)

        # Row 4: Pressure solver
        solver_layout.addWidget(QLabel("Pressure Solver:"), 4, 0)
        self.pressure_solver_combo = QComboBox()
        self.pressure_solver_combo.addItems(["multigrid", "cg", "fft", "jacobi"])
        self.pressure_solver_combo.setMaximumWidth(150)
        solver_layout.addWidget(self.pressure_solver_combo, 4, 1, 1, 2)
        self.apply_pressure_solver_btn = QPushButton("Apply")
        self.apply_pressure_solver_btn.setMaximumWidth(50)
        solver_layout.addWidget(self.apply_pressure_solver_btn, 4, 3)

        solver_group.setLayout(solver_layout)
        sidebar_layout.addWidget(solver_group)

        # ========== BOUNDARY CONDITIONS GROUP ==========
        boundary_group = CollapsibleGroupBox("Boundary Conditions", start_collapsed=True)
        boundary_layout = QVBoxLayout()
        slip_row = QHBoxLayout()
        self.slip_walls_checkbox = QCheckBox("Slip Walls")
        self.slip_walls_checkbox.setChecked(True)
        slip_row.addWidget(self.slip_walls_checkbox)
        slip_row.addStretch()
        boundary_layout.addLayout(slip_row)
        epsilon_row = QHBoxLayout()
        epsilon_row.addWidget(QLabel("Mask ε:"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 10000)  # Range for eps_multiplier (divided by 1000 in handler)
        self.epsilon_slider.setValue(100)  # Default 0.1 (matches default eps_multiplier in params.py)
        self.epsilon_slider.setMaximumWidth(100)
        epsilon_row.addWidget(self.epsilon_slider)
        self.epsilon_label = QLabel("0.10")
        self.epsilon_label.setMinimumWidth(35)
        epsilon_row.addWidget(self.epsilon_label)
        self.apply_epsilon_btn = QPushButton("Apply")
        self.apply_epsilon_btn.setMaximumWidth(50)
        epsilon_row.addWidget(self.apply_epsilon_btn)
        epsilon_row.addStretch()
        boundary_layout.addLayout(epsilon_row)
        self.epsilon_slider.valueChanged.connect(self._update_epsilon_label)
        boundary_group.setLayout(boundary_layout)
        sidebar_layout.addWidget(boundary_group)

        # ========== SIMULATION INFO GROUP ==========
        info_group = CollapsibleGroupBox("Simulation Info", start_collapsed=True)
        info_layout = QVBoxLayout()
        solver_row = QHBoxLayout()
        self.solver_status_label = QLabel("Solver: Not initialized")
        self.solver_status_label.setMinimumWidth(150)
        solver_row.addWidget(self.solver_status_label)
        solver_row.addStretch()
        info_layout.addLayout(solver_row)
        fps_row = QHBoxLayout()
        self.sim_fps_label = QLabel("Sim FPS: 0")
        self.sim_fps_label.setMinimumWidth(80)
        fps_row.addWidget(self.sim_fps_label)
        self.viz_fps_label = QLabel("Vis FPS: 0")
        self.viz_fps_label.setMinimumWidth(80)
        fps_row.addWidget(self.viz_fps_label)
        fps_row.addStretch()
        info_layout.addLayout(fps_row)
        time_row = QHBoxLayout()
        self.sim_time_label = QLabel("Time: 0.000")
        self.sim_time_label.setMinimumWidth(100)
        time_row.addWidget(self.sim_time_label)
        self.dt_label = QLabel("dt: 0.0000")
        self.dt_label.setMinimumWidth(80)
        time_row.addWidget(self.dt_label)
        time_row.addStretch()
        info_layout.addLayout(time_row)
        div_row = QHBoxLayout()
        self.max_div_label = QLabel("RMS Divergence: 0.000")
        self.max_div_label.setMinimumWidth(150)
        div_row.addWidget(self.max_div_label)
        div_row.addStretch()
        info_layout.addLayout(div_row)
        info_group.setLayout(info_layout)
        sidebar_layout.addWidget(info_group)

        # ========== OBSTACLE CONTROLS (moved to right panel) ==========
        self.obstacle_controls = ObstacleControls(self)
        # Not added to sidebar_layout - moved to right_control_panel

        # ========== VISUALIZATION CONTROLS ==========
        self.visualization_controls = VisualizationControls(self)
        sidebar_layout.addWidget(self.visualization_controls)

        # ========== DYE INJECTION CONTROLS ==========
        self.dye_controls = DyeControls(self)
        sidebar_layout.addWidget(self.dye_controls)

        # ========== TIME STEPPING CONTROLS ==========
        self.time_controls = TimeControls(self)
        sidebar_layout.addWidget(self.time_controls)

        # ========== NEURAL OPERATOR TRAINING CONTROLS ==========
        neural_group = CollapsibleGroupBox("Neural Operator Training", start_collapsed=True)
        neural_layout = QVBoxLayout()
        self.neural_operator_training = NeuralOperatorTraining(self)
        neural_layout.addWidget(self.neural_operator_training)
        neural_group.setLayout(neural_layout)
        sidebar_layout.addWidget(neural_group)

        sidebar_layout.addStretch()

        # Set sidebar content as scroll area widget
        scroll_area.setWidget(sidebar_content)
        main_layout.addWidget(scroll_area, 1)

        self.setLayout(main_layout)

    def _update_epsilon_label(self):
        """Update epsilon label when slider changes."""
        value = self.epsilon_slider.value() / 100.0
        self.epsilon_label.setText(f"{value:.2f}")

    def _update_tau_from_slider(self):
        """Update tau spinbox when slider changes."""
        value = self.tau_slider.value() / 100.0
        self.tau_spinbox.blockSignals(True)
        self.tau_spinbox.setValue(value)
        self.tau_spinbox.blockSignals(False)

    def _update_tau_from_spinbox(self):
        """Update tau slider when spinbox changes."""
        value = self.tau_spinbox.value()
        slider_value = int(value * 100)
        self.tau_slider.blockSignals(True)
        self.tau_slider.setValue(slider_value)
        self.tau_slider.blockSignals(False)

    def on_solver_type_index_changed(self, index: int) -> None:
        """Handle solver type dropdown change - show/hide LBM Tau controls and update flow options."""
        # Get solver type from selected index
        solver_type = self.solver_type_combo.itemData(index)
        is_lbm = (solver_type == "lattice_boltzmann")
        
        print(f"Solver type changed to: {solver_type} (index: {index})")
        print(f"Is LBM: {is_lbm}")
        
        # Show Tau controls only for lattice_boltzmann
        self.lbm_tau_widget.setVisible(is_lbm)
        
        # Show Outlet controls only for lattice_boltzmann
        self.lbm_outlet_widget.setVisible(is_lbm)
        
        # Show BC mode controls only for lattice_boltzmann
        self.lbm_bc_mode_widget.setVisible(is_lbm)
        
        # Update flow combo to include/exclude Kelvin-Helmholtz based on solver type
        current_flow = self.flow_combo.currentText()
        self.flow_combo.clear()
        
        if is_lbm:
            # KH is only available for LBM (periodic, initial-condition-driven)
            self.flow_combo.addItems(["von_karman", "lid_driven_cavity", "taylor_green", "kelvin_helmholtz"])
        else:
            # Navier-Stokes: KH not available (requires LBM streaming)
            self.flow_combo.addItems(["von_karman", "lid_driven_cavity", "taylor_green"])
        
        # Restore previous selection if it's still valid
        if current_flow in [self.flow_combo.itemText(i) for i in range(self.flow_combo.count())]:
            self.flow_combo.setCurrentText(current_flow)
        else:
            # Default to von_karman if previous selection is no longer valid
            self.flow_combo.setCurrentText("von_karman")
        
        # Show/hide KH parameters widget based on flow type
        current_flow_after = self.flow_combo.currentText()
        show_kh_params = is_lbm and current_flow_after == 'kelvin_helmholtz'
        self.kh_widget.setVisible(show_kh_params)
        if show_kh_params:
            # Auto-expand the flow type groupbox to show KH parameters
            parent = self.kh_widget.parent()
            while parent is not None:
                if hasattr(parent, 'set_collapsed'):
                    parent.set_collapsed(False)
                    break
                parent = parent.parent()
        
        print(f"Tau widget visibility set to: {is_lbm}")
        print(f"KH parameters visibility set to: {show_kh_params}")
        
        # Auto-expand the solver type groupbox when LBM is selected
        if is_lbm:
            # Find the parent CollapsibleGroupBox and expand it
            parent = self.lbm_tau_widget.parent()
            while parent is not None:
                if hasattr(parent, 'set_collapsed'):
                    parent.set_collapsed(False)
                    print("Expanded solver type groupbox")
                    break
                parent = parent.parent()
        
        # Also delegate to parent viewer
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_solver_type_changed'):
                self.parent_viewer.on_solver_type_changed(solver_type)

    def _update_hyper_viscosity_label(self):
        """Update hyperviscosity label when slider changes."""
        value = self.hyper_viscosity_slider.value()
        percentage = value  # 0-100%
        ratio = percentage / 100.0 * 0.05  # Convert to 0.0-0.05 range
        self.hyper_viscosity_label.setText(f"{percentage}%")
        self.hyper_viscosity_spinbox.blockSignals(True)
        self.hyper_viscosity_spinbox.setValue(ratio)
        self.hyper_viscosity_spinbox.blockSignals(False)

    def _update_hyper_viscosity_spinbox(self):
        """Update hyperviscosity slider when spinbox changes."""
        value = self.hyper_viscosity_spinbox.value()
        percentage = int((value / 0.05) * 100)  # Convert to 0-100% range
        self.hyper_viscosity_slider.blockSignals(True)
        self.hyper_viscosity_slider.setValue(percentage)
        self.hyper_viscosity_slider.blockSignals(False)
        self.hyper_viscosity_label.setText(f"{percentage}%")

    def _update_vcycles_label(self):
        """Update V-cycles label when slider changes."""
        value = self.vcycles_slider.value()
        self.vcycles_label.setText(str(value))

    def _update_kh_strength_label(self):
        """Update KH strength label when slider changes."""
        value = self.kh_strength_slider.value()
        strength = value / 10.0  # Convert to 0.1-5.0 range
        self.kh_strength_label.setText(f"{strength:.1f}")

    def _update_kh_perturbation_label(self):
        """Update KH perturbation label when slider changes."""
        value = self.kh_perturbation_slider.value()
        perturbation = value / 1000.0  # Convert to 0.001-0.1 range
        self.kh_perturbation_label.setText(f"{perturbation:.3f}")

    def _update_kh_thickness_label(self):
        """Update KH thickness label when slider changes."""
        value = self.kh_thickness_slider.value()
        thickness = value / 100.0  # Convert to 0.05-0.5 range
        self.kh_thickness_label.setText(f"{thickness:.2f}")

    def on_kh_strength_changed(self):
        """Handle KH strength slider change."""
        value = self.kh_strength_slider.value()
        strength = value / 10.0  # Convert to 0.1-5.0 range
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_kh_parameters'):
                self.parent_viewer.apply_kh_parameters('strength', strength)

    def on_kh_perturbation_changed(self):
        """Handle KH perturbation slider change."""
        value = self.kh_perturbation_slider.value()
        perturbation = value / 1000.0  # Convert to 0.001-0.1 range
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_kh_parameters'):
                self.parent_viewer.apply_kh_parameters('perturbation', perturbation)

    def on_kh_thickness_changed(self):
        """Handle KH thickness slider change."""
        value = self.kh_thickness_slider.value()
        thickness = value / 100.0  # Convert to 0.05-0.5 range
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_kh_parameters'):
                self.parent_viewer.apply_kh_parameters('thickness', thickness)

    def _on_les_checkbox_changed(self, state):
        """Enable/disable LES controls when checkbox is toggled"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.les_model_combo.setEnabled(is_checked)
        self.apply_les_btn.setEnabled(is_checked)

    def _on_slip_walls_changed(self, state):
        """Handle slip walls checkbox state change."""
        is_slip = (state == 2)  # Qt.CheckState.Checked
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_wall_boundary_condition'):
                self.parent_viewer.apply_wall_boundary_condition(is_slip)

    # Expose child component attributes for backward compatibility
    @property
    def start_btn(self):
        return self.top_console.start_btn

    @property
    def pause_btn(self):
        return self.top_console.pause_btn

    @property
    def reset_btn(self):
        return self.top_console.reset_btn

    @property
    def theme_toggle_btn(self):
        return self.top_console.theme_toggle_btn

    # Obstacle controls
    @property
    def naca_combo(self):
        return self.obstacle_controls.naca_combo

    @property
    def chord_spinbox(self):
        return self.obstacle_controls.chord_spinbox

    @property
    def angle_spinbox(self):
        return self.obstacle_controls.angle_spinbox

    @property
    def angle_slider(self):
        return self.obstacle_controls.angle_slider

    @property
    def apply_naca_btn(self):
        return self.obstacle_controls.apply_naca_btn

    @property
    def cylinder_radius_spinbox(self):
        return self.obstacle_controls.cylinder_radius_spinbox

    @property
    def apply_cylinder_btn(self):
        return self.obstacle_controls.apply_cylinder_btn

    @property
    def cylinder_diameter_spinbox(self):
        return self.obstacle_controls.cylinder_diameter_spinbox

    @property
    def cylinder_spacing_spinbox(self):
        return self.obstacle_controls.cylinder_spacing_spinbox

    @property
    def apply_cylinder_array_btn(self):
        return self.obstacle_controls.apply_cylinder_array_btn

    @property
    def x_position_slider(self):
        return self.obstacle_controls.x_position_slider

    @property
    def x_position_label(self):
        return self.obstacle_controls.x_position_label

    @property
    def y_position_slider(self):
        return self.obstacle_controls.y_position_slider

    @property
    def y_position_label(self):
        return self.obstacle_controls.y_position_label

    @property
    def dynamic_airfoil_checkbox(self):
        return self.obstacle_controls.dynamic_airfoil_checkbox

    @property
    def min_aoa_spinbox(self):
        return self.obstacle_controls.min_aoa_spinbox

    @property
    def max_aoa_spinbox(self):
        return self.obstacle_controls.max_aoa_spinbox

    @property
    def aoa_increment_spinbox(self):
        return self.obstacle_controls.aoa_increment_spinbox

    @property
    def steps_per_increment_slider(self):
        return self.obstacle_controls.steps_per_increment_slider

    @property
    def cylinder_radio(self):
        return self.obstacle_controls.cylinder_radio

    @property
    def naca_radio(self):
        return self.obstacle_controls.naca_radio

    @property
    def cow_radio(self):
        return self.obstacle_controls.cow_radio

    @property
    def cylinder_array_radio(self):
        return self.obstacle_controls.cylinder_array_radio

    @property
    def obstacle_button_group(self):
        return self.obstacle_controls.obstacle_button_group

    @property
    def draw_custom_btn(self):
        return self.obstacle_controls.draw_custom_btn

    # Time controls
    @property
    def dt_spinbox(self):
        return self.time_controls.dt_spinbox

    @property
    def apply_dt_btn(self):
        return self.time_controls.apply_dt_btn

    @property
    def adaptive_dt_checkbox(self):
        return self.time_controls.adaptive_dt_checkbox

    @property
    def cfl_label(self):
        return self.time_controls.cfl_label

    # Visualization controls
    @property
    def frame_skip_input(self):
        return self.visualization_controls.frame_skip_input

    @property
    def apply_frame_skip_btn(self):
        return self.visualization_controls.apply_frame_skip_btn

    @property
    def vis_fps_input(self):
        return self.visualization_controls.vis_fps_input

    @property
    def apply_vis_fps_btn(self):
        return self.visualization_controls.apply_vis_fps_btn

    @property
    def show_velocity_checkbox(self):
        return self.visualization_controls.show_velocity_checkbox

    @property
    def show_vorticity_checkbox(self):
        return self.visualization_controls.show_vorticity_checkbox

    @property
    def show_pressure_checkbox(self):
        return self.visualization_controls.show_pressure_checkbox

    @property
    def show_dye_checkbox(self):
        return self.visualization_controls.show_dye_checkbox

    @property
    def particle_mode_checkbox(self):
        return self.visualization_controls.particle_mode_checkbox

    @property
    def show_sdf_checkbox(self):
        return self.visualization_controls.show_sdf_checkbox

    @property
    def show_streamlines_checkbox(self):
        return self.visualization_controls.show_streamlines_checkbox

    @property
    def log_colorscale_checkbox(self):
        return self.visualization_controls.log_colorscale_checkbox

    @property
    def spatial_colorscale_checkbox(self):
        return self.visualization_controls.spatial_colorscale_checkbox

    @property
    def adaptive_colorscale_checkbox(self):
        return self.visualization_controls.adaptive_colorscale_checkbox

    @property
    def show_quivers_checkbox(self):
        return self.visualization_controls.show_quivers_checkbox

    @property
    def upscale_slider(self):
        return self.visualization_controls.upscale_slider

    @property
    def upscale_label(self):
        return self.visualization_controls.upscale_label

    @property
    def velocity_colormap_combo(self):
        return self.visualization_controls.velocity_colormap_combo

    @property
    def vorticity_colormap_combo(self):
        return self.visualization_controls.vorticity_colormap_combo

    @property
    def pressure_colormap_combo(self):
        return self.visualization_controls.pressure_colormap_combo

    @property
    def export_btn(self):
        return self.visualization_controls.export_btn

    @property
    def record_btn(self):
        return self.visualization_controls.record_btn

    @property
    def save_btn(self):
        return self.visualization_controls.save_video_btn

    @property
    def save_state_btn(self):
        return self.visualization_controls.save_state_btn

    @property
    def load_state_btn(self):
        return self.visualization_controls.load_state_btn

    @property
    def autofit_velocity_btn(self):
        return self.visualization_controls.autofit_velocity_btn

    @property
    def autofit_vorticity_btn(self):
        return self.visualization_controls.autofit_vorticity_btn

    @property
    def autofit_pressure_btn(self):
        return self.visualization_controls.autofit_pressure_btn

    @property
    def autofit_dye_btn(self):
        return self.visualization_controls.autofit_dye_btn

    @property
    def autofit_all_btn(self):
        return self.visualization_controls.autofit_all_btn

    # Dye controls
    @property
    def dye_x_input(self):
        return self.dye_controls.dye_x_input

    @property
    def dye_y_input(self):
        return self.dye_controls.dye_y_input

    @property
    def dye_amount_slider(self):
        return self.dye_controls.dye_amount_slider

    @property
    def dye_amount_label(self):
        return self.dye_controls.dye_amount_label

    @property
    def inject_dye_btn(self):
        return self.dye_controls.inject_dye_btn

    @property
    def dye_x_slider(self):
        return self.dye_controls.dye_x_slider

    @property
    def dye_y_slider(self):
        return self.dye_controls.dye_y_slider

    # Methods for backward compatibility
    def set_chord_range_for_domain(self, max_chord: float):
        """Update chord spinbox range based on domain size"""
        self.obstacle_controls.set_chord_range_for_domain(max_chord)

    def show_naca_controls(self, show: bool) -> None:
        """Show/hide NACA controls based on obstacle selection"""
        self.obstacle_controls.show_naca_controls(show)

    def _on_les_checkbox_changed(self, state):
        """Enable/disable LES controls when checkbox is toggled"""
        is_checked = state == 2  # Qt.CheckState.Checked
        self.les_model_combo.setEnabled(is_checked)
        self.apply_les_btn.setEnabled(is_checked)

    def _on_obstacle_radio_changed(self, button):
        """Handle obstacle type radio button selection changes"""
        self.obstacle_controls._on_obstacle_radio_changed(button)

    def _on_x_position_changed(self, value):
        """Handle x-position slider changes"""
        self.obstacle_controls._on_x_position_changed(value)

    def _on_y_position_changed(self, value):
        """Handle y-position slider changes"""
        self.obstacle_controls._on_y_position_changed(value)

    def _on_naca_hover(self, index):
        """Show airfoil preview when selection changes"""
        self.obstacle_controls._on_naca_hover(index)


    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        return self.obstacle_controls._check_naca_availability()

    def _populate_velocity_colormaps(self):
        """Populate velocity colormap dropdown"""
        self.visualization_controls._populate_velocity_colormaps()

    def _populate_vorticity_colormaps(self):
        """Populate vorticity colormap dropdown"""
        self.visualization_controls._populate_vorticity_colormaps()

    def on_obstacle_type_selected(self, obstacle_type: str) -> None:
        """Delegate obstacle type selection to parent viewer (FlowManager)"""
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_obstacle_type_selected'):
                self.parent_viewer.on_obstacle_type_selected(obstacle_type)
    
    def on_flow_type_changed(self, flow_type: str) -> None:
        """Handle flow type dropdown change - show/hide LDC benchmark controls and KH parameters."""
        # Show LDC Re radio buttons only for lid_driven_cavity
        is_ldc = (flow_type == "lid_driven_cavity")
        self.ldc_re_widget.setVisible(is_ldc)
        
        # Show KH parameters only for kelvin_helmholtz with LBM solver
        is_kh = (flow_type == "kelvin_helmholtz")
        # Check if current solver is LBM
        solver_type = self.solver_type_combo.currentData()
        is_lbm = (solver_type == "lattice_boltzmann")
        show_kh_params = is_kh and is_lbm
        
        self.kh_widget.setVisible(show_kh_params)
        if show_kh_params:
            # Auto-expand the flow type groupbox to show KH parameters
            parent = self.kh_widget.parent()
            while parent is not None:
                if hasattr(parent, 'set_collapsed'):
                    parent.set_collapsed(False)
                    break
                parent = parent.parent()
        
        print(f"Flow type changed to: {flow_type}")
        print(f"KH parameters visibility set to: {show_kh_params}")
        
        # Also delegate to parent viewer
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_flow_type_changed'):
                self.parent_viewer.on_flow_type_changed(flow_type)
    
    def on_ldc_re_selected(self, re_value: int) -> None:
        """Handle LDC benchmark Re radio button selection."""
        # Guard against incorrect calls
        if re_value is None:
            return
        
        # Update Reynolds number input
        self.re_input.setValue(float(re_value))
        self.lock_re_cb.setChecked(True)
        self.lock_nu_cb.setChecked(False)  # Allow viscosity to be computed
        
        # Delegate to parent viewer to apply the Re change
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_ldc_re_selected'):
                self.parent_viewer.on_ldc_re_selected(re_value)
    
    def on_outlet_type_changed(self) -> None:
        """Handle outlet type dropdown change."""
        outlet_type = self.outlet_type_combo.currentData()
        print(f"Outlet type changed to: {outlet_type}")
        
        # Delegate to parent viewer to apply the outlet type change
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_outlet_type_changed'):
                self.parent_viewer.on_outlet_type_changed(outlet_type)
    
    def on_bc_mode_changed(self) -> None:
        """Handle BC mode dropdown change."""
        bc_mode = self.bc_mode_combo.currentData()
        print(f"BC mode changed to: {bc_mode}")
        
        # Delegate to parent viewer to apply the BC mode change
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_bc_mode_changed'):
                self.parent_viewer.on_bc_mode_changed(bc_mode)
