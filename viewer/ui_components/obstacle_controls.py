"""
Obstacle control widgets for the CFD viewer.
Handles obstacle type selection, NACA airfoil parameters, and cylinder parameters.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QRadioButton, QButtonGroup, QComboBox, QDoubleSpinBox,
                             QSlider, QPushButton, QSizePolicy, QGridLayout, QFileDialog, QInputDialog, QDialog, QLineEdit, QSpinBox)
from PyQt6.QtCore import Qt, QUrl, QProcess
from PyQt6.QtGui import QDesktopServices
import webbrowser
import numpy as np
import subprocess
import os
from viewer.state import store, set_obstacle_position, set_obstacle_type
from .collapsible_groupbox import CollapsibleGroupBox


class BBoxDialog(QDialog):
    """Dialog for selecting bounding box with templates"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Area")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Location templates
        layout.addWidget(QLabel("Select a location template:"))
        self.location_combo = QComboBox()
        self.location_combo.addItem("Custom (enter below)", "")
        self.location_combo.addItem("World Trade Center, NYC", "-74.0175,40.7115,-74.0065,40.7185")
        self.location_combo.addItem("Manhattan Downtown, NYC", "-74.0200,40.7000,-73.9900,40.7200")
        self.location_combo.addItem("Central Park, NYC", "-73.9810,40.7630,-73.9580,40.8000")
        self.location_combo.addItem("San Francisco Downtown", "-122.4200,37.7700,-122.3900,37.7900")
        self.location_combo.addItem("London City Centre", "-0.1000,51.5050,-0.0700,51.5200")
        self.location_combo.addItem("Tokyo Shibuya", "139.6800,35.6500,139.7100,35.6700")
        layout.addWidget(self.location_combo)
        
        # Custom bbox input
        layout.addWidget(QLabel("Or enter custom bbox (west,south,east,north):"))
        self.bbox_edit = QLineEdit()
        self.bbox_edit.setPlaceholderText("e.g., -74.02,40.68,-73.93,40.78")
        layout.addWidget(self.bbox_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect combo box to auto-fill custom input
        self.location_combo.currentIndexChanged.connect(self._on_location_changed)
    
    def _on_location_changed(self, index):
        try:
            if index > 0:  # Not custom
                bbox = self.location_combo.currentData()
                self.bbox_edit.setText(bbox)
        except Exception as e:
            print(f"Error updating bbox: {e}")
    
    def get_bbox(self):
        text = self.bbox_edit.text()
        if not text:
            return None
        try:
            west, south, east, north = [float(x.strip()) for x in text.split(',')]
            return west, south, east, north
        except ValueError:
            return None


class ObstacleControls(CollapsibleGroupBox):
    """Group for obstacle selection (cylinder / NACA airfoil / cow) and parameters"""

    def __init__(self, parent=None):
        super().__init__("Obstacle Configuration")
        self.parent_viewer = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup obstacle configuration UI"""
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Row 1: Obstacle type with radio buttons (stacked vertically)
        row1 = QVBoxLayout()
        row1.addWidget(QLabel("Type:"))

        self.obstacle_button_group = QButtonGroup()

        # Cylinder option
        cylinder_radio = QRadioButton("Cylinder")
        cylinder_radio.setChecked(False)
        self.obstacle_button_group.addButton(cylinder_radio, 0)
        row1.addWidget(cylinder_radio)

        # NACA airfoil option
        naca_radio = QRadioButton("NACA Airfoil")
        naca_radio.setChecked(True)
        self.obstacle_button_group.addButton(naca_radio, 1)
        row1.addWidget(naca_radio)

        # Cow option
        cow_radio = QRadioButton("Cow")
        self.obstacle_button_group.addButton(cow_radio, 2)
        row1.addWidget(cow_radio)

        # Three-cylinder array option
        cylinder_array_radio = QRadioButton("3 Cylinders")
        self.obstacle_button_group.addButton(cylinder_array_radio, 3)
        row1.addWidget(cylinder_array_radio)

        # Solid wall option
        solid_wall_radio = QRadioButton("Solid Wall")
        self.obstacle_button_group.addButton(solid_wall_radio, 4)
        row1.addWidget(solid_wall_radio)

        # Urban map option
        urban_map_radio = QRadioButton("Urban Map")
        self.obstacle_button_group.addButton(urban_map_radio, 5)
        row1.addWidget(urban_map_radio)

        # Tesla valve option
        tesla_valve_radio = QRadioButton("Tesla Valve")
        self.obstacle_button_group.addButton(tesla_valve_radio, 6)
        row1.addWidget(tesla_valve_radio)

        # Store radio buttons for later access
        self.cylinder_radio = cylinder_radio
        self.naca_radio = naca_radio
        self.cow_radio = cow_radio
        self.cylinder_array_radio = cylinder_array_radio
        self.solid_wall_radio = solid_wall_radio
        self.urban_map_radio = urban_map_radio
        self.tesla_valve_radio = tesla_valve_radio

        # Connect button group to obstacle type selection
        self.obstacle_button_group.buttonClicked.connect(self._on_obstacle_radio_changed)

        layout.addLayout(row1)

        # Row 2: NACA controls (initially visible)
        self.naca_widget = QWidget()
        naca_layout = QVBoxLayout(self.naca_widget)
        naca_layout.setContentsMargins(0, 0, 0, 0)
        naca_layout.setSpacing(5)

        # Row 0: NACA combo
        naca_row = QHBoxLayout()
        naca_row.addWidget(QLabel("NACA:"))
        self.naca_combo = QComboBox()
        if self._check_naca_availability():
            from obstacles.naca_airfoils import NACA_AIRFOILS
            self.naca_combo.addItems(list(NACA_AIRFOILS.keys()))
            self.naca_combo.setCurrentText("NACA 0012")
        else:
            self.naca_combo.addItems(["NACA 0012"])
        self.naca_combo.setMaximumWidth(150)
        self.naca_combo.setMouseTracking(True)
        self.naca_combo.currentIndexChanged.connect(self._on_naca_hover)
        naca_row.addWidget(self.naca_combo)
        naca_row.addStretch()
        naca_layout.addLayout(naca_row)

        # Row 1: Chord
        chord_row = QHBoxLayout()
        chord_row.addWidget(QLabel("Chord:"))
        self.chord_spinbox = QDoubleSpinBox()
        self.chord_spinbox.setRange(0.1, 5.0)
        self.chord_spinbox.setValue(3.0)
        self.chord_spinbox.setDecimals(2)
        self.chord_spinbox.setSingleStep(0.1)
        self.chord_spinbox.setMaximumWidth(120)
        chord_row.addWidget(self.chord_spinbox)
        chord_row.addStretch()
        naca_layout.addLayout(chord_row)

        # Row 2: AoA with slider and spinbox
        aoa_row = QHBoxLayout()
        aoa_row.addWidget(QLabel("AoA:"))
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(-20.0, 20.0)
        self.angle_spinbox.setValue(-10.0)
        self.angle_spinbox.setDecimals(1)
        self.angle_spinbox.setSingleStep(1.0)
        self.angle_spinbox.setMaximumWidth(80)
        aoa_row.addWidget(self.angle_spinbox)
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-200, 200)
        self.angle_slider.setValue(0)
        self.angle_slider.setMaximumWidth(120)
        aoa_row.addWidget(self.angle_slider)
        aoa_row.addStretch()
        naca_layout.addLayout(aoa_row)

        # Row 3: Apply button
        apply_row = QHBoxLayout()
        self.apply_naca_btn = QPushButton("Apply")
        self.apply_naca_btn.setMaximumWidth(60)
        apply_row.addWidget(self.apply_naca_btn)
        apply_row.addStretch()
        naca_layout.addLayout(apply_row)

        # Row 4: Dynamic airfoil motion controls
        dynamic_row = QHBoxLayout()
        self.dynamic_airfoil_checkbox = QSlider(Qt.Orientation.Horizontal)
        self.dynamic_airfoil_checkbox.setRange(0, 1)
        self.dynamic_airfoil_checkbox.setValue(0)
        self.dynamic_airfoil_checkbox.setMaximumWidth(40)
        self.dynamic_airfoil_checkbox.setToolTip("Enable dynamic airfoil motion")
        dynamic_row.addWidget(QLabel("Dynamic:"))
        dynamic_row.addWidget(self.dynamic_airfoil_checkbox)
        dynamic_row.addStretch()
        naca_layout.addLayout(dynamic_row)

        # Row 5: Min AoA
        min_aoa_row = QHBoxLayout()
        min_aoa_row.addWidget(QLabel("Min AoA:"))
        self.min_aoa_spinbox = QDoubleSpinBox()
        self.min_aoa_spinbox.setRange(-20.0, 20.0)
        self.min_aoa_spinbox.setValue(-10.0)
        self.min_aoa_spinbox.setDecimals(1)
        self.min_aoa_spinbox.setSingleStep(1.0)
        self.min_aoa_spinbox.setMaximumWidth(80)
        min_aoa_row.addWidget(self.min_aoa_spinbox)
        min_aoa_row.addStretch()
        naca_layout.addLayout(min_aoa_row)

        # Row 6: Max AoA
        max_aoa_row = QHBoxLayout()
        max_aoa_row.addWidget(QLabel("Max AoA:"))
        self.max_aoa_spinbox = QDoubleSpinBox()
        self.max_aoa_spinbox.setRange(-20.0, 20.0)
        self.max_aoa_spinbox.setValue(10.0)
        self.max_aoa_spinbox.setDecimals(1)
        self.max_aoa_spinbox.setSingleStep(1.0)
        self.max_aoa_spinbox.setMaximumWidth(80)
        max_aoa_row.addWidget(self.max_aoa_spinbox)
        max_aoa_row.addStretch()
        naca_layout.addLayout(max_aoa_row)

        # Row 7: AoA increment
        aoa_increment_row = QHBoxLayout()
        aoa_increment_row.addWidget(QLabel("AoA Step:"))
        self.aoa_increment_spinbox = QDoubleSpinBox()
        self.aoa_increment_spinbox.setRange(0.1, 5.0)
        self.aoa_increment_spinbox.setValue(1.0)
        self.aoa_increment_spinbox.setDecimals(1)
        self.aoa_increment_spinbox.setSingleStep(0.1)
        self.aoa_increment_spinbox.setMaximumWidth(80)
        aoa_increment_row.addWidget(self.aoa_increment_spinbox)
        aoa_increment_row.addStretch()
        naca_layout.addLayout(aoa_increment_row)

        # Row 8: Steps per increment
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_per_increment_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_per_increment_slider.setRange(10, 500)
        self.steps_per_increment_slider.setValue(100)
        self.steps_per_increment_slider.setMaximumWidth(120)
        self.steps_label = QLabel("100")
        self.steps_label.setMinimumWidth(40)
        steps_row.addWidget(self.steps_per_increment_slider)
        steps_row.addWidget(self.steps_label)
        steps_row.addStretch()
        naca_layout.addLayout(steps_row)

        # Connect steps slider to update label
        self.steps_per_increment_slider.valueChanged.connect(
            lambda v: self.steps_label.setText(str(v))
        )

        self.naca_widget.setVisible(True)
        layout.addWidget(self.naca_widget)

        # Row 3: Cylinder controls (initially hidden)
        self.cylinder_widget = QWidget()
        cylinder_layout = QHBoxLayout(self.cylinder_widget)
        cylinder_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_layout.setSpacing(8)

        cylinder_layout.addWidget(QLabel("Radius:"))
        self.cylinder_radius_spinbox = QDoubleSpinBox()
        self.cylinder_radius_spinbox.setRange(0.05, 2.0)
        self.cylinder_radius_spinbox.setValue(0.18)
        self.cylinder_radius_spinbox.setDecimals(3)
        self.cylinder_radius_spinbox.setSingleStep(0.01)
        self.cylinder_radius_spinbox.setMaximumWidth(120)
        cylinder_layout.addWidget(self.cylinder_radius_spinbox)

        self.apply_cylinder_btn = QPushButton("Apply")
        self.apply_cylinder_btn.setMaximumWidth(60)
        cylinder_layout.addWidget(self.apply_cylinder_btn)
        
        # Connect radius spinbox for live preview
        self.cylinder_radius_spinbox.valueChanged.connect(self._on_cylinder_radius_changed)

        cylinder_layout.addStretch()
        self.cylinder_widget.setVisible(False)
        layout.addWidget(self.cylinder_widget)

        # Row 4: Cylinder array controls (initially hidden)
        self.cylinder_array_widget = QWidget()
        cylinder_array_layout = QHBoxLayout(self.cylinder_array_widget)
        cylinder_array_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_array_layout.setSpacing(8)

        cylinder_array_layout.addWidget(QLabel("Diameter:"))
        self.cylinder_diameter_spinbox = QDoubleSpinBox()
        self.cylinder_diameter_spinbox.setRange(0.1, 2.0)
        self.cylinder_diameter_spinbox.setValue(0.5)
        self.cylinder_diameter_spinbox.setDecimals(2)
        self.cylinder_diameter_spinbox.setSingleStep(0.1)
        self.cylinder_diameter_spinbox.setMaximumWidth(120)
        cylinder_array_layout.addWidget(self.cylinder_diameter_spinbox)

        cylinder_array_layout.addWidget(QLabel("Spacing:"))
        self.cylinder_spacing_spinbox = QDoubleSpinBox()
        self.cylinder_spacing_spinbox.setRange(0.1, 5.0)
        self.cylinder_spacing_spinbox.setValue(0.5)
        self.cylinder_spacing_spinbox.setDecimals(2)
        self.cylinder_spacing_spinbox.setSingleStep(0.1)
        self.cylinder_spacing_spinbox.setMaximumWidth(120)
        cylinder_array_layout.addWidget(self.cylinder_spacing_spinbox)

        self.apply_cylinder_array_btn = QPushButton("Apply")
        self.apply_cylinder_array_btn.setMaximumWidth(60)
        cylinder_array_layout.addWidget(self.apply_cylinder_array_btn)

        cylinder_array_layout.addStretch()
        self.cylinder_array_widget.setVisible(False)
        layout.addWidget(self.cylinder_array_widget)

        # Row 4: Solid wall controls (initially hidden)
        self.solid_wall_widget = QWidget()
        solid_wall_layout = QVBoxLayout(self.solid_wall_widget)
        solid_wall_layout.setContentsMargins(0, 0, 0, 0)
        solid_wall_layout.setSpacing(5)

        # Add label explaining solid wall configuration
        info_label = QLabel("Solid wall configuration")
        info_label.setWordWrap(True)
        solid_wall_layout.addWidget(info_label)

        # Y-bottom slider
        y_bottom_layout = QHBoxLayout()
        y_bottom_layout.addWidget(QLabel("Y-Bottom:"))
        self.solid_wall_y_bottom_slider = QSlider(Qt.Orientation.Horizontal)
        self.solid_wall_y_bottom_slider.setRange(0, 50)
        self.solid_wall_y_bottom_slider.setValue(0)
        self.solid_wall_y_bottom_slider.setMaximumWidth(150)
        y_bottom_layout.addWidget(self.solid_wall_y_bottom_slider)
        self.solid_wall_y_bottom_label = QLabel("0%")
        self.solid_wall_y_bottom_label.setMinimumWidth(40)
        y_bottom_layout.addWidget(self.solid_wall_y_bottom_label)
        y_bottom_layout.addStretch()
        solid_wall_layout.addLayout(y_bottom_layout)

        # Y-top slider
        y_top_layout = QHBoxLayout()
        y_top_layout.addWidget(QLabel("Y-Top:"))
        self.solid_wall_y_top_slider = QSlider(Qt.Orientation.Horizontal)
        self.solid_wall_y_top_slider.setRange(50, 100)
        self.solid_wall_y_top_slider.setValue(50)
        self.solid_wall_y_top_slider.setMaximumWidth(150)
        y_top_layout.addWidget(self.solid_wall_y_top_slider)
        self.solid_wall_y_top_label = QLabel("50%")
        self.solid_wall_y_top_label.setMinimumWidth(40)
        y_top_layout.addWidget(self.solid_wall_y_top_label)
        y_top_layout.addStretch()
        solid_wall_layout.addLayout(y_top_layout)

        # Connect sliders to update labels and trigger updates
        self.solid_wall_y_bottom_slider.valueChanged.connect(self._on_solid_wall_y_bottom_changed)
        self.solid_wall_y_top_slider.valueChanged.connect(self._on_solid_wall_y_top_changed)

        self.solid_wall_widget.setVisible(False)
        layout.addWidget(self.solid_wall_widget)

        # Row 5: Urban map controls (initially hidden)
        self.urban_map_widget = QWidget()
        urban_map_layout = QVBoxLayout(self.urban_map_widget)
        urban_map_layout.setContentsMargins(0, 0, 0, 0)
        urban_map_layout.setSpacing(5)

        # Add label explaining urban map configuration
        urban_info_label = QLabel("Load building footprints from Overture Maps")
        urban_info_label.setWordWrap(True)
        urban_map_layout.addWidget(urban_info_label)

        # Download button
        download_layout = QHBoxLayout()
        self.download_btn = QPushButton("Download from Overture Maps")
        self.download_btn.setMaximumWidth(200)
        self.download_btn.setToolTip("Download building data using overturemaps CLI")
        download_layout.addWidget(self.download_btn)
        download_layout.addStretch()
        urban_map_layout.addLayout(download_layout)

        # Select area button
        select_area_layout = QHBoxLayout()
        self.select_area_btn = QPushButton("Open Map Explorer")
        self.select_area_btn.setMaximumWidth(200)
        self.select_area_btn.setToolTip("Open browser to select area on Overture Maps")
        select_area_layout.addWidget(self.select_area_btn)
        select_area_layout.addStretch()
        urban_map_layout.addLayout(select_area_layout)

        # Load file button
        load_file_layout = QHBoxLayout()
        self.load_geojson_btn = QPushButton("Load GeoJSON File")
        self.load_geojson_btn.setMaximumWidth(200)
        self.load_geojson_btn.setToolTip("Load downloaded GeoJSON file")
        load_file_layout.addWidget(self.load_geojson_btn)
        
        # Number of buildings spinbox
        num_buildings_layout = QHBoxLayout()
        num_buildings_layout.addWidget(QLabel("Max Buildings:"))
        self.max_buildings_spinbox = QSpinBox()
        self.max_buildings_spinbox.setMinimum(1)
        self.max_buildings_spinbox.setMaximum(100)
        self.max_buildings_spinbox.setValue(20)
        self.max_buildings_spinbox.setMaximumWidth(80)
        self.max_buildings_spinbox.setToolTip("Maximum number of buildings to import from GeoJSON")
        num_buildings_layout.addWidget(self.max_buildings_spinbox)
        num_buildings_layout.addStretch()
        load_file_layout.addLayout(num_buildings_layout)
        load_file_layout.addStretch()
        urban_map_layout.addLayout(load_file_layout)

        # Generate urban SDF button
        generate_layout = QHBoxLayout()
        self.generate_urban_btn = QPushButton("Generate Urban SDF")
        self.generate_urban_btn.setMaximumWidth(200)
        self.generate_urban_btn.setToolTip("Generate Manhattan-style urban SDF")
        generate_layout.addWidget(self.generate_urban_btn)
        generate_layout.addStretch()
        urban_map_layout.addLayout(generate_layout)

        # Status label
        self.urban_map_status_label = QLabel("No map loaded")
        self.urban_map_status_label.setWordWrap(True)
        self.urban_map_status_label.setStyleSheet("color: gray;")
        urban_map_layout.addWidget(self.urban_map_status_label)

        # Connect urban map buttons
        self.download_btn.clicked.connect(self._on_download_clicked)
        self.select_area_btn.clicked.connect(self._on_select_area_clicked)
        self.load_geojson_btn.clicked.connect(self._on_load_geojson_clicked)
        self.generate_urban_btn.clicked.connect(self._on_generate_urban_clicked)

        self.urban_map_widget.setVisible(False)
        layout.addWidget(self.urban_map_widget)

        # Row 6: Tesla valve controls (initially hidden)
        self.tesla_valve_widget = QWidget()
        tesla_valve_layout = QVBoxLayout(self.tesla_valve_widget)
        tesla_valve_layout.setContentsMargins(0, 0, 0, 0)
        tesla_valve_layout.setSpacing(5)

        # Add label explaining Tesla valve configuration
        tesla_info_label = QLabel("Tesla valve configuration")
        tesla_info_label.setWordWrap(True)
        tesla_valve_layout.addWidget(tesla_info_label)

        # Number of stages
        stages_layout = QHBoxLayout()
        stages_layout.addWidget(QLabel("Stages:"))
        self.tesla_valve_stages = QSpinBox()
        self.tesla_valve_stages.setRange(1, 10)
        self.tesla_valve_stages.setValue(3)
        self.tesla_valve_stages.setMaximumWidth(80)
        stages_layout.addWidget(self.tesla_valve_stages)
        stages_layout.addStretch()
        tesla_valve_layout.addLayout(stages_layout)

        # Stage length
        stage_len_layout = QHBoxLayout()
        stage_len_layout.addWidget(QLabel("Stage length:"))
        self.tesla_valve_stage_len = QDoubleSpinBox()
        self.tesla_valve_stage_len.setRange(0.5, 5.0)
        self.tesla_valve_stage_len.setValue(1.5)
        self.tesla_valve_stage_len.setDecimals(2)
        self.tesla_valve_stage_len.setSingleStep(0.1)
        self.tesla_valve_stage_len.setMaximumWidth(80)
        stage_len_layout.addWidget(self.tesla_valve_stage_len)
        stage_len_layout.addStretch()
        tesla_valve_layout.addLayout(stage_len_layout)

        # Main channel width
        main_width_layout = QHBoxLayout()
        main_width_layout.addWidget(QLabel("Main width:"))
        self.tesla_valve_main_width = QDoubleSpinBox()
        self.tesla_valve_main_width.setRange(0.1, 1.0)
        self.tesla_valve_main_width.setValue(0.4)
        self.tesla_valve_main_width.setDecimals(2)
        self.tesla_valve_main_width.setSingleStep(0.05)
        self.tesla_valve_main_width.setMaximumWidth(80)
        main_width_layout.addWidget(self.tesla_valve_main_width)
        main_width_layout.addStretch()
        tesla_valve_layout.addLayout(main_width_layout)

        # Branch width
        branch_width_layout = QHBoxLayout()
        branch_width_layout.addWidget(QLabel("Branch width:"))
        self.tesla_valve_branch_width = QDoubleSpinBox()
        self.tesla_valve_branch_width.setRange(0.05, 0.8)
        self.tesla_valve_branch_width.setValue(0.2)
        self.tesla_valve_branch_width.setDecimals(2)
        self.tesla_valve_branch_width.setSingleStep(0.05)
        self.tesla_valve_branch_width.setMaximumWidth(80)
        branch_width_layout.addWidget(self.tesla_valve_branch_width)
        branch_width_layout.addStretch()
        tesla_valve_layout.addLayout(branch_width_layout)

        # Diagonal length
        diagonal_len_layout = QHBoxLayout()
        diagonal_len_layout.addWidget(QLabel("Diagonal length:"))
        self.tesla_valve_diagonal_length = QDoubleSpinBox()
        self.tesla_valve_diagonal_length.setRange(0.1, 10.0)
        self.tesla_valve_diagonal_length.setValue(0.4)
        self.tesla_valve_diagonal_length.setDecimals(2)
        self.tesla_valve_diagonal_length.setSingleStep(0.1)
        self.tesla_valve_diagonal_length.setMaximumWidth(80)
        diagonal_len_layout.addWidget(self.tesla_valve_diagonal_length)
        diagonal_len_layout.addStretch()
        tesla_valve_layout.addLayout(diagonal_len_layout)

        # Branch angle (in degrees, then convert to radians)
        branch_angle_layout = QHBoxLayout()
        branch_angle_layout.addWidget(QLabel("Branch angle (°):"))
        self.tesla_valve_branch_angle = QDoubleSpinBox()
        self.tesla_valve_branch_angle.setRange(-80, 80)
        self.tesla_valve_branch_angle.setValue(35)
        self.tesla_valve_branch_angle.setDecimals(1)
        self.tesla_valve_branch_angle.setSingleStep(5)
        self.tesla_valve_branch_angle.setMaximumWidth(80)
        branch_angle_layout.addWidget(self.tesla_valve_branch_angle)
        branch_angle_layout.addStretch()
        tesla_valve_layout.addLayout(branch_angle_layout)

        # Forward/Backward checkbox
        tesla_direction_layout = QHBoxLayout()
        self.tesla_forward_checkbox = QSlider(Qt.Orientation.Horizontal)
        self.tesla_forward_checkbox.setRange(0, 1)
        self.tesla_forward_checkbox.setValue(1)  # Default to forward
        self.tesla_forward_checkbox.setMaximumWidth(40)
        self.tesla_forward_checkbox.setToolTip("Check for forward Tesla valve, uncheck for backward")
        tesla_direction_layout.addWidget(QLabel("Forward:"))
        tesla_direction_layout.addWidget(self.tesla_forward_checkbox)
        tesla_direction_layout.addWidget(QLabel("(unchecked = backward)"))
        tesla_direction_layout.addStretch()
        tesla_valve_layout.addLayout(tesla_direction_layout)

        # Apply button
        tesla_apply_layout = QHBoxLayout()
        self.apply_tesla_valve_btn = QPushButton("Apply")
        self.apply_tesla_valve_btn.setMaximumWidth(60)
        tesla_apply_layout.addWidget(self.apply_tesla_valve_btn)
        tesla_apply_layout.addStretch()
        tesla_valve_layout.addLayout(tesla_apply_layout)

        # Connect Tesla valve buttons
        self.apply_tesla_valve_btn.clicked.connect(self._on_tesla_valve_apply_clicked)
        self.tesla_forward_checkbox.valueChanged.connect(self._on_tesla_direction_changed)

        self.tesla_valve_widget.setVisible(False)
        layout.addWidget(self.tesla_valve_widget)

        # Row 7: X-position slider (always visible)
        x_pos_layout = QHBoxLayout()
        x_pos_layout.addWidget(QLabel("X-Position:"))
        self.x_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_position_slider.setRange(10, 90)
        self.x_position_slider.setValue(25)
        self.x_position_slider.setMaximumWidth(200)
        x_pos_layout.addWidget(self.x_position_slider)
        self.x_position_label = QLabel("25%")
        self.x_position_label.setMinimumWidth(40)
        x_pos_layout.addWidget(self.x_position_label)
        x_pos_layout.addStretch()
        layout.addLayout(x_pos_layout)

        # Row 6: Y-position slider (always visible)
        y_pos_layout = QHBoxLayout()
        y_pos_layout.addWidget(QLabel("Y-Position:"))
        self.y_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_position_slider.setRange(10, 90)
        self.y_position_slider.setValue(50)
        self.y_position_slider.setMaximumWidth(200)
        y_pos_layout.addWidget(self.y_position_slider)
        self.y_position_label = QLabel("50%")
        self.y_position_label.setMinimumWidth(40)
        y_pos_layout.addWidget(self.y_position_label)
        y_pos_layout.addStretch()
        layout.addLayout(y_pos_layout)

        # Row 7: Custom obstacle drawing button
        draw_button_layout = QHBoxLayout()
        self.draw_custom_btn = QPushButton("Draw Custom Obstacle")
        self.draw_custom_btn.setMaximumWidth(200)
        draw_button_layout.addWidget(self.draw_custom_btn)
        draw_button_layout.addStretch()
        layout.addLayout(draw_button_layout)

        # Row 8: Load PNG mask button
        png_button_layout = QHBoxLayout()
        self.load_png_mask_btn = QPushButton("Load PNG Mask")
        self.load_png_mask_btn.setMaximumWidth(200)
        self.load_png_mask_btn.setToolTip("Load PNG file as SDF mask (white=fluid, non-white=solid)")
        png_button_layout.addWidget(self.load_png_mask_btn)
        png_button_layout.addStretch()
        layout.addLayout(png_button_layout)

        # Row 9: Grayscale penalization strength slider
        eta_layout = QHBoxLayout()
        eta_layout.addWidget(QLabel("Penalization Strength:"))
        self.eta_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.eta_max_slider.setMinimum(1)
        self.eta_max_slider.setMaximum(500)  # Will be divided by 100 to get 0.01 to 5.0
        self.eta_max_slider.setValue(50)  # Default eta_max = 0.5
        self.eta_max_slider.setMaximumWidth(200)
        self.eta_max_slider.setToolTip("Maximum penalization strength for grayscale obstacles (eta_max)")
        self.eta_max_label = QLabel("0.5")
        self.eta_max_label.setFixedWidth(40)
        eta_layout.addWidget(self.eta_max_slider)
        eta_layout.addWidget(self.eta_max_label)
        eta_layout.addStretch()
        layout.addLayout(eta_layout)

        # Row 10: Show mask outline checkbox
        outline_layout = QHBoxLayout()
        from PyQt6.QtWidgets import QCheckBox
        self.show_outline_checkbox = QCheckBox("Show Mask Outline")
        self.show_outline_checkbox.setChecked(True)
        self.show_outline_checkbox.setToolTip("Toggle visibility of mask overlay outlines (grey on velocity, black on vorticity)")
        outline_layout.addWidget(self.show_outline_checkbox)
        outline_layout.addStretch()
        layout.addLayout(outline_layout)

        # Connect slider signals
        self.x_position_slider.valueChanged.connect(self._on_x_position_changed)
        self.y_position_slider.valueChanged.connect(self._on_y_position_changed)
        self.load_png_mask_btn.clicked.connect(self._on_load_png_mask_clicked)
        self.eta_max_slider.valueChanged.connect(self._on_eta_max_changed)
        self.show_outline_checkbox.stateChanged.connect(self._on_outline_visibility_changed)

        self.setLayout(layout)

        # If NACA module missing, disable controls
        if not self._check_naca_availability():
            self.naca_combo = None
            self.chord_spinbox = None
            self.angle_spinbox = None
            self.angle_slider = None
            self.apply_naca_btn = None

    def _on_obstacle_radio_changed(self, button):
        """Handle obstacle type radio button selection changes."""
        if button == self.cylinder_radio:
            obstacle_type = 'cylinder'
        elif button == self.naca_radio:
            obstacle_type = 'naca_airfoil'
        elif button == self.cow_radio:
            obstacle_type = 'cow'
        elif button == self.cylinder_array_radio:
            obstacle_type = 'three_cylinder_array'
        elif button == self.solid_wall_radio:
            obstacle_type = 'solid_wall'
        elif button == self.urban_map_radio:
            obstacle_type = 'urban_map'
        elif button == self.tesla_valve_radio:
            obstacle_type = 'tesla_valve'
        else:
            return

        # Update store state with current slider position BEFORE changing obstacle type
        # This ensures the new obstacle type gets the correct position from the start
        if hasattr(self, 'x_position_slider') and hasattr(self, 'y_position_slider'):
            # Get viewer to access grid dimensions
            viewer = None
            if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
                if hasattr(self.parent_viewer, 'parent_viewer'):
                    viewer = self.parent_viewer.parent_viewer
                elif hasattr(self.parent_viewer, 'solver'):
                    viewer = self.parent_viewer
            
            if viewer is not None and hasattr(viewer, 'solver'):
                grid_lx = viewer.solver.grid.lx
                grid_ly = viewer.solver.grid.ly
                x_position = (self.x_position_slider.value() / 100.0) * grid_lx
                y_position = (self.y_position_slider.value() / 100.0) * grid_ly
                # Dispatch position update for the NEW obstacle type
                store.dispatch(set_obstacle_position(obstacle_type, x_position, y_position))

        # Dispatch Redux action to update store state
        store.dispatch(set_obstacle_type(obstacle_type))

        # Update UI controls visibility
        if hasattr(self, 'naca_widget'):
            self.naca_widget.setVisible(obstacle_type == 'naca_airfoil')
        if hasattr(self, 'cylinder_widget'):
            self.cylinder_widget.setVisible(obstacle_type == 'cylinder')
        if hasattr(self, 'cylinder_array_widget'):
            self.cylinder_array_widget.setVisible(obstacle_type == 'three_cylinder_array')
        if hasattr(self, 'solid_wall_widget'):
            self.solid_wall_widget.setVisible(obstacle_type == 'solid_wall')
        if hasattr(self, 'urban_map_widget'):
            self.urban_map_widget.setVisible(obstacle_type == 'urban_map')
        if hasattr(self, 'tesla_valve_widget'):
            self.tesla_valve_widget.setVisible(obstacle_type == 'tesla_valve')

        # Notify parent viewer if available (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'on_obstacle_type_selected'):
                self.parent_viewer.on_obstacle_type_selected(obstacle_type)

    def _on_cylinder_radius_changed(self, value):
        """Handle cylinder radius spinbox changes for live preview."""
        # Get viewer to access solver
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer
        
        if viewer is not None and hasattr(viewer, 'solver'):
            # Update cylinder radius in solver
            import jax.numpy as jnp
            viewer.solver.geom.radius = jnp.array(value)
            
            # Recompute mask for preview
            viewer.solver.mask = viewer.solver._compute_mask()
            
            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)

    def _on_x_position_changed(self, value):
        """Handle x-position slider changes."""
        self.x_position_label.setText(f"{value}%")

        # Dispatch Redux action for live preview update
        # Access solver through parent_viewer.parent_viewer (ControlPanel -> Main Viewer)
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer

        if viewer is not None and hasattr(viewer, 'solver'):
            grid_lx = viewer.solver.grid.lx
            x_position = (value / 100.0) * grid_lx
            obstacle_type = getattr(viewer.solver.sim_params, 'obstacle_type', 'cylinder')

            # Special handling for solid wall - update solid_wall_x parameter
            if obstacle_type == 'solid_wall':
                viewer.solver.sim_params.solid_wall_x = value / 100.0
                # Recompute mask for preview
                viewer.solver.mask = viewer.solver._compute_mask()
                # Update obstacle outline preview
                if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                    viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            else:
                # Get current y position
                if hasattr(self, 'y_position_slider'):
                    y_value = self.y_position_slider.value()
                    grid_ly = viewer.solver.grid.ly
                    y_position = (y_value / 100.0) * grid_ly
                else:
                    y_position = None

                print(f"[SLIDER] Dispatching action: x={x_position:.2f}, obstacle_type={obstacle_type}")
                # Dispatch Redux action
                store.dispatch(set_obstacle_position(obstacle_type, x_position, y_position))

        # Notify parent viewer (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_x_position_change'):
                self.parent_viewer.apply_x_position_change(value)

    def _on_y_position_changed(self, value):
        """Handle y-position slider changes."""
        self.y_position_label.setText(f"{value}%")

        # Dispatch Redux action for live preview update
        # Access solver through parent_viewer.parent_viewer (ControlPanel -> Main Viewer)
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer

        if viewer is not None and hasattr(viewer, 'solver'):
            grid_ly = viewer.solver.grid.ly
            y_position = (value / 100.0) * grid_ly
            obstacle_type = getattr(viewer.solver.sim_params, 'obstacle_type', 'cylinder')

            # Get current x position
            if hasattr(self, 'x_position_slider'):
                x_value = self.x_position_slider.value()
                grid_lx = viewer.solver.grid.lx
                x_position = (x_value / 100.0) * grid_lx
            else:
                x_position = None

            print(f"[SLIDER] Dispatching action: y={y_position:.2f}, obstacle_type={obstacle_type}")
            # Dispatch Redux action
            store.dispatch(set_obstacle_position(obstacle_type, x_position, y_position))

        # Notify parent viewer (backward compatibility)
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'apply_y_position_change'):
                self.parent_viewer.apply_y_position_change(value)

    def _on_outline_visibility_changed(self, state):
        """Handle outline visibility checkbox changes."""
        # Check if LBM solver is currently selected - mask overlay should not show for LBM
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer

        # Prevent showing mask overlay when LBM solver is selected
        if viewer is not None and hasattr(viewer, 'solver'):
            solver_type = getattr(viewer.solver.sim_params, 'solver_type', 'navier_stokes')
            if solver_type == 'lattice_boltzmann' and state == 2:  # 2 = checked
                # Force uncheck the checkbox
                self.show_outline_checkbox.blockSignals(True)
                self.show_outline_checkbox.setChecked(False)
                self.show_outline_checkbox.blockSignals(False)
                print("Mask overlay not available for LBM solver")
                return

        if viewer is not None and hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
            # Set the outline visibility flag in the obstacle renderer
            is_visible = (state == 2)  # 2 = checked, 0 = unchecked
            viewer.obstacle_renderer.show_outlines = is_visible

            # Directly set visibility on outline items with error handling for deleted objects
            renderer = viewer.obstacle_renderer
            try:
                if renderer.vel_outline is not None and hasattr(renderer.vel_outline, 'setVisible'):
                    renderer.vel_outline.setVisible(is_visible)
            except RuntimeError:
                pass  # Object was deleted, skip
            try:
                if renderer.vort_outline is not None and hasattr(renderer.vort_outline, 'setVisible'):
                    renderer.vort_outline.setVisible(is_visible)
            except RuntimeError:
                pass  # Object was deleted, skip
            try:
                if renderer.div_outline is not None and hasattr(renderer.div_outline, 'setVisible'):
                    renderer.div_outline.setVisible(is_visible)
            except RuntimeError:
                pass  # Object was deleted, skip
            try:
                if renderer.scalar_outline is not None and hasattr(renderer.scalar_outline, 'setVisible'):
                    renderer.scalar_outline.setVisible(is_visible)
            except RuntimeError:
                pass  # Object was deleted, skip
            try:
                if renderer.pressure_outline is not None and hasattr(renderer.pressure_outline, 'setVisible'):
                    renderer.pressure_outline.setVisible(is_visible)
            except RuntimeError:
                pass  # Object was deleted, skip

    def _on_naca_hover(self, index):
        """Show airfoil preview when selection changes"""
        if not self._check_naca_availability():
            return

        designation = self.naca_combo.currentText()
        if not designation or designation == "NACA 0012":
            return

        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS, parse_naca_4digit, parse_naca_5digit
            if designation not in NACA_AIRFOILS:
                return

            digits = ''.join(filter(str.isdigit, designation))
            tooltip_text = f"<b>{designation}</b><br><br>"

            if len(digits) == 4:
                m, p, t = parse_naca_4digit(designation)
                tooltip_text += f"Type: 4-digit series<br>"
                tooltip_text += f"Max camber: {m*100:.1f}%<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"
            elif len(digits) == 5:
                cl, p, m, t = parse_naca_5digit(designation)
                tooltip_text += f"Type: 5-digit series<br>"
                tooltip_text += f"Design lift coeff: {cl:.2f}<br>"
                tooltip_text += f"Camber position: {p*100:.0f}% chord<br>"
                tooltip_text += f"Max camber: {m*10:.1f}%<br>"
                tooltip_text += f"Max thickness: {t*100:.1f}% chord"

            self.naca_combo.setToolTip(tooltip_text)
        except Exception as e:
            pass

    def _on_solid_wall_y_bottom_changed(self, value):
        """Handle solid wall y-bottom slider changes."""
        self.solid_wall_y_bottom_label.setText(f"{value}%")

        # Get viewer to access solver
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer

        if viewer is not None and hasattr(viewer, 'solver'):
            # Update sim_params
            viewer.solver.sim_params.solid_wall_y_bottom = value / 100.0

            # Recompute mask for preview
            viewer.solver.mask = viewer.solver._compute_mask()

            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)

    def _on_solid_wall_y_top_changed(self, value):
        """Handle solid wall y-top slider changes."""
        self.solid_wall_y_top_label.setText(f"{value}%")

        # Get viewer to access solver
        viewer = None
        if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'parent_viewer'):
                viewer = self.parent_viewer.parent_viewer
            elif hasattr(self.parent_viewer, 'solver'):
                viewer = self.parent_viewer

        if viewer is not None and hasattr(viewer, 'solver'):
            # Update sim_params
            viewer.solver.sim_params.solid_wall_y_top = value / 100.0

            # Recompute mask for preview
            viewer.solver.mask = viewer.solver._compute_mask()

            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)

    def _check_naca_availability(self):
        """Check if NACA airfoils are available"""
        try:
            from obstacles.naca_airfoils import NACA_AIRFOILS
            return True
        except ImportError:
            return False

    def _generate_mockup_urban_map(self, viewer):
        """Generate a mockup urban map SDF with simple building rectangles"""
        try:
            import numpy as np
            
            # Get grid dimensions
            X = np.array(viewer.solver.grid.X)
            Y = np.array(viewer.solver.grid.Y)
            nx, ny = X.shape
            
            print(f"DEBUG mockup: Grid shape is (nx, ny) = ({nx}, {ny})")
            
            # Start with all fluid (positive SDF)
            sdf = np.ones((nx, ny), dtype=np.float32) * 10.0
            
            # Define some simple building rectangles as (x_min, x_max, y_min, y_max)
            # Using normalized coordinates relative to grid bounds
            lx = viewer.solver.grid.lx
            ly = viewer.solver.grid.ly
            
            buildings = [
                (0.2 * lx, 0.35 * lx, 0.2 * ly, 0.35 * ly),  # Building 1
                (0.4 * lx, 0.5 * lx, 0.15 * ly, 0.25 * ly),  # Building 2
                (0.6 * lx, 0.7 * lx, 0.25 * ly, 0.35 * ly),  # Building 3
                (0.3 * lx, 0.4 * lx, 0.4 * ly, 0.5 * ly),  # Building 4
                (0.5 * lx, 0.6 * lx, 0.5 * ly, 0.6 * ly),  # Building 5
            ]
            
            # Compute SDF for each building (negative inside, positive outside)
            for bx_min, bx_max, by_min, by_max in buildings:
                # Distance to rectangle edges
                dx = np.maximum(np.maximum(bx_min - X, X - bx_max), 0)
                dy = np.maximum(np.maximum(by_min - Y, Y - by_max), 0)
                
                # Distance to rectangle (outside distance)
                outside_dist = np.sqrt(dx**2 + dy**2)
                
                # Inside distance (negative): max of distances to each edge (negative)
                inside_dist = -np.minimum(np.minimum(bx_min - X, X - bx_max), 
                                          np.minimum(by_min - Y, Y - by_max))
                
                # Signed distance: negative inside, positive outside
                building_sdf = np.where((X >= bx_min) & (X <= bx_max) & (Y >= by_min) & (Y <= by_max),
                                       inside_dist, outside_dist)
                
                # Take minimum with current SDF (union of buildings)
                sdf = np.minimum(sdf, building_sdf)
            
            # Store the mockup SDF
            viewer.solver.sim_params.sdf_field = sdf
            
            # Debug: Check if any negative SDF values exist (inside buildings)
            negative_count = np.sum(sdf < 0)
            total_cells = sdf.size
            print(f"DEBUG mockup: {negative_count}/{total_cells} cells have negative SDF (inside buildings)")
            
            # Update Redux store to keep it in sync
            from viewer.state import store, set_obstacle_type
            store.dispatch(set_obstacle_type('urban_map'))
            
            self.urban_map_status_label.setText("Loaded mockup urban map (5 buildings)")
            self.urban_map_status_label.setStyleSheet("color: green;")
            print("Generated mockup urban map SDF")
            
        except Exception as e:
            print(f"Error generating mockup urban map: {e}")
            import traceback
            traceback.print_exc()

    def set_chord_range_for_domain(self, max_chord: float):
        """Update chord spinbox range based on domain size"""
        if hasattr(self, 'chord_spinbox') and self.chord_spinbox is not None:
            self.chord_spinbox.setRange(0.1, max_chord)

    def show_naca_controls(self, show: bool) -> None:
        """Show/hide NACA controls based on obstacle selection"""
        if hasattr(self, 'naca_widget'):
            self.naca_widget.setVisible(show)

    def _on_download_clicked(self):
        """Download building data from Overture Maps using CLI"""
        try:
            # Check if overturemaps is installed
            try:
                subprocess.run(["overturemaps", "--help"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.urban_map_status_label.setText("Error: overturemaps not installed. Run: pip install overturemaps")
                self.urban_map_status_label.setStyleSheet("color: red;")
                return
            
            # Show bbox dialog with templates
            dialog = BBoxDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            bbox = dialog.get_bbox()
            if bbox is None:
                self.urban_map_status_label.setText("Error: Invalid bbox format. Use: west,south,east,north")
                self.urban_map_status_label.setStyleSheet("color: red;")
                return
            
            west, south, east, north = bbox
        except Exception as e:
            self.urban_map_status_label.setText(f"Error in download dialog: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            print(f"Download dialog error: {e}")
            return
        
        # Ask for output filename
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save GeoJSON File",
            "buildings.geojson",
            "GeoJSON Files (*.geojson);;All Files (*)"
        )
        
        if not output_file:
            return
        
        # Run overturemaps download command
        self.urban_map_status_label.setText("Downloading building data...")
        self.urban_map_status_label.setStyleSheet("color: blue;")
        
        try:
            # Use Python to run overturemaps with UTF-8 encoding
            python_script = f'''
import sys
import subprocess
import os

# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Set stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Run overturemaps
cmd = ["overturemaps", "download", "--bbox={west},{south},{east},{north}", "--type=building", "-f", "geojson", "-o", r"{output_file}"]
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)

print("SUCCESS")
'''
            
            process = subprocess.run(
                ["python", "-c", python_script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if process.returncode == 0:
                self.urban_map_status_label.setText(f"Downloaded to {output_file}")
                self.urban_map_status_label.setStyleSheet("color: green;")
                # Optionally auto-load
                self._load_geojson_file(output_file)
            else:
                error_msg = process.stderr or process.stdout or "Unknown error"
                self.urban_map_status_label.setText(f"Download failed: {error_msg}")
                self.urban_map_status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.urban_map_status_label.setText(f"Error: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")

    def _on_select_area_clicked(self):
        """Open browser to select area on Overture Maps"""
        try:
            # Overture Maps explorer URL
            url = "https://overturemaps.org/explorer/"
            webbrowser.open(url)
            self.urban_map_status_label.setText("Browser opened. Select area and download GeoJSON.")
        except Exception as e:
            self.urban_map_status_label.setText(f"Error opening browser: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            print(f"Browser open error: {e}")

    def _on_load_geojson_clicked(self):
        """Load GeoJSON file and generate SDF"""
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load GeoJSON File",
                "",
                "GeoJSON Files (*.geojson *.json);;All Files (*)"
            )
            
            if not file_path:
                return
            
            self._load_geojson_file(file_path)
        except Exception as e:
            self.urban_map_status_label.setText(f"Error loading file: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            print(f"File load error: {e}")

    def _load_geojson_file(self, file_path: str):
        """Load GeoJSON file at given path and generate SDF"""
        try:
            print("DEBUG: Starting _load_geojson_file")
            
            # Get viewer to access solver and grid
            viewer = None
            if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
                print("DEBUG: parent_viewer exists")
                if hasattr(self.parent_viewer, 'parent_viewer'):
                    viewer = self.parent_viewer.parent_viewer
                    print("DEBUG: Found parent_viewer.parent_viewer")
                elif hasattr(self.parent_viewer, 'solver'):
                    viewer = self.parent_viewer
                    print("DEBUG: Found parent_viewer with solver")
            
            if viewer is None:
                print("DEBUG: viewer is None")
                self.urban_map_status_label.setText("Error: Cannot access solver.")
                self.urban_map_status_label.setStyleSheet("color: red;")
                return
            
            if not hasattr(viewer, 'solver'):
                print("DEBUG: viewer has no solver attribute")
                self.urban_map_status_label.setText("Error: Cannot access solver.")
                self.urban_map_status_label.setStyleSheet("color: red;")
                return
            
            print("DEBUG: Solver access confirmed")
        except Exception as e:
            print(f"DEBUG: Error in viewer setup: {e}")
            self.urban_map_status_label.setText(f"Error accessing solver: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            return
        
        try:
            # Use pure numpy version to avoid JAX DLL issues
            from obstacles.sdf_generator_numpy import load_building_polygons_numpy, polygons_to_sdf_numpy
            
            # Get grid bounds
            grid_lx = viewer.solver.grid.lx
            grid_ly = viewer.solver.grid.ly
            X = viewer.solver.grid.X
            Y = viewer.solver.grid.Y
            
            # Define bbox (use full grid)
            bbox = (0, grid_lx, 0, grid_ly)
            
            self.urban_map_status_label.setText("Loading building polygons...")
            self.urban_map_status_label.setStyleSheet("color: blue;")
            
            # Load building polygons
            polygons = load_building_polygons_numpy(file_path, bbox, max_buildings=self.max_buildings_spinbox.value(), spinbox_widget=self.max_buildings_spinbox)
            
            if not polygons:
                self.urban_map_status_label.setText("No buildings found in file.")
                self.urban_map_status_label.setStyleSheet("color: orange;")
                return
            
            self.urban_map_status_label.setText(f"Loaded {len(polygons)} buildings. Computing SDF...")
            
            # Convert numpy arrays to regular numpy (not JAX) for SDF computation
            X_np = np.array(X)
            Y_np = np.array(Y)
            
            # Check grid size - large grids can cause memory issues with distance_transform_edt
            nx, ny = X_np.shape
            max_grid_size = 1000 * 1000  # 1 million cells
            if nx * ny > max_grid_size:
                self.urban_map_status_label.setText(f"Error: Grid too large ({nx}x{ny}={nx*ny} cells). Max recommended: 1000x1000.")
                self.urban_map_status_label.setStyleSheet("color: red;")
                print(f"Grid size {nx}x{ny} exceeds maximum {max_grid_size}")
                return
            
            self.urban_map_status_label.setText(f"Computing SDF for {nx}x{ny} grid...")
            
            # Compute SDF using pure numpy
            try:
                sdf_result = polygons_to_sdf_numpy(polygons, X_np, Y_np)
                # Handle tuple return (combined_sdf, individual_sdfs)
                if isinstance(sdf_result, tuple):
                    sdf = sdf_result[0]  # Extract combined SDF (numpy array)
                    individual_sdfs = sdf_result[1] if len(sdf_result) > 1 else []
                else:
                    sdf = sdf_result
                    individual_sdfs = []
            except Exception as e:
                self.urban_map_status_label.setText(f"SDF computation failed: {str(e)}")
                self.urban_map_status_label.setStyleSheet("color: red;")
                import traceback
                traceback.print_exc()
                return
            
            # Store numpy array directly in sim_params
            viewer.solver.sim_params.sdf_field = sdf
            
            # Store individual building SDFs for separate contour extraction
            if individual_sdfs and len(individual_sdfs) > 0:
                viewer.solver.sim_params.individual_sdfs = individual_sdfs
                print(f"DEBUG: Stored {len(individual_sdfs)} individual building SDFs in sim_params")
                
                # Pre-extract contours once during loading to avoid slow matplotlib on every frame
                try:
                    print(f"DEBUG: Pre-extracting {len(individual_sdfs)} building contours...")
                    import matplotlib.pyplot as plt
                    cached_polygons = []
                    
                    # Store original building polygons for direct outline rendering
                    if hasattr(viewer.solver.sim_params, 'original_building_polygons'):
                        original_polygons = viewer.solver.sim_params.original_building_polygons
                        print(f"DEBUG: Found {len(original_polygons)} original building polygons")
                        cached_polygons = original_polygons
                    else:
                        # Fallback to matplotlib contour extraction
                        for i, building_sdf in enumerate(individual_sdfs):
                            if building_sdf is None or not isinstance(building_sdf, np.ndarray):
                                continue
                            # Use aspect ratio matching the plot to prevent distortion
                            aspect_ratio = viewer.solver.grid.lx / viewer.solver.grid.ly
                            fig_width, fig_height = 4, 4 / aspect_ratio
                            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                            ax.set_aspect('equal')  # Force equal aspect ratio
                            
                            nx, ny = building_sdf.shape
                            # Use cell center coordinates like solver grid
                            dx = viewer.solver.grid.lx / nx
                            dy = viewer.solver.grid.ly / ny
                            x = np.linspace(dx/2, viewer.solver.grid.lx - dx/2, nx)
                            y = np.linspace(dy/2, viewer.solver.grid.ly - dy/2, ny)
                            print(f"DEBUG: Matplotlib grid (cell centers): x=[{x[0]:.1f}, {x[-1]:.1f}], y=[{y[0]:.1f}, {y[-1]:.1f}]")
                            print(f"DEBUG: Solver grid: dx={dx:.3f}, dy={dy:.3f}")
                            contours = ax.contour(x, y, building_sdf.T, levels=[0])
                            
                            # Handle different matplotlib versions
                            if hasattr(contours, 'collections'):
                                for collection in contours.collections:
                                    for path in collection.get_paths():
                                        verts = [(float(v[0]), float(v[1])) for v in path.vertices]
                                        if len(verts) >= 3:
                                            # Debug: Compare contour coordinates with original building coordinates
                                            x_coords = [v[0] for v in verts]
                                            y_coords = [v[1] for v in verts]
                                            print(f"DEBUG: Contour {len(cached_polygons)}: bounds x=[{min(x_coords):.1f}, {max(x_coords):.1f}], y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
                                            print(f"DEBUG: First 3 vertices: {verts[:3]}")
                                            cached_polygons.append(verts)
                            else:
                                for path in contours.get_paths():
                                    verts = [(float(v[0]), float(v[1])) for v in path.vertices]
                                    if len(verts) >= 3:
                                        cached_polygons.append(verts)
                            
                            plt.close(fig)
                    
                    viewer.solver.sim_params.urban_map_polygons = cached_polygons
                    print(f"DEBUG: Cached {len(cached_polygons)} building polygon contours")
                
                except Exception as e:
                    print(f"Error extracting building contours: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.urban_map_status_label.setText("Switching to urban map obstacle type...")
            
            # Switch obstacle type to urban_map (don't dispatch to store to avoid redundant subscription)
            viewer.solver.sim_params.obstacle_type = 'urban_map'
            # Also select the radio button for visual feedback
            self.urban_map_radio.setChecked(True)
            
            self.urban_map_status_label.setText("Computing mask from SDF...")
            
            # Recompute mask
            try:
                viewer.solver.mask = viewer.solver._compute_mask()
            except Exception as e:
                self.urban_map_status_label.setText(f"Mask computation failed: {str(e)}")
                self.urban_map_status_label.setStyleSheet("color: red;")
                import traceback
                traceback.print_exc()
                return
            
            # Update obstacle outline preview
            try:
                if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                    viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            except Exception as e:
                print(f"Warning: Failed to update obstacle outlines: {e}")
            
            self.urban_map_status_label.setText(f"Loaded {len(polygons)} buildings successfully.")
            self.urban_map_status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            self.urban_map_status_label.setText(f"Error: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            import traceback
            traceback.print_exc()

    def _on_generate_urban_clicked(self):
        """Generate Manhattan-style urban SDF using the new generator"""
        try:
            # Get viewer to access solver
            viewer = None
            if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
                if hasattr(self.parent_viewer, 'parent_viewer'):
                    viewer = self.parent_viewer.parent_viewer
                elif hasattr(self.parent_viewer, 'solver'):
                    viewer = self.parent_viewer
            
            if viewer is None:
                self.urban_map_status_label.setText("Error: No viewer available")
                self.urban_map_status_label.setStyleSheet("color: red;")
                return
            
            # Import the urban SDF generator
            from obstacles.urban_sdf_generator import generate_urban_sdf
            
            # Get grid dimensions from solver
            nx = viewer.solver.grid.nx
            ny = viewer.solver.grid.ny
            
            self.urban_map_status_label.setText(f"Generating urban SDF for {nx}x{ny} grid...")
            self.urban_map_status_label.setStyleSheet("color: blue;")
            
            # Generate the urban SDF
            sdf_field = generate_urban_sdf(nx=nx, ny=ny)
            print(f"DEBUG: Generated new urban SDF: shape={sdf_field.shape}, min={sdf_field.min():.3f}, max={sdf_field.max():.3f}, negative_cells={(sdf_field < 0).sum()}")
            
            # Store SDF in simulation parameters
            viewer.solver.sim_params.sdf_field = sdf_field
            viewer.solver.sim_params.individual_sdfs = []  # No individual SDFs for generated map
            
            # Clear any cached polygons since we're using a different SDF
            if hasattr(viewer.solver.sim_params, 'urban_map_polygons'):
                delattr(viewer.solver.sim_params, 'urban_map_polygons')
            
            # Set obstacle type and recompute mask
            viewer.solver.sim_params.obstacle_type = 'urban_map'
            viewer.solver.mask = viewer.solver._compute_mask()
            
            # Update Redux store to keep it in sync
            from viewer.state import store, set_obstacle_type
            store.dispatch(set_obstacle_type('urban_map'))
            
            # Update obstacle outline preview
            try:
                if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                    viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            except Exception as e:
                print(f"Warning: Failed to update obstacle outlines: {e}")
            
            self.urban_map_status_label.setText(f"Generated Manhattan-style urban SDF ({nx}x{ny})")
            self.urban_map_status_label.setStyleSheet("color: green;")
            print(f"Generated urban SDF: shape={sdf_field.shape}, range=[{sdf_field.min():.3f}, {sdf_field.max():.3f}]")
            
        except Exception as e:
            self.urban_map_status_label.setText(f"Error generating urban SDF: {str(e)}")
            self.urban_map_status_label.setStyleSheet("color: red;")
            import traceback
            traceback.print_exc()

    def _on_tesla_valve_apply_clicked(self):
        """Handle Tesla valve apply button click."""
        try:
            # Get viewer to access solver
            viewer = None
            if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
                if hasattr(self.parent_viewer, 'parent_viewer'):
                    viewer = self.parent_viewer.parent_viewer
                elif hasattr(self.parent_viewer, 'solver'):
                    viewer = self.parent_viewer
            
            if viewer is None or not hasattr(viewer, 'solver'):
                print("Error: Cannot access solver for Tesla valve")
                return
            
            # Import jnp for angle conversion
            import jax.numpy as jnp
            
            # Store all Tesla valve parameters
            viewer.solver.sim_params.tesla_valve_stages = self.tesla_valve_stages.value()
            viewer.solver.sim_params.tesla_valve_stage_length = self.tesla_valve_stage_len.value()
            viewer.solver.sim_params.tesla_valve_main_width = self.tesla_valve_main_width.value()
            viewer.solver.sim_params.tesla_valve_branch_width = self.tesla_valve_branch_width.value()
            viewer.solver.sim_params.tesla_valve_diagonal_length = self.tesla_valve_diagonal_length.value()
            viewer.solver.sim_params.tesla_valve_branch_angle = self.tesla_valve_branch_angle.value() * jnp.pi / 180.0
            viewer.solver.sim_params.tesla_valve_forward = self.tesla_forward_checkbox.value() == 1
            
            # Set position from sliders
            if hasattr(self, 'x_position_slider') and hasattr(self, 'y_position_slider'):
                grid_lx = viewer.solver.grid.lx
                grid_ly = viewer.solver.grid.ly
                viewer.solver.sim_params.tesla_valve_x = (self.x_position_slider.value() / 100.0) * grid_lx
                viewer.solver.sim_params.tesla_valve_y = (self.y_position_slider.value() / 100.0) * grid_ly
            
            # Recompute mask with Tesla valve
            viewer.solver.mask = viewer.solver._compute_mask()
            
            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            
            direction_text = "forward" if viewer.solver.sim_params.tesla_valve_forward else "backward"
            stages = viewer.solver.sim_params.tesla_valve_stages
            print(f"Applied {direction_text} Tesla valve with {stages} stages")
            
        except Exception as e:
            print(f"Error applying Tesla valve: {e}")
            import traceback
            traceback.print_exc()

    def _on_tesla_direction_changed(self, value):
        """Handle Tesla valve direction checkbox changes."""
        # This provides immediate feedback when checkbox is toggled
        direction_text = "forward" if value == 1 else "backward"
        print(f"Tesla valve direction changed to: {direction_text}")

    def _resample_png_mask(self, viewer):
        """Re-sample PNG mask to current grid size"""
        try:
            # Check if original PNG data exists
            png_original = getattr(viewer.solver.sim_params, 'png_original_image', None)
            if png_original is None:
                return False
            
            # Get current grid dimensions
            nx = viewer.solver.grid.nx
            ny = viewer.solver.grid.ny
            
            # Start with original image
            img_array = png_original.copy()
            
            # Resize image to match grid dimensions if needed
            if img_array.shape[0] != ny or img_array.shape[1] != nx:
                from scipy.ndimage import zoom
                # Calculate zoom factors - ensure correct orientation
                zoom_y = ny / img_array.shape[0]
                zoom_x = nx / img_array.shape[1]
                img_array = zoom(img_array, (zoom_y, zoom_x), order=1)
            
            # Ensure final shape matches grid exactly
            if img_array.shape[0] != ny or img_array.shape[1] != nx:
                print(f"Warning: Resampling shape mismatch after zoom: {img_array.shape} vs ({ny}, {nx})")
                return False
            
            # Transpose to match grid indexing: (ny, nx) -> (nx, ny)
            img_array = img_array.T
            
            # Final shape check
            if img_array.shape != (nx, ny):
                print(f"Warning: Final shape mismatch: {img_array.shape} vs ({nx}, {ny})")
                return False
            
            # Normalize to 0-1 range
            img_array = img_array / 255.0
            
            # Mirror horizontally (flip left-right)
            img_array = np.fliplr(img_array)
            
            # Create binary mask: white pixels (close to 1.0) = solid (0), non-white = fluid (1)
            # Use threshold of 0.9 to identify white pixels
            threshold = 0.9
            binary_mask = np.where(img_array > threshold, 0.0, 1.0)  # Inverted: white=0 (solid), non-white=1 (fluid)
            
            # Convert binary mask to SDF using distance transform
            from scipy.ndimage import distance_transform_edt
            distance_outside = distance_transform_edt(1.0 - binary_mask)
            distance_inside = distance_transform_edt(binary_mask)
            sdf = distance_outside - distance_inside
            
            # Scale SDF to reasonable values (multiply by grid cell size)
            dx = viewer.solver.grid.lx / nx
            dy = viewer.solver.grid.ly / ny
            avg_cell_size = (dx + dy) / 2.0
            sdf = sdf * avg_cell_size
            
            # Store both SDF field and custom mask in simulation parameters
            viewer.solver.sim_params.sdf_field = sdf
            viewer.solver.sim_params.custom_mask = (sdf > 0).astype(np.float32)  # Binary mask (1=fluid, 0=solid)
            
            # Recompute mask
            viewer.solver.mask = viewer.solver._compute_mask()
            
            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            
            print(f"Re-sampled PNG mask to new grid size: shape={sdf.shape}, range=[{sdf.min():.3f}, {sdf.max():.3f}]")
            return True
            
        except Exception as e:
            print(f"Error re-sampling PNG mask: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_load_png_mask_clicked(self):
        """Load PNG file and convert to SDF mask"""
        try:
            # Open file dialog for PNG files
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load PNG Mask",
                "",
                "PNG Files (*.png);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Get viewer to access solver
            viewer = None
            if hasattr(self, 'parent_viewer') and self.parent_viewer is not None:
                if hasattr(self.parent_viewer, 'parent_viewer'):
                    viewer = self.parent_viewer.parent_viewer
                elif hasattr(self.parent_viewer, 'solver'):
                    viewer = self.parent_viewer
            
            if viewer is None or not hasattr(viewer, 'solver'):
                print("Error: Cannot access solver for PNG mask loading")
                return
            
            # Load PNG image using PIL
            from PIL import Image
            img = Image.open(file_path)
            
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # Convert to numpy array
            img_array = np.array(img_gray, dtype=np.float32)
            
            # Store original PNG data for re-sampling when grid size changes
            viewer.solver.sim_params.png_original_image = img_array.copy()
            
            # Get grid dimensions
            nx = viewer.solver.grid.nx
            ny = viewer.solver.grid.ny
            
            # Resize image to match grid dimensions if needed
            # Note: img_array has shape (height, width) = (rows, cols)
            # Grid X, Y have shape (nx, ny) with indexing='ij'
            # So we need img_array shape (ny, nx) -> transpose to (nx, ny)
            if img_array.shape[0] != ny or img_array.shape[1] != nx:
                from scipy.ndimage import zoom
                # Calculate zoom factors - ensure correct orientation
                zoom_y = ny / img_array.shape[0]
                zoom_x = nx / img_array.shape[1]
                img_array = zoom(img_array, (zoom_y, zoom_x), order=1)
            
            # Ensure final shape matches grid exactly
            assert img_array.shape[0] == ny, f"Height mismatch: {img_array.shape[0]} != {ny}"
            assert img_array.shape[1] == nx, f"Width mismatch: {img_array.shape[1]} != {nx}"
            
            # Transpose to match grid indexing: (ny, nx) -> (nx, ny)
            img_array = img_array.T
            
            # Final shape check
            assert img_array.shape == (nx, ny), f"Final shape mismatch: {img_array.shape} != ({nx}, {ny})"
            
            # Normalize to 0-1 range
            img_array = img_array / 255.0
            
            # Mirror horizontally (flip left-right)
            img_array = np.fliplr(img_array)
            
            # Store grayscale field for Brinkman penalization
            # White (1.0) = fluid (low penalization), Black (0.0) = solid (high penalization)
            # Invert: grayscale_penalization = 1.0 - img_array
            # So white (1.0) -> 0.0 penalization, black (0.0) -> 1.0 penalization
            grayscale_penalization = 1.0 - img_array
            viewer.solver.sim_params.grayscale_penalization = grayscale_penalization.copy()
            
            # Use grayscale values directly as continuous mask (no binary threshold)
            # White (1.0) = 1.0 (fluid), Black (0.0) = 0.0 (solid), Grey = intermediate
            viewer.solver.sim_params.custom_mask = img_array.copy()  # Continuous mask (0-1)
            viewer.solver.sim_params.obstacle_type = 'custom'
            
            # Don't create SDF - we'll use the continuous grayscale mask directly
            # This allows grey regions to have partial flow
            
            # Recompute mask
            viewer.solver.mask = viewer.solver._compute_mask()
            
            # Update obstacle outline preview
            if hasattr(viewer, 'obstacle_renderer') and viewer.obstacle_renderer:
                viewer.obstacle_renderer.update_obstacle_outlines(viewer.solver, force_update=True)
            
            print(f"Loaded PNG mask from {file_path}: shape={img_array.shape}, grayscale range=[{img_array.min():.3f}, {img_array.max():.3f}]")
            
        except Exception as e:
            print(f"Error loading PNG mask: {e}")
            import traceback
            traceback.print_exc()

    def _on_eta_max_changed(self, value: int):
        """Handle eta_max slider change"""
        eta_max = float(value) / 100.0  # Convert slider 1-500 to eta_max 0.01-5.0
        self.eta_max_label.setText(f"{eta_max:.2f}")
        
        viewer = self.parent().parent().parent()  # Navigate to BaselineViewerRefactored
        
        # Update NS solver eta_max
        if hasattr(viewer, 'solver') and hasattr(viewer.solver, 'sim_params'):
            viewer.solver.sim_params.eta_max = eta_max
            # Clear JIT cache to recompile with new eta_max
            if hasattr(viewer.solver, '_step_jit'):
                viewer.solver._step_jit = None
        
        # Update LBM solver eta_max
        if hasattr(viewer, 'lbm_solver') and hasattr(viewer.lbm_solver, 'lbm_params'):
            viewer.lbm_solver.lbm_params.eta_max = eta_max
            # Clear JIT cache to recompile with new eta_max
            if hasattr(viewer.lbm_solver, '_jit_cache'):
                viewer.lbm_solver._jit_cache = {}
            if hasattr(viewer.lbm_solver, 'get_step_jit'):
                viewer.lbm_solver._step_jit = viewer.lbm_solver.get_step_jit()
        
        print(f"Updated eta_max to {eta_max:.2f}")
