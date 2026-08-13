"""
3D Baseline Viewer for AeroJAX with STL Voxelization & Interaction Engine
Features: Voxelization (dx/grid), face picking, selection, transform,
clipping planes, measurements, domain creation.
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QSlider, QButtonGroup,
    QRadioButton
)
from PyQt6.QtCore import Qt, QTimer

import vispy
from vispy import scene
from vispy.visuals.transforms import MatrixTransform
from vispy.geometry import create_box
from vispy.scene.visuals import Text

import trimesh

try:
    import skimage  # noqa: F401
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# --------------------------------------------------------------------
#  Interaction Engine – handles picking, transforms, clipping, etc.
# --------------------------------------------------------------------
class InteractionEngine:
    """Handles all interactive tools on the canvas."""
    
    def __init__(self, canvas):
        self.canvas = canvas
        self.view = canvas.view
        
        # State
        self.selected_mesh = None          # the visual
        self.selected_face_index = None
        self.selected_face_vertices = None
        self.highlight_visual = None
        self.transform_mode = 'select'     # 'select', 'translate', 'rotate', 'scale'
        self.dragging = False
        self.drag_start = None
        self.drag_initial_transform = None
        self.clipping_enabled = False
        self.clip_plane = None             # (point, normal)
        self.clip_offset = 0.0
        self.clip_normal = [0, 0, 1]       # default z
        self.measurements = []             # list of (visual_line, text_visual)
        self.domain_box = None             # visual for domain
        self.domain_corners = []           # two clicked points
        self.measurement_points = []       # temp for measuring
        
        # Connect mouse events
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
    
    def set_transform_mode(self, mode):
        self.transform_mode = mode
        # Change cursor
        if mode == 'select':
            self.canvas.native.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode in ('translate', 'rotate', 'scale'):
            self.canvas.native.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.canvas.native.setCursor(Qt.CursorShape.ArrowCursor)
    
    def on_mouse_press(self, event):
        """Handle click: picking or starting transform."""
        if event.button == 1:  # left click
            # If in selection/transform mode, try to pick
            if self.transform_mode in ('select', 'translate', 'rotate', 'scale'):
                picked = self.pick(event.pos)
                if picked:
                    self.select_mesh(picked)
                    if self.transform_mode != 'select':
                        self.dragging = True
                        self.drag_start = event.pos
                        self.drag_initial_transform = self.selected_mesh.transform.matrix.copy()
                else:
                    self.deselect_all()
            elif self.transform_mode == 'measure_distance':
                self.add_measurement_point(event.pos)
            elif self.transform_mode == 'measure_angle':
                self.add_angle_point(event.pos)
            elif self.transform_mode == 'create_domain':
                self.add_domain_corner(event.pos)
        elif event.button == 2:  # right click – reset transform
            if self.selected_mesh is not None:
                self.selected_mesh.transform.matrix = np.eye(4)
                self.canvas.update()
    
    def on_mouse_move(self, event):
        """Drag to transform selected mesh."""
        if self.dragging and self.selected_mesh is not None:
            if self.transform_mode == 'translate':
                delta = event.pos - self.drag_start
                cam = self.view.camera
                dist = cam.distance
                scale = dist / 500.0
                dx = delta[0] * scale
                dy = -delta[1] * scale
                T = np.eye(4)
                T[:3, 3] = [dx, dy, 0]
                self.selected_mesh.transform.matrix = self.drag_initial_transform @ T
                self.canvas.update()
            elif self.transform_mode == 'rotate':
                delta = event.pos - self.drag_start
                angle = delta[0] * 0.01
                # Build rotation matrix around Y
                c = np.cos(angle)
                s = np.sin(angle)
                R = np.eye(4)
                R[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                self.selected_mesh.transform.matrix = self.drag_initial_transform @ R
                self.canvas.update()
            elif self.transform_mode == 'scale':
                delta = event.pos - self.drag_start
                scale_factor = 1.0 + delta[0] * 0.005
                S = np.eye(4)
                S[0,0] = S[1,1] = S[2,2] = scale_factor
                self.selected_mesh.transform.matrix = self.drag_initial_transform @ S
                self.canvas.update()
    
    def on_mouse_release(self, event):
        self.dragging = False
    
    def pick(self, pos):
        """Perform picking (simplified: always select main mesh if exists)."""
        if self.canvas.voxel_mesh_visual is not None:
            return self.canvas.voxel_mesh_visual
        return None
    
    def select_mesh(self, visual):
        """Highlight selected mesh."""
        self.deselect_all()
        self.selected_mesh = visual
        if visual is not None:
            self._orig_color = visual.color
            visual.color = (0, 1, 0, 0.8)
            self.canvas.update()
    
    def deselect_all(self):
        if self.selected_mesh is not None:
            if hasattr(self, '_orig_color'):
                self.selected_mesh.color = self._orig_color
            self.selected_mesh = None
            self.canvas.update()
    
    # ----- Clipping -----
    def set_clipping_enabled(self, enabled):
        self.clipping_enabled = enabled
        self.update_clipping()
    
    def set_clip_offset(self, offset):
        self.clip_offset = offset
        self.update_clipping()
    
    def set_clip_normal(self, normal):
        self.clip_normal = normal
        self.update_clipping()
    
    def update_clipping(self):
        """Apply clipping by slicing the original mesh."""
        canvas = self.canvas
        if not self.clipping_enabled or canvas.current_mesh is None:
            # Restore voxelized mesh (if we have stored its data)
            if hasattr(canvas, '_voxel_vertices') and hasattr(canvas, '_voxel_faces'):
                # Re-create the voxel visual
                if canvas.voxel_mesh_visual is not None:
                    canvas.voxel_mesh_visual.parent = None
                canvas.voxel_mesh_visual = scene.visuals.Mesh(
                    vertices=canvas._voxel_vertices,
                    faces=canvas._voxel_faces,
                    color=(0.2, 0.6, 1.0, 0.9),
                    shading='smooth',
                    parent=canvas.view.scene
                )
                canvas.update()
            return
        
        # Slice the original mesh
        mesh = canvas.current_mesh
        if mesh is None:
            return
        center = mesh.vertices.mean(axis=0)
        normal = np.array(self.clip_normal)
        # Compute a point on the plane
        origin = center + normal * self.clip_offset
        try:
            sliced = mesh.slice_plane(plane_origin=origin, plane_normal=normal)
            if sliced is None or len(sliced.vertices) == 0:
                return
            # Convert to visual
            vertices = sliced.vertices.astype(np.float32)
            faces = sliced.faces.astype(np.int32)
            # Replace visual
            if canvas.voxel_mesh_visual is not None:
                canvas.voxel_mesh_visual.parent = None
            canvas.voxel_mesh_visual = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=(0.2, 0.6, 1.0, 0.9),
                shading='smooth',
                parent=canvas.view.scene
            )
            canvas.update()
        except Exception as e:
            print(f"Clipping error: {e}")
    
    # ----- Measurements -----
    def add_measurement_point(self, pos):
        # For demo, use mesh center as point; real implementation would ray-cast
        if self.canvas.current_mesh is not None:
            center = self.canvas.current_mesh.vertices.mean(axis=0)
            self.measurement_points.append(center)
            if len(self.measurement_points) >= 2:
                self.draw_measurement()
    
    def add_angle_point(self, pos):
        # Similar to distance but needs 3 points; for simplicity we just use distance
        self.add_measurement_point(pos)  # placeholder
    
    def draw_measurement(self):
        if len(self.measurement_points) >= 2:
            p1 = self.measurement_points[-2]
            p2 = self.measurement_points[-1]
            line = scene.visuals.Line(
                pos=np.array([p1, p2]),
                color=(1, 0, 0, 1),
                parent=self.canvas.view.scene
            )
            dist = np.linalg.norm(p2-p1)
            mid = (p1+p2)/2
            text = Text(str(dist)[:6], pos=mid, color=(1,1,1,1), parent=self.canvas.view.scene)
            self.measurements.append((line, text))
            self.measurement_points = []
    
    # ----- Domain creation -----
    def add_domain_corner(self, pos):
        if self.canvas.current_mesh is not None:
            center = self.canvas.current_mesh.vertices.mean(axis=0)
            self.domain_corners.append(center)
            if len(self.domain_corners) >= 2:
                self.draw_domain_box()
    
    def draw_domain_box(self):
        if len(self.domain_corners) >= 2:
            p1 = self.domain_corners[-2]
            p2 = self.domain_corners[-1]
            center = (p1+p2)/2
            size = np.abs(p2-p1)
            # Ensure positive size
            size = np.maximum(size, 0.01)
            box_data = create_box(width=size[0], height=size[1], depth=size[2])
            vertices = box_data['vertices'] + center - size/2
            faces = box_data['faces']
            if self.domain_box is not None:
                self.domain_box.parent = None
            self.domain_box = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=(1, 0.5, 0, 0.2),
                wireframe=True,
                parent=self.canvas.view.scene
            )
            self.canvas.update()
            self.domain_corners = []


# --------------------------------------------------------------------
#  VisPyCanvas (with voxel storage for clipping restore)
# --------------------------------------------------------------------
class VisPyCanvas(scene.SceneCanvas):
    def __init__(self, parent=None):
        super().__init__(parent=parent, keys='interactive')
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 5.0
        self.view.camera.center = (0, 0, 0)

        # Grid and axes
        scene.visuals.GridLines(parent=self.view.scene, color=(0.5,0.5,0.5,0.3))
        axis_data = [
            ([0,0,0], [2,0,0], [1,0,0,1]),
            ([0,0,0], [0,2,0], [0,1,0,1]),
            ([0,0,0], [0,0,2], [0,0,1,1]),
        ]
        for start, end, color in axis_data:
            scene.visuals.Line(pos=np.array([start, end]), color=color,
                               parent=self.view.scene, width=2)

        # Data storage
        self.current_mesh = None
        self.current_extent = None
        self.voxel_mesh_visual = None
        self._voxel_vertices = None   # store for clipping restore
        self._voxel_faces = None
        self._create_placeholder_cube()

        # Interaction engine
        self.interaction = InteractionEngine(self)

        self.freeze()
        self.show()

    def _create_placeholder_cube(self):
        vertices = np.array([
            [-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5],
            [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5]
        ], dtype=np.float32)
        faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2],
            [3,2,6],[3,6,7],[0,4,5],[0,5,1]
        ], dtype=np.int32)
        face_colors = np.array([
            [1,0,0,1],[1,0,0,1],[0,1,0,1],[0,1,0,1],
            [0,0,1,1],[0,0,1,1],[1,1,0,1],[1,1,0,1],
            [1,0,1,1],[1,0,1,1],[0,1,1,1],[0,1,1,1]
        ], dtype=np.float32)
        self.voxel_mesh_visual = scene.visuals.Mesh(
            vertices=vertices, faces=faces, face_colors=face_colors,
            shading='flat', parent=self.view.scene
        )
        self.voxel_mesh_visual.transform = MatrixTransform()

    def set_mesh(self, mesh):
        self.current_mesh = mesh
        if mesh is not None:
            bounds = mesh.bounds
            extent = bounds[1] - bounds[0]
            self.current_extent = max(extent)
        else:
            self.current_extent = None

    def voxelize_mesh(self, dx):
        if self.current_mesh is None:
            return False, "No mesh loaded."
        if not SKIMAGE_AVAILABLE:
            return False, "scikit‑image missing. Run: pip install scikit-image"

        try:
            mesh = self.current_mesh
            voxel_grid = trimesh.voxel.creation.voxelize(mesh, pitch=dx)
            if voxel_grid is None or voxel_grid.matrix is None:
                raise ValueError("Voxelization failed.")
            voxel_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
                voxel_grid.matrix, pitch=dx
            )
            if voxel_mesh is None or len(voxel_mesh.vertices) == 0:
                raise ValueError("Failed to generate surface mesh.")

            center = voxel_mesh.vertices.mean(axis=0)
            vertices = voxel_mesh.vertices - center
            faces = voxel_mesh.faces

            # Store for clipping restore
            self._voxel_vertices = vertices.astype(np.float32)
            self._voxel_faces = faces.astype(np.int32)

            if self.voxel_mesh_visual is not None:
                self.voxel_mesh_visual.parent = None

            self.voxel_mesh_visual = scene.visuals.Mesh(
                vertices=self._voxel_vertices,
                faces=self._voxel_faces,
                color=(0.2, 0.6, 1.0, 0.9),
                shading='smooth',
                parent=self.view.scene
            )

            extents = vertices.max(axis=0) - vertices.min(axis=0)
            max_ext = max(extents)
            if max_ext > 0:
                self.view.camera.distance = max_ext * 2.5
            self.view.camera.center = (0,0,0)
            self.update()

            num_voxels = int(np.sum(voxel_grid.matrix))
            return True, f"Voxels: {num_voxels}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def clear_mesh(self):
        if self.voxel_mesh_visual is not None:
            self.voxel_mesh_visual.parent = None
            self.voxel_mesh_visual = None
        self._create_placeholder_cube()
        self.update()


# --------------------------------------------------------------------
#  Control Panel with Interaction Tools
# --------------------------------------------------------------------
class ControlPanel3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.current_stl_path = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5,5,5,5)
        layout.setSpacing(10)

        # ---- STL Loader ----
        file_group = QGroupBox("STL Loader")
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton("📂 Load STL")
        self.load_btn.clicked.connect(self.load_stl)
        file_layout.addWidget(self.load_btn)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        # Resolution mode
        res_group = QGroupBox("Resolution Mode")
        res_layout = QVBoxLayout()
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Voxel Size (dx)", "Grid Size"])
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        res_layout.addLayout(mode_row)

        self.dx_widget = QWidget()
        dx_row = QHBoxLayout(self.dx_widget)
        dx_row.addWidget(QLabel("dx ="))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.001, 10.0)
        self.dx_spin.setSingleStep(0.01)
        self.dx_spin.setValue(0.05)
        self.dx_spin.setDecimals(3)
        dx_row.addWidget(self.dx_spin)
        dx_row.addStretch()
        res_layout.addWidget(self.dx_widget)

        self.grid_widget = QWidget()
        grid_row = QHBoxLayout(self.grid_widget)
        grid_row.addWidget(QLabel("Voxels per axis:"))
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["32", "64", "128", "256"])
        grid_row.addWidget(self.grid_combo)
        grid_row.addStretch()
        res_layout.addWidget(self.grid_widget)

        self.on_mode_changed(1)

        self.apply_btn = QPushButton("🔄 Apply")
        self.apply_btn.clicked.connect(self.apply_voxelization)
        res_layout.addWidget(self.apply_btn)
        res_group.setLayout(res_layout)
        file_layout.addWidget(res_group)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # ---- Interaction Tools ----
        inter_group = QGroupBox("Interaction Tools")
        inter_layout = QVBoxLayout()

        # Transform mode radio buttons
        inter_layout.addWidget(QLabel("Transform Mode:"))
        self.transform_btn_group = QButtonGroup(self)
        modes = [("Select", "select"), ("Translate", "translate"),
                 ("Rotate", "rotate"), ("Scale", "scale")]
        self.transform_btns = []
        for text, mode in modes:
            rb = QRadioButton(text)
            rb.setProperty("mode", mode)
            self.transform_btn_group.addButton(rb)
            if mode == "select":
                rb.setChecked(True)
            self.transform_btns.append(rb)
            inter_layout.addWidget(rb)
        self.transform_btn_group.buttonClicked.connect(self.on_transform_mode_changed)

        # Clipping
        clip_row = QHBoxLayout()
        self.clip_cb = QCheckBox("Clipping Enabled")
        self.clip_cb.stateChanged.connect(self.on_clip_toggle)
        clip_row.addWidget(self.clip_cb)
        inter_layout.addLayout(clip_row)

        clip_slider_row = QHBoxLayout()
        clip_slider_row.addWidget(QLabel("Offset:"))
        self.clip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_slider.setRange(-100, 100)
        self.clip_slider.setValue(0)
        self.clip_slider.valueChanged.connect(self.on_clip_offset)
        clip_slider_row.addWidget(self.clip_slider)
        inter_layout.addLayout(clip_slider_row)

        # Measurements
        measure_row = QHBoxLayout()
        self.measure_dist_btn = QPushButton("Measure Distance")
        self.measure_dist_btn.clicked.connect(lambda: self.set_measure_mode("measure_distance"))
        measure_row.addWidget(self.measure_dist_btn)
        self.measure_angle_btn = QPushButton("Measure Angle")
        self.measure_angle_btn.clicked.connect(lambda: self.set_measure_mode("measure_angle"))
        measure_row.addWidget(self.measure_angle_btn)
        inter_layout.addLayout(measure_row)

        clear_meas_btn = QPushButton("Clear Measurements")
        clear_meas_btn.clicked.connect(self.clear_measurements)
        inter_layout.addWidget(clear_meas_btn)

        # Domain
        domain_row = QHBoxLayout()
        self.domain_btn = QPushButton("Create Domain")
        self.domain_btn.clicked.connect(lambda: self.set_measure_mode("create_domain"))
        domain_row.addWidget(self.domain_btn)
        clear_domain_btn = QPushButton("Clear Domain")
        clear_domain_btn.clicked.connect(self.clear_domain)
        domain_row.addWidget(clear_domain_btn)
        inter_layout.addLayout(domain_row)

        inter_group.setLayout(inter_layout)
        layout.addWidget(inter_group)

        # ---- View Controls ----
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout()
        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self.reset_camera)
        view_layout.addWidget(reset_btn)
        wireframe_cb = QCheckBox("Show Wireframe")
        wireframe_cb.stateChanged.connect(self.toggle_wireframe)
        view_layout.addWidget(wireframe_cb)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # ---- Simulation Controls ----
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.clicked.connect(self.toggle_simulation)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_simulation)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        sim_layout.addLayout(btn_layout)
        self.info_label = QLabel("Simulation: Idle")
        sim_layout.addWidget(self.info_label)
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # ---- Visualization Options ----
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        self.velocity_cb = QCheckBox("Velocity")
        self.velocity_cb.setChecked(True)
        self.pressure_cb = QCheckBox("Pressure")
        self.vorticity_cb = QCheckBox("Vorticity")
        vis_layout.addWidget(self.velocity_cb)
        vis_layout.addWidget(self.pressure_cb)
        vis_layout.addWidget(self.vorticity_cb)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        layout.addStretch()
        self.setLayout(layout)

    def on_mode_changed(self, index):
        self.dx_widget.setVisible(index == 0)
        self.grid_widget.setVisible(index == 1)

    def on_transform_mode_changed(self, btn):
        mode = btn.property("mode")
        canvas = self.parent_viewer.canvas
        canvas.interaction.set_transform_mode(mode)

    def set_measure_mode(self, mode):
        canvas = self.parent_viewer.canvas
        canvas.interaction.set_transform_mode(mode)

    def on_clip_toggle(self, state):
        canvas = self.parent_viewer.canvas
        canvas.interaction.set_clipping_enabled(state == Qt.CheckState.Checked)

    def on_clip_offset(self, value):
        offset = value / 100.0  # map -1 to 1
        canvas = self.parent_viewer.canvas
        canvas.interaction.set_clip_offset(offset)

    def clear_measurements(self):
        canvas = self.parent_viewer.canvas
        for line, text in canvas.interaction.measurements:
            line.parent = None
            text.parent = None
        canvas.interaction.measurements = []
        canvas.update()

    def clear_domain(self):
        canvas = self.parent_viewer.canvas
        if canvas.interaction.domain_box is not None:
            canvas.interaction.domain_box.parent = None
            canvas.interaction.domain_box = None
            canvas.update()

    # ---- other methods ----
    def load_stl(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select STL File", "", "STL Files (*.stl);;All Files (*)"
        )
        if not filepath:
            return
        self.current_stl_path = filepath
        self.file_label.setText(f"Loading: {filepath}")
        try:
            mesh = trimesh.load_mesh(filepath)
            if mesh is None:
                raise ValueError("Failed to load mesh.")
            canvas = self.parent_viewer.canvas
            canvas.set_mesh(mesh)
            self.file_label.setText(f"✅ Loaded: {filepath}  (Press Apply to voxelize)")
            self.info_label.setText("STL loaded. Click Apply to voxelize.")
        except Exception as e:
            self.file_label.setText(f"❌ {filepath}")
            QMessageBox.critical(self, "Load Error", str(e))

    def apply_voxelization(self):
        if self.current_stl_path is None:
            QMessageBox.warning(self, "No File", "Load an STL first.")
            return
        canvas = self.parent_viewer.canvas
        if canvas.current_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Mesh not loaded.")
            return
        mode = self.mode_combo.currentIndex()
        if mode == 0:
            dx = self.dx_spin.value()
        else:
            grid_size = int(self.grid_combo.currentText())
            extent = canvas.current_extent
            if extent is None or extent <= 0:
                QMessageBox.warning(self, "Error", "Cannot determine model extent.")
                return
            dx = extent / grid_size

        success, message = canvas.voxelize_mesh(dx)
        if success:
            self.info_label.setText(message)
            self.file_label.setText(f"✅ {self.current_stl_path}  (voxelized)")
        else:
            self.info_label.setText("Voxelization failed")
            QMessageBox.warning(self, "Voxelization Failed", message)

    def reset_camera(self):
        canvas = self.parent_viewer.canvas
        canvas.view.camera.reset()

    def toggle_wireframe(self, state):
        canvas = self.parent_viewer.canvas
        mesh = canvas.voxel_mesh_visual
        if mesh is None:
            return
        mesh.draw_mode = 'lines' if (state == Qt.CheckState.Checked) else 'triangles'
        canvas.update()

    def toggle_simulation(self):
        self.run_btn.setText("⏸ Pause")
        self.info_label.setText("Simulation: Running")
        if hasattr(self.parent_viewer, 'start_simulation'):
            self.parent_viewer.start_simulation()

    def stop_simulation(self):
        self.run_btn.setText("▶ Run")
        self.info_label.setText("Simulation: Stopped")
        if hasattr(self.parent_viewer, 'stop_simulation'):
            self.parent_viewer.stop_simulation()


# --------------------------------------------------------------------
#  Main Window
# --------------------------------------------------------------------
class MainViewer3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroJAX 3D Viewer - STL Voxelization + Interaction")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        self.canvas = VisPyCanvas(parent=self)
        main_layout.addWidget(self.canvas.native, 3)

        self.control_panel = ControlPanel3D(parent=self)
        main_layout.addWidget(self.control_panel, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(16)
        self.sim_running = False

    def _update(self):
        pass

    def start_simulation(self):
        self.sim_running = True
        self.control_panel.info_label.setText("Simulation: Running")

    def stop_simulation(self):
        self.sim_running = False
        self.control_panel.info_label.setText("Simulation: Idle")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


# --------------------------------------------------------------------
def main():
    vispy.app.use_app('pyqt6')
    qapp = QApplication(sys.argv)
    window = MainViewer3D()
    window.show()
    sys.exit(qapp.exec())

if __name__ == "__main__":
    main()