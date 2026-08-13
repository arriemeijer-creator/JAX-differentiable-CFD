#!/usr/bin/env python3
"""
2D LBM flow past cylinder using Taichi (CPU) – High-Speed Memory Alignment.
Fully stable, explicit multi-kernel approach, optimized memory layout.
"""

import sys
from PyQt5 import QtCore, QtWidgets

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

import pyqtgraph as pg
import numpy as np
import taichi as ti

# Max out processing allocations across your target CPU cores
ti.init(arch=ti.cpu, cpu_max_num_threads=4, advanced_optimization=True)

# ----------------------------------------------------------------------
# Simulation parameters (ultra-stable)
Nx = 200
Ny = 80
Re = 10.0
U_in = 0.02
rho0 = 1.0
nu = U_in * (Ny / 3) / Re
omega = 1.0 / (3.0 * nu + 0.5)

print(f"omega = {omega:.4f}")

cx, cy = Nx // 4, Ny // 2
radius = Ny // 10

# ----------------------------------------------------------------------
# MEMORY FIX: Shift the velocity track index '9' to the trailing dimension.
# This forces the 9 population values for cell (i,j) to sit side-by-side in RAM.
f = ti.field(dtype=ti.f32, shape=(Nx, Ny, 9))
f_new = ti.field(dtype=ti.f32, shape=(Nx, Ny, 9))

rho = ti.field(dtype=ti.f32, shape=(Nx, Ny))
ux = ti.field(dtype=ti.f32, shape=(Nx, Ny))
uy = ti.field(dtype=ti.f32, shape=(Nx, Ny))
mask = ti.field(dtype=ti.f32, shape=(Nx, Ny))
mag = ti.field(dtype=ti.f32, shape=(Nx, Ny))

# Constants
w = ti.field(dtype=ti.f32, shape=9)
ex = ti.field(dtype=ti.i32, shape=9)
ey = ti.field(dtype=ti.i32, shape=9)
opp = ti.field(dtype=ti.i32, shape=9)

w.from_numpy(np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32))
ex.from_numpy(np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32))
ey.from_numpy(np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32))
opp.from_numpy(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32))

# Initial condition (uniform flow mapped to trailing dimensions)
f_host = np.zeros((Nx, Ny, 9), dtype=np.float32)
w_np = w.to_numpy()
ex_np = ex.to_numpy()
for i in range(Nx):
    for j in range(Ny):
        for k in range(9):
            f_host[i, j, k] = w_np[k] * rho0 * (
                1.0 + 3.0 * (ex_np[k] * U_in) + 4.5 * (ex_np[k] * U_in)**2 - 1.5 * U_in**2
            )
f.from_numpy(f_host)

# Cylinder mask
mask_np = np.ones((Nx, Ny), dtype=np.float32)
x_coords = np.arange(Nx).reshape(Nx, 1)
y_coords = np.arange(Ny).reshape(1, Ny)
mask_np[(x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2] = 0.0
mask.from_numpy(mask_np)

# ----------------------------------------------------------------------
# Optimized Kernels

@ti.kernel
def compute_macroscopic():
    for i, j in ti.ndrange(Nx, Ny):
        if mask[i, j] < 0.5:
            continue
        rho_local = 0.0
        ux_local = 0.0
        uy_local = 0.0
        # ti.static() unrolls the loop, replacing indexing overhead with direct math instructions
        for k in ti.static(range(9)):
            val = f[i, j, k]
            rho_local += val
            ux_local += val * ti.cast(ex[k], ti.f32)
            uy_local += val * ti.cast(ey[k], ti.f32)
        rho[i, j] = rho_local
        ux[i, j] = ux_local / rho_local
        uy[i, j] = uy_local / rho_local

@ti.kernel
def collide():
    for i, j in ti.ndrange(Nx, Ny):
        if mask[i, j] < 0.5:
            for k in ti.static(range(9)):
                f_new[i, j, k] = f[i, j, k]
            continue

        rho_local = rho[i, j]
        ux_local = ux[i, j]
        uy_local = uy[i, j]

        for k in ti.static(range(9)):
            eu = ti.cast(ex[k], ti.f32) * ux_local + ti.cast(ey[k], ti.f32) * uy_local
            eq = w[k] * rho_local * (
                1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * (ux_local * ux_local + uy_local * uy_local)
            )
            f_new[i, j, k] = f[i, j, k] + omega * (eq - f[i, j, k])

@ti.kernel
def stream():
    for i, j in ti.ndrange(Nx, Ny):
        if mask[i, j] < 0.5:
            for k in ti.static(range(9)):
                f[i, j, k] = f_new[i, j, k]
            continue

        for k in ti.static(range(9)):
            ni = i + ex[k]
            nj = j + ey[k]
            if 0 <= ni < Nx and 0 <= nj < Ny:
                if mask[ni, nj] < 0.5:
                    f[i, j, opp[k]] = f_new[i, j, k]
                else:
                    f[ni, nj, k] = f_new[i, j, k]
            else:
                f[i, j, opp[k]] = f_new[i, j, k]

@ti.kernel
def apply_boundaries():
    # Zou-He inlet (left boundary)
    for j in range(Ny):
        rho_in = 1.0
        ux_in = U_in
        f[0, j, 1] = f[0, j, 3] + 2.0 / 3.0 * rho_in * ux_in
        f[0, j, 5] = f[0, j, 7] + 0.5 * (f[0, j, 4] - f[0, j, 2]) + 1.0 / 6.0 * rho_in * ux_in
        f[0, j, 8] = f[0, j, 6] + 0.5 * (f[0, j, 2] - f[0, j, 4]) + 1.0 / 6.0 * rho_in * ux_in

    # Outlet (right boundary)
    for j in range(Ny):
        for k in ti.static(range(9)):
            f[Nx - 1, j, k] = f[Nx - 2, j, k]

    # Bottom wall (j=0)
    for i in range(Nx):
        f[i, 0, 4] = f[i, 0, opp[4]]
        f[i, 0, 7] = f[i, 0, opp[7]]
        f[i, 0, 8] = f[i, 0, opp[8]]

    # Top wall (j=Ny-1)
    for i in range(Nx):
        f[i, Ny - 1, 2] = f[i, Ny - 1, opp[2]]
        f[i, Ny - 1, 5] = f[i, Ny - 1, opp[5]]
        f[i, Ny - 1, 6] = f[i, Ny - 1, opp[6]]

@ti.kernel
def compute_mag():
    for i, j in ti.ndrange(Nx, Ny):
        mag[i, j] = ti.sqrt(ux[i, j]**2 + uy[i, j]**2)

# ----------------------------------------------------------------------
# PyQtGraph viewer
class LBMViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized Taichi LBM – Stable & Fast")
        self.resize(800, 350)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot = pg.ImageView()
        layout.addWidget(self.plot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(1) # Drop scheduling delay to minimum
        self.step = 0

    def update_sim(self):
        # Increased steps per frame to emphasize memory performance improvements
        for _ in range(10):  
            compute_macroscopic()
            collide()
            stream()
            apply_boundaries()
            self.step += 1

        compute_mag()
        arr = mag.to_numpy()
        mask_np = mask.to_numpy()
        arr[mask_np < 0.5] = 0.0
        self.plot.setImage(arr.T, autoRange=False, autoLevels=False, levels=[0, U_in * 1.5])

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    win = LBMViewer()
    win.show()
    sys.exit(app.exec_())
