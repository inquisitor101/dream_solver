"""
Inviscid non-lifiting flow over a circular cylinder

We benchmark the compressible Solver in the special case of an
inviscid non-lifting flow around a circular cylinder.
From literature it is well known, that the analytic solution is composed
by the superposition of two potential flows, namely the uniform flow
and doublet flow.

As validation, we compare the numerical pressure coefficient along the
cylinder surface against the analytical solution. The exact pressure coefficient
is given by c_p = 1 - 4 * sin(phi)**2.

Literature:
[1] - J.D. Anderson,
      Fundamentals of Aerodynamics, 6th edition
      New York, NY: McGraw-Hill Education, 2017.
"""
# ------- Import Modules ------- #
from ngsolve import *
from dream.mesh import get_cylinder_omesh
from dream.compressible import CompressibleFlowSolver, InviscidWall, FarField, Initial, flowfields
from dream.io import PointSensor

ngsglobals.msg_level = 0
SetNumThreads(8)

# ------- Define Geometry and Mesh ------- #
ri = 1
ro = ri * 30
mesh = get_cylinder_omesh(ri, ro, 28, 12, geom=1.8)

# ------- Set Configuration ------- #
cfg = CompressibleFlowSolver(mesh)
cfg.time = "pseudo_time_stepping"
cfg.time.timer.step = 0.001
cfg.time.max_time_step = 10

cfg.fem = 'conservative_hdg'
cfg.fem.order = 5
cfg.fem.scheme = "implicit_euler"

cfg.scaling = "acoustic"
cfg.riemann_solver = "hllem"
cfg.mach_number = 0.001
cfg.equation_of_state.heat_capacity_ratio = 1.4

cfg.fem.solver = "direct"
cfg.fem.solver.method = "newton"
cfg.fem.solver.method.max_iterations = 300
cfg.fem.solver.method.convergence_criterion = 1e-12


# ------- Curve Mesh ------- #
mesh.Curve(cfg.fem.order)

# ------- Setup Boundary Conditions and Domain Conditions ------- #
Uinf = cfg.get_farfield_fields((1, 0))
cfg.bcs['left|right'] = FarField(fields=Uinf)
cfg.bcs['cylinder'] = InviscidWall()
cfg.dcs['default'] = Initial(fields=Uinf)

# ------- Setup Spaces and Gridfunctions ------- #
cfg.initialize()

# ------- Setup Outputs ------- #
fields = cfg.get_solution_fields()
cfg.io.draw(fields, autoscale=False, min=-1e-4, max=1e-4)

c_p = flowfields(c_p=cfg.pressure_coefficient(fields, Uinf))
cfg.io.sensor.enable = True
cfg.io.sensor.add(PointSensor.from_boundary(c_p, mesh, 'cylinder', name='pressure_coefficient'))

# ------- Solve System ------- #
with TaskManager():
    cfg.solve()

# ------- Postprocess Results ------- #
import numpy as np
import matplotlib.pyplot as plt

index, data = cfg.io.sensor.load_csv_as_dict('pressure_coefficient')
coords = np.array([header[0][1:-1].split(',') for header in data], dtype=float)
cp_h = np.array(list(data.values()))[:, 1]

# ------- Calculate Angle and Sort ------- #
phi_h = np.angle(coords[:, 0] + 1j*coords[:, 1])
cp_h = cp_h[np.argsort(phi_h)]
phi_h.sort()

# ------- Exact solution ------- #
phi = np.linspace(-np.pi, np.pi, 100)
cp = 1 - 4 * np.sin(phi)**2

# ------- Draw ------- #
fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.subplots(1, 1)
ax.plot(phi, cp, color='k')
ax.scatter(phi_h, cp_h, color='red')
ax.set_xlabel(r"$\varphi$")
ax.set_ylabel(r"$c_p$")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.show()
