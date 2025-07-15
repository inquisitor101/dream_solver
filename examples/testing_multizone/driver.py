# Import preliminary modules.
from dream import *
from dream.compressible import Initial, CompressibleFlowSolver, flowfields
import ngsolve as ngs 
import netgen.occ as occ
from netgen.meshing import IdentificationType


def get_grid() -> ngs.Mesh:
    
    # Outer domain.
    domain_outer = occ.WorkPlane().RectangleC(5, 2).Face()
    domain_outer.name = "sdg"
    domain_outer.edges[0].name = 'sdg_bottom'
    domain_outer.edges[1].name = 'sdg_right'
    domain_outer.edges[2].name = 'sdg_top'
    domain_outer.edges[3].name = 'sdg_left'
    
    # Inner domain.
    domain_inner = occ.WorkPlane().Circle(0, 0, 0.5).Face()
    domain_inner.name = "hdg"
    for edge in domain_inner.edges:
        edge.name = 'interface'
    
    # Subtract inner from the outer domain.
    domain_outer -= domain_inner
    
    # Cylinder(/hole) domain.
    domain_cylinder = occ.WorkPlane().Circle(0, 0, 0.1).Face()
    domain_inner -= domain_cylinder
    domain_inner.edges[1].name = 'cylinder'
    
    # Glue them together, at their common face: 'interface'.
    geo = occ.Glue([domain_outer, domain_inner])
    
    # Make the vertical and horizontal boundaries periodic.
    geo.edges[0].Identify( geo.edges[3], "periodic_vertical", occ.IdentificationType.PERIODIC) # vertical
    geo.edges[1].Identify( geo.edges[2], "periodic_horizontal", occ.IdentificationType.PERIODIC) # horizontal

    #geo.faces[0].quad_dominated = True # SDG
    #geo.faces[1].quad_dominated = True # HDG
    
    # Mesh the domain(s).
    domain_inner.maxh = 0.2
    mesh = ngs.Mesh(occ.OCCGeometry(geo, dim=2).GenerateMesh(maxh=0.5, grading=0.25))
    
    # One beautiful mesh, coming right up :)
    return mesh


# Message output detail from netgen.
ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(4)

# Generate two-zone grid.
mesh = get_grid()

# Solver configuration: Compressible (inviscid) flow.
cfg = CompressibleFlowSolver(mesh)

cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "acoustic"
cfg.mach_number = 0.1

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "conservative_domainsplit_sdg_hdg"
cfg.fem.order = 1

cfg.time = "transient"
cfg.fem.scheme = "imex_euler"
cfg.time.timer.interval = (0.0, 0.5)
cfg.time.timer.step = 0.01

cfg.fem.solver = "direct"
cfg.fem.solver.method = "newton"
cfg.fem.solver.method.damping_factor = 1
cfg.fem.solver.max_iterations = 10
cfg.fem.solver.convergence_criterion = 1e-10

Uic = cfg.get_farfield_fields( (1,0) ) 

cfg.bcs['sdg_left|sdg_right'] = "periodic"
cfg.bcs['sdg_top|sdg_bottom'] = "periodic"
cfg.dcs['sdg|hdg'] = Initial(fields=Uic)

# Curve mesh (optional).
mesh.Curve(cfg.fem.order)

# Allocate the necessary data.
cfg.initialize()


# Post-processing.
fields = cfg.get_solution_fields()
cfg.io.vtk.fields = fields
cfg.io.vtk.enable = True
cfg.io.vtk.rate = 10
cfg.io.vtk.subdivision = cfg.fem.order + 1
cfg.io.vtk.filename = "test"


with ngs.TaskManager():
    cfg.solve()




