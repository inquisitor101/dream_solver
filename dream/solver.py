from __future__ import annotations
import ngsolve as ngs
import logging

from typing import Optional, NamedTuple, Any
from math import isnan

from .sensor import Sensor

from .compressible import CompressibleFlowConfiguration

from .config import UniqueConfiguration, MultipleConfiguration, multiple, any, unique
from .time_schemes import StationaryConfig, TransientConfig, PseudoTimeSteppingConfig

logger = logging.getLogger(__name__)


class BonusIntegrationOrder(UniqueConfiguration):

    @any(default=0)
    def vol(self, vol):
        return int(vol)

    @any(default=0)
    def bnd(self, bnd):
        return int(bnd)

    @any(default=0)
    def bbnd(self, bbnd):
        return int(bbnd)

    @any(default=0)
    def bbbnd(self, bbbnd):
        return int(bbbnd)

    vol: int
    bnd: int
    bbnd: int
    bbbnd: int


class Inverse(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def get_inverse(self, blf: ngs.BilinearForm, pre: ngs.BilinearForm, fes: ngs.FESpace):
        NotImplementedError()


class Direct(Inverse):

    name: str = "direct"

    @any(default="umfpack")
    def solver(self, solver: str):
        return solver

    def get_inverse(self, blf: ngs.BilinearForm, pre: ngs.BilinearForm, fes: ngs.FESpace):
        return blf.mat.Inverse(fes.FreeDofs(blf.condense), inverse=self.solver)

    solver: str


class NonlinearMethod(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def reset_status(self):
        self.is_converged = False
        self.is_nan = False

    def set_solution_routine_attributes(self):
        raise NotImplementedError()

    def solve_update_step(self):
        raise NotImplementedError()

    def update_solution(self):
        raise NotImplementedError()

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        raise NotImplementedError()

    def log_iteration_error(self, error: float, it: int | None = None, t: float | None = None):
        msg = f"Iteration error: {error:8e}"
        if it is not None:
            msg += f" - iteration number: {it}"
        if t is not None:
            msg += f" - time step: {t}"
        logger.info(msg)


class NewtonsMethod(NonlinearMethod):

    name: str = "newton"

    @any(default=1)
    def damping_factor(self, damping_factor: float):
        return float(damping_factor)

    def set_solution_routine_attributes(self):
        self.fes = self.cfg.pde.fes
        self.gfu = self.cfg.pde.gfu
        self.inverse = self.cfg.solver.inverse

        self.blf = self.cfg.solver.blf
        self.lf = self.cfg.solver.lf

        self.residual = self.gfu.vec.CreateVector()
        self.temporary = self.gfu.vec.CreateVector()

    def solve_update_step(self):

        self.blf.Apply(self.gfu.vec, self.residual)
        self.residual.data -= self.lf.vec
        self.blf.AssembleLinearization(self.gfu.vec)

        inv = self.cfg.solver.inverse.get_inverse(self.blf, fes=self.fes)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

    def update_solution(self):
        self.gfu.vec.data -= self.damping_factor * self.temporary

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        error = ngs.sqrt(ngs.InnerProduct(self.temporary, self.residual)**2)

        self.is_converged = error < self.cfg.solver.convergence_criterion

        if isnan(error):
            self.is_nan = True
            logger.error("Solution process diverged!")

        self.log_iteration_error(error, it, t)
        self.error = error

    damping_factor: float


class Solver(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @multiple(default=Direct)
    def inverse(self, inverse: Inverse):
        return inverse

    def set_discrete_system(self):

        fes = self.cfg.pde.fes
        condense = self.cfg.optimizations.static_condensation

        self.blf = ngs.BilinearForm(fes, condense=condense)
        self.lf = ngs.LinearForm(fes)

        for name, cf in self.cfg.pde.blf:
            logger.debug(f"Adding {name} to the BilinearForm!")
            self.blf += cf

        for name, cf in self.cfg.pde.lf:
            logger.debug(f"Adding {name} to the LinearForm!")
            self.lf += cf

    inverse: Direct


class LinearSolver(Solver):

    name: str = "linear"

    def solve(self):

        for t in self.cfg.time.routine():
            raise NotImplementedError()


class NonlinearSolver(Solver):

    name: str = "nonlinear"

    @any(default=10)
    def max_iterations(self, max_iterations: int):
        return int(max_iterations)

    @any(default=1e-8)
    def convergence_criterion(self, convergence_criterion: float):
        if convergence_criterion <= 0:
            raise ValueError("Convergence Criterion must be greater zero!")
        return float(convergence_criterion)

    @multiple(default=NewtonsMethod)
    def method(self, method: NonlinearMethod):
        return method

    def set_discrete_system(self):
        super().set_discrete_system()
        self.method.set_solution_routine_attributes()

    def solve(self):

        for t in self.cfg.time.routine():

            self.method.reset_status()

            for it in range(self.max_iterations):

                self.cfg.time.iteration_update(it)

                self.method.solve_update_step()
                self.method.set_iteration_error(it, t)

                if self.method.is_nan:
                    break

                self.method.update_solution()

                if self.method.is_converged:
                    break

            if self.method.is_nan:
                break

    max_iterations: int
    convergence_criterion: float
    method: NonlinearMethod


class Optimizations(UniqueConfiguration):

    @any(default=False)
    def compile_flag(self, compile_flag: bool):
        return bool(compile_flag)

    @any(default=True)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @unique(default=BonusIntegrationOrder)
    def bonus_int_order(self, dict_):
        return dict_

    compile_flag: bool
    static_condensation: bool
    bonus_int_order: BonusIntegrationOrder


class SolverConfiguration(UniqueConfiguration):

    name: str = 'cfg'

    def __init__(self, mesh: ngs.Mesh, **kwargs) -> None:
        self.mesh = mesh
        super().__init__(root=self, **kwargs)

        if not hasattr(mesh, 'normal'):
            self.mesh.normal = ngs.specialcf.normal(mesh.dim)

        if not hasattr(mesh, 'tangential'):
            self.mesh.tangential = ngs.specialcf.tangential(mesh.dim)

        if not hasattr(mesh, 'mesh_size'):
            self.mesh.meshsize = ngs.specialcf.mesh_size

    @multiple(default=LinearSolver)
    def solver(self, solver: Solver):
        return solver

    @multiple(default=CompressibleFlowConfiguration)
    def pde(self, pde):
        return pde

    @multiple(default=StationaryConfig)
    def time(self, time):
        return time

    @unique(default=Optimizations)
    def optimizations(self, optimizations):
        return optimizations

    def set_grid_deformation(self):
        grid_deformation = self.pde.dcs.get_grid_deformation_function()
        self.mesh.SetDeformation(grid_deformation)

    def set_spaces_and_gridfunctions(self):
        self.pde.set_finite_element_spaces()
        self.pde.set_trial_and_test_functions()
        self.pde.set_gridfunctions()

        if not self.time.is_stationary:
            self.pde.set_transient_gridfunctions()

    def set_discrete_system_tree(self,
                                 set_initial: bool = True,
                                 set_boundary: bool = True):

        if set_boundary:
            self.pde.set_boundary_conditions()

        if set_initial:
            self.pde.set_initial_conditions()

        self.pde.set_discrete_system_tree()

    solver: LinearSolver | NonlinearSolver
    pde: CompressibleFlowConfiguration
    time: StationaryConfig | TransientConfig | PseudoTimeSteppingConfig
    optimizations: Optimizations


class Solver_:

    def __init__(self, mesh: ngs.Mesh, cfg: SolverConfiguration):

        self._formulation = formulation_factory(mesh, solver_configuration)
        self._status = SolverStatus(self.solver_configuration)
        self._sensors = []
        self._drawer = Drawer(self.formulation)

    @property
    def formulation(self) -> _Formulation:
        return self._formulation

    @property
    def dmesh(self):
        return self.formulation.dmesh

    @property
    def mesh(self):
        return self.formulation.mesh

    @property
    def solver_configuration(self):
        return self.formulation.cfg

    @property
    def boundary_conditions(self):
        return self.formulation.dmesh.bcs

    @property
    def domain_conditions(self):
        return self.formulation.dmesh.dcs

    @property
    def sensors(self) -> list[Sensor]:
        return self._sensors

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def drawer(self) -> Drawer:
        return self._drawer

    def setup(self, reinitialize: bool = True):

        if reinitialize:
            self.formulation.initialize()

        self.residual = self.formulation.gfu.vec.CreateVector()
        self.temporary = self.formulation.gfu.vec.CreateVector()

        self._set_linearform()
        self._set_bilinearform()

        if reinitialize and self.domain_conditions.initial_conditions:
            self._solve_initial()

    def solve_stationary(self,
                         increment_at_iteration: int = 10,
                         increment_time_step_factor: int = 10,
                         state_name: str = "stationary",
                         stop_at_iteration: bool = False) -> bool:

        self.solver_configuration.time.simulation = 'stationary'
        max_iterations = self.solver_configuration.max_iterations

        self.status.reset()
        saver = self.get_saver()

        for it in range(max_iterations):

            self.formulation.update_gridfunctions()
            self._update_pseudo_time_step(it, increment_at_iteration, increment_time_step_factor)

            self._solve_update_step()
            self.status.check_convergence(self.temporary, self.residual, iteration_number=it)

            self.drawer.redraw()

            if stop_at_iteration:
                input("Iteration stopped. Hit any key to continue.")

            if self.status.is_nan:
                break

            self._update_solution()

            if self.status.is_converged:
                break

        for sensor in self.sensors:
            sensor.take_single_sample()

        if self.solver_configuration.save_state:
            saver.save_gridfunction(name=state_name)

        self.drawer.redraw()

        self.formulation.update_gridfunctions()

    def solve_transient(self, state_name: str = "transient",  save_state_every_num_step: int = 1):

        self.solver_configuration.time.simulation = 'transient'

        self.status.reset()
        saver = self.get_saver()

        for idx, t in enumerate(self.solver_configuration.time):
            self._solve_timestep(t)

            for sensor in self.sensors:
                sensor.take_single_sample(t)

            if self.solver_configuration.save_state:
                if idx % save_state_every_num_step == 0:
                    saver.save_gridfunction(name=f"{state_name}_{t}")

            if self.status.is_nan:
                break

    def solve_perturbation(self) -> GridFunction:

        lf = LinearForm(self.formulation.fes)
        self.formulation.add_perturbation_linearform(lf)
        perturbation_gfu = self._solve_mass(lf)
        for gfu in self.formulation.gridfunctions.values():
            gfu.vec.data += perturbation_gfu.vec

        self.drawer.redraw()
        logger.info("Perturbation applied!")
        return perturbation_gfu

    def add_sensor(self, sensor: Sensor):
        sensor.assign_solver(self)
        self.sensors.append(sensor)

    def get_saver(self, directory_tree: Optional[ResultsDirectoryTree] = None) -> SolverSaver:
        saver = SolverSaver(self)
        if directory_tree is not None:
            saver.tree = directory_tree
        return saver

    def get_loader(self, directory_tree: Optional[ResultsDirectoryTree] = None) -> SolverLoader:
        loader = SolverLoader(self)
        if directory_tree is not None:
            loader.tree = directory_tree
        return loader

    def _set_linearform(self):

        fes = self.formulation.fes
        self.f = LinearForm(fes)
        self.formulation.add_forcing_linearform(self.f)
        self.f.Assemble()

    def _set_bilinearform(self):

        fes = self.formulation.fes

        condense = self.solver_configuration.static_condensation
        mu = self.solver_configuration.dynamic_viscosity

        self.blf = BilinearForm(fes, condense=condense)

        self.formulation.add_time_bilinearform(self.blf)
        self.formulation.add_convective_bilinearform(self.blf)

        if not mu.is_inviscid:
            self.formulation.add_diffusive_bilinearform(self.blf)

        self.formulation.add_boundary_conditions_bilinearform(self.blf)
        self.formulation.add_domain_conditions_bilinearform(self.blf)

    def _solve_mass(self, linearform: LinearForm) -> GridFunction:
        formulation = self.formulation
        gfu = GridFunction(formulation.fes)

        blf = BilinearForm(formulation.fes)
        formulation.add_mass_bilinearform(blf)

        blf.Assemble()
        linearform.Assemble()

        blf_inverse = blf.mat.Inverse(formulation.fes.FreeDofs(), inverse="sparsecholesky")

        gfu.vec.data = blf_inverse * linearform.vec
        return gfu

    def _solve_initial(self):
        lf = LinearForm(self.formulation.fes)
        self.formulation.add_initial_linearform(lf)
        self.formulation.gfu.vec.data = self._solve_mass(lf).vec
        self.formulation.update_gridfunctions(initial_value=True)

    def _solve_update_step(self):

        linear_solver = self.solver_configuration.linear_solver
        fes = self.formulation.fes
        gfu = self.formulation.gfu

        self.blf.Apply(gfu.vec, self.residual)
        self.residual.data -= self.f.vec
        self.blf.AssembleLinearization(gfu.vec)

        inv = self.blf.mat.Inverse(fes.FreeDofs(self.blf.condense), inverse=linear_solver)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

    def _update_solution(self):
        damping_factor = self.solver_configuration.damping_factor
        gfu = self.formulation.gfu

        gfu.vec.data -= damping_factor * self.temporary

    def _update_pseudo_time_step(self,
                                 iteration_number: int,
                                 increment_at_iteration: int = 10,
                                 increment_time_step_factor: int = 10):

        cfg = self.solver_configuration.time

        max_time_step = cfg.max_step
        old_time_step = cfg.step.Get()

        if max_time_step > old_time_step:
            if (iteration_number % increment_at_iteration == 0) and (iteration_number > 0):
                new_time_step = old_time_step * increment_time_step_factor

                if new_time_step > max_time_step:
                    new_time_step = max_time_step

                cfg.step = new_time_step
                logger.info(f"Successfully updated time step at iteration {iteration_number}")
                logger.info(f"Updated time step 𝚫t = {new_time_step}. Previous time step 𝚫t = {old_time_step}")

    def _solve_timestep(self, time, stop_at_iteration: bool = False) -> bool:

        max_iterations = self.solver_configuration.max_iterations

        self.status.reset_convergence_status()

        for it in range(max_iterations):

            self._solve_update_step()
            self.status.check_convergence(self.temporary, self.residual, time, it)

            if self.status.is_nan:
                break

            self._update_solution()

            if stop_at_iteration:
                input("Iteration stopped. Hit any key to continue.")

            if self.status.is_converged:
                break

        self.formulation.update_gridfunctions()

        self.drawer.redraw()
