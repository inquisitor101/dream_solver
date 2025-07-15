""" Definitions of conservative multizone spatial discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.time import TimeSchemes, TransientRoutine
from dream.config import Configuration, dream_configuration, Integrals
from dream.mesh import Periodic, Initial
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField,
                                       Outflow,
                                       InviscidWall,
                                       Symmetry,
                                       IsothermalWall,
                                       AdiabaticWall)

from .time import IMEX_EULER

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from ..solver import CompressibleFlowSolver


class ConservativeDomainSplitSDG_HDG(ConservativeFiniteElementMethod):
    name: str = "conservative_domainsplit_sdg_hdg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'static_condensation': True,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> IMEX_EULER:
        """ Time scheme for the SDG-HDG method depending on the choosen time routine.

            :getter: Returns the time scheme
            :setter: Sets the time scheme
        """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:
        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [IMEX_EULER]
        else:
            raise TypeError("SDG-HDG method only supports transient!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order, dgjumps=True)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)
        Uhat = ngs.Compress(Uhat, Uhat.GetDofs(self.mesh.Materials('hdg')))

        # NOTE, we might run into problems where periodicity is only on the SSDG domain, not the HDG.
        #if self.root.bcs.has_condition(Periodic):
        #    Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim


    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        super().add_symbolic_spatial_forms(blf, lf)

    def add_convection_form(self, blf: Integrals, lf: Integrals):

        self.add_sdg_convection_form(blf, lf)
        self.add_hdg_convection_form(blf, lf)
         
    def add_sdg_convection_form(self, blf: Integrals, lf: Integrals):

        # NOTE, bonus must account for nonlinearities..
        bonus = self.bonus_int_order['convection']
        dV = ngs.dx(definedon=self.mesh.Materials('sdg'), bonus_intorder=bonus['vol'])
        dX = ngs.dx(element_boundary=True, definedon=self.mesh.Materials('sdg'), bonus_intorder=bonus['bnd'])

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']

        Ui = self.get_conservative_fields(U)
        Uj = self.get_conservative_fields(U.Other())

        F = self.root.get_convective_flux(Ui)
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)

        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * dV 
        blf['U']['convection'] += mask*bla.inner(Fn, V) * dX 

    def add_hdg_convection_form(self, blf: Integrals, lf: Integrals):
        
        bonus = self.bonus_int_order['convection']
        dV = ngs.dx(definedon=self.mesh.Materials('hdg'), bonus_intorder=bonus['vol'])
        dX = ngs.dx(element_boundary=True, definedon=self.mesh.Materials('hdg'), bonus_intorder=bonus['bnd'])

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        F = self.root.get_convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * dV 
        blf['U']['convection'] += bla.inner(Fn, V) * dX

        if self.root.dynamic_viscosity.is_inviscid:
            tau_cs = self.root.riemann_solver.get_simplified_convective_stabilisation_matrix_hdg(Uhat, self.mesh.normal)
            blf['Uhat']['convection'] = -mask * (tau_cs*U.U - Uhat.U) * Vhat * dX
        else:
            blf['Uhat']['convection'] = -mask * bla.inner(Fn, Vhat) * dX

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Not implemented yet.")

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, (InviscidWall, Symmetry)):
                self.add_inviscid_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, IsothermalWall):
                self.add_isothermal_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue

            elif isinstance(bc, AdiabaticWall):
                self.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        raise NotImplementedError("Not implemented yet.")

    def add_outflow_formulation(self, blf: Integrals, lf: Integrals, bc: Outflow, bnd: str):
        raise NotImplementedError("Not implemented yet.")

    def add_inviscid_wall_formulation(self, blf: Integrals, lf: Integrals, bc: InviscidWall, bnd: str):
        raise NotImplementedError("Not implemented yet.")

    def add_isothermal_wall_formulation(self, blf: Integrals, lf: Integrals, bc: IsothermalWall, bnd: str):
        raise NotImplementedError("Not implemented yet.")

    def add_adiabatic_wall_formulation(self, blf: Integrals, lf: Integrals, bc: AdiabaticWall, bnd: str):
        raise NotImplementedError("Not implemented yet.")

    def get_convective_numerical_flux(self, U: flowfields, Uhat: flowfields, unit_vector: ngs.CF):
       
        unit_vector = bla.as_vector(unit_vector)

        tau_c = self.root.riemann_solver.get_convective_stabilisation_matrix_hdg(Uhat, unit_vector)

        return self.root.get_convective_flux(Uhat) * unit_vector + tau_c * (U.U - Uhat.U)

    def get_diffusive_numerical_flux(
            self, U: flowfields, Uhat: flowfields, Q: flowfields, unit_vector: ngs.CF):
        raise NotImplementedError("Not implemented yet.")

    def get_solution_fields(self) -> flowfields:
        U = super().get_solution_fields()
        return U

    def set_initial_conditions(self):

        U = self.mesh.MaterialCF({dom: ngs.CF(
            (self.root.density(dc.fields),
                self.root.momentum(dc.fields),
                self.root.energy(dc.fields))) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        gfu = self.gfus['Uhat']
        fes = self.gfus['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec

        super().set_initial_conditions()



