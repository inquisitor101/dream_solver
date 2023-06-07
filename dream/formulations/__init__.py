from __future__ import annotations
from .interface import CompressibleFormulations, MixedMethods, Formulation, RiemannSolver
from .conservative2d import ConservativeFormulation2D
from .primitive2d import PrimitiveFormulation2D


def formulation_factory(mesh, solver_configuration) -> Formulation:

    if mesh.dim == 2:

        if solver_configuration.formulation is CompressibleFormulations.CONSERVATIVE:
            return ConservativeFormulation2D(mesh, solver_configuration)
        elif solver_configuration.formulation is CompressibleFormulations.PRIMITIVE:
            return PrimitiveFormulation2D(mesh, solver_configuration)