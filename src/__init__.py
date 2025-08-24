"""
Wave Equation Solver Package

A modular implementation for solving the 2D wave equation
using finite difference methods.
"""

from .boundary_conditions import BoundaryCondition, DirichletBC, NeumannBC
from .initial_conditions import InitialCondition
from .wave_solver import WaveEquationSolver2D


__version__ = "0.1.0"
__all__ = [
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "InitialCondition",
    "WaveEquationSolver2D",
    "test_gaussian_pulse",
    "test_standing_wave",
    "test_energy_conservation"
]
