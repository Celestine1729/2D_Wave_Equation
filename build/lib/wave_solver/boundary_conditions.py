from abc import ABC, abstractmethod
import numpy as np

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions"""
    
    @abstractmethod
    def apply(self, u: np.ndarray) -> None:
        """Apply boundary condition to the solution array"""
        pass


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition (fixed values)"""
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, u: np.ndarray) -> None:
        u[0, :] = self.value    # Top boundary
        u[-1, :] = self.value   # Bottom boundary
        u[:, 0] = self.value    # Left boundary
        u[:, -1] = self.value   # Right boundary


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition (zero gradient)"""
    
    def apply(self, u: np.ndarray) -> None:
        u[0, :] = u[1, :]      # Top boundary
        u[-1, :] = u[-2, :]    # Bottom boundary
        u[:, 0] = u[:, 1]      # Left boundary
        u[:, -1] = u[:, -2]    # Right boundary
