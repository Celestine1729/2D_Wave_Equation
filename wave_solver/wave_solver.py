import numpy as np
from typing import Callable, Optional
from .boundary_conditions import BoundaryCondition
from .initial_conditions import InitialCondition

class WaveEquationSolver2D:
    """Solver for the 2D wave equation"""
    
    def __init__(self, 
                 c: float, 
                 Lx: float, Ly: float, 
                 Nx: int, Ny: int,
                 boundary_condition: BoundaryCondition,
                 initial_condition: InitialCondition):
        """
        Initialize the wave equation solver
        
        Args:
            c: Wave speed
            Lx, Ly: Domain dimensions
            Nx, Ny: Number of grid points
            boundary_condition: Boundary condition object
            initial_condition: Initial condition object
        """
        self.c = c
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.bc = boundary_condition
        self.ic = initial_condition
        
        # Grid spacing
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        
        # Grid points
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Time step (CFL condition)
        self.dt = 0.9 * (1/c) * 1/np.sqrt(1/self.dx**2 + 1/self.dy**2)
        
        # Solution arrays
        self.u = np.zeros((Nx, Ny))      # Current time step
        self.u_prev = np.zeros((Nx, Ny)) # Previous time step
        self.u_next = np.zeros((Nx, Ny)) # Next time step
        
        # Precompute constants
        self.Cx2 = (c * self.dt / self.dx)**2
        self.Cy2 = (c * self.dt / self.dy)**2
        
        # Initialize solution
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the solution with initial conditions"""
        # Set initial displacement
        self.u = self.ic.displacement(self.X, self.Y)
        self.u_prev = self.u.copy()
        
        # Apply initial velocity if provided
        if self.ic.velocity:
            velocity_field = self.ic.velocity(self.X, self.Y)
            self.u_prev = self.u - self.dt * velocity_field
        
        # Apply boundary conditions
        self.bc.apply(self.u)
        self.bc.apply(self.u_prev)
    
    def step(self) -> None:
        """Advance the solution by one time step"""
        # Update interior points using finite difference scheme
        self.u_next[1:-1, 1:-1] = (
            2*self.u[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
            self.Cx2 * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1]) +
            self.Cy2 * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2])
        )
        
        # Apply boundary conditions
        self.bc.apply(self.u_next)
        
        # Update solution arrays
        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()
    
    def solve(self, T: float, callback: Optional[Callable] = None) -> None:
        """
        Solve the wave equation for time T
        
        Args:
            T: Total simulation time
            callback: Optional function to call at each time step
        """
        n_steps = int(T / self.dt)
        
        for step in range(n_steps):
            self.step()
            if callback:
                callback(self.u, step * self.dt)
