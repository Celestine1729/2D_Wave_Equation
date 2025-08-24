# 2D Wave Equation Solver

A modular, testable implementation of a 2D wave equation solver using finite difference methods.

## Features

- Modular design with separate components for boundary conditions, initial conditions, and solver
- Support for different boundary conditions (Dirichlet, Neumann)
- Configurable domain size, resolution, and wave speed
- Built-in visualization and animation capabilities
- Comprehensive test suite

## Installation

```bash

pip install wave-solver-2d
```
## Usage
```


# Show help
wave-solver --help

# Run with default parameters
wave-solver

# Run with custom parameters
wave-solver --wave-speed 1.5 --domain-size 3.0 --resolution 151 --time 4.0

# Run without animation
wave-solver --no-animation --time 2.0

# Use different color maps
wave-solver --colormap seismic
```
## API 
```
from wave_solver import WaveEquationSolver2D, DirichletBC, InitialCondition
import numpy as np

# Create initial condition
ic = InitialCondition(
    displacement=lambda x, y: np.exp(-((x-1)**2 + (y-1)**2) / 0.1)
)

# Create solver
solver = WaveEquationSolver2D(
    c=1.0, Lx=2.0, Ly=2.0, Nx=101, Ny=101,
    boundary_condition=DirichletBC(),
    initial_condition=ic
)

# Solve for 2 seconds
solver.solve(T=2.0)

```

## Contribute and Contact

feel free to drop a PR!!    
reach me at : Celestine1729@proton.me <3



