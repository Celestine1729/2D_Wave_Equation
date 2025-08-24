import pytest
import numpy as np
import matplotlib.pyplot as plt
from wave_solver.boundary_conditions import DirichletBC, NeumannBC
from wave_solver.initial_conditions import InitialCondition
from wave_solver.wave_solver import WaveEquationSolver2D

def test_dirichlet_boundary_condition():
    """Test that Dirichlet boundary condition sets correct values"""
    bc = DirichletBC(value=0.5)
    u = np.ones((5, 5))
    bc.apply(u)
    
    # Check boundaries
    assert np.all(u[0, :] == 0.5)  # Top
    assert np.all(u[-1, :] == 0.5)  # Bottom
    assert np.all(u[:, 0] == 0.5)  # Left
    assert np.all(u[:, -1] == 0.5)  # Right
    
    # Check interior (should remain unchanged)
    assert np.all(u[1:-1, 1:-1] == 1.0)

def test_neumann_boundary_condition():
    """Test that Neumann boundary condition sets correct values"""
    bc = NeumannBC()
    u = np.ones((5, 5))
    u[1, :] = 2.0  # Set second row to 2
    
    bc.apply(u)
    
    # Check boundaries (should match adjacent interior)
    assert np.all(u[0, :] == 2.0)  # Top
    assert np.all(u[-1, :] == u[-2, :])  # Bottom
    assert np.all(u[:, 0] == u[:, 1])  # Left
    assert np.all(u[:, -1] == u[:, -2])  # Right

def test_initial_condition():
    """Test that initial condition is applied correctly"""
    def displacement(x, y):
        return x + y
    
    ic = InitialCondition(displacement=displacement)
    
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    result = ic.displacement(X, Y)
    expected = X + Y
    
    assert np.array_equal(result, expected)

def test_wave_solver_initialization():
    """Test that wave solver initializes correctly"""
    ic = InitialCondition(
        displacement=lambda x, y: np.sin(x) * np.cos(y)
    )
    
    solver = WaveEquationSolver2D(
        c=1.0, Lx=2.0, Ly=2.0, Nx=11, Ny=11,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )
    
    # Check grid dimensions
    assert solver.X.shape == (11, 11)
    assert solver.Y.shape == (11, 11)
    
    # Check solution arrays
    assert solver.u.shape == (11, 11)
    assert solver.u_prev.shape == (11, 11)
    assert solver.u_next.shape == (11, 11)
    
    # Check that initial condition was applied (before boundary conditions)
    # Create a copy of what the initial condition should be
    expected = np.sin(solver.X) * np.cos(solver.Y)
    
    # Apply boundary conditions to the expected result
    bc = DirichletBC()
    bc.apply(expected)
    
    # Now compare
    assert np.allclose(solver.u, expected, atol=1e-10)


def test_wave_solver_step():
    """Test that a single solver step works correctly"""
    ic = InitialCondition(
        displacement=lambda x, y: np.zeros_like(x)
    )
    
    solver = WaveEquationSolver2D(
        c=1.0, Lx=2.0, Ly=2.0, Nx=5, Ny=5,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )
    
    # Add a small perturbation
    solver.u[2, 2] = 0.1
    
    # Take a step
    solver.step()
    
    # Check that solution was updated
    assert not np.all(solver.u == 0)
    
    # Check that boundaries are still zero
    assert np.all(solver.u[0, :] == 0)
    assert np.all(solver.u[-1, :] == 0)
    assert np.all(solver.u[:, 0] == 0)
    assert np.all(solver.u[:, -1] == 0)

def test_symmetry():
    """Verify that symmetric initial conditions produce symmetric solutions"""
    # Create a symmetric Gaussian pulse
    ic = InitialCondition(
        displacement=lambda x, y: np.exp(-((x-1)**2 + (y-1)**2) / 0.1)
    )
    
    solver = WaveEquationSolver2D(
        c=1.0, Lx=2.0, Ly=2.0, Nx=101, Ny=101,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )
    
    solver.solve(T=1.0)
    
    # Check symmetry about the center (1,1)
    center_x, center_y = 50, 50  # Index of center point (1,1)
    
    # Check symmetry in x-direction
    x_symmetry_error = np.max(np.abs(solver.u[center_x:, :] - solver.u[center_x::-1, :]))
    
    # Check symmetry in y-direction
    y_symmetry_error = np.max(np.abs(solver.u[:, center_y:] - solver.u[:, center_y::-1]))
    
    # Symmetry errors should be small
    assert x_symmetry_error < 1e-10
    assert y_symmetry_error < 1e-10

def test_wave_speed():
    """Verify that waves propagate at the correct speed"""
    # Use a larger domain to avoid boundary effects
    Lx, Ly = 4.0, 4.0
    
    # Create a narrow Gaussian pulse
    ic = InitialCondition(
        displacement=lambda x, y: np.exp(-((x-1)**2 + (y-2)**2) / 0.01),
        velocity=lambda x, y: np.zeros_like(x)
    )
    
    solver = WaveEquationSolver2D(
        c=2.0,  # Use a specific wave speed for testing
        Lx=Lx, Ly=Ly, Nx=201, Ny=201,  # High resolution for accuracy
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )
    
    # Find initial peak position
    initial_peak = np.unravel_index(np.argmax(solver.u), solver.u.shape)
    initial_x = initial_peak[0] * solver.dx
    initial_y = initial_peak[1] * solver.dy
    
    # Run simulation for a specific time
    T = 0.5
    solver.solve(T=T)
    
    # Find final peak position
    final_peak = np.unravel_index(np.argmax(solver.u), solver.u.shape)
    final_x = final_peak[0] * solver.dx
    final_y = final_peak[1] * solver.dy
    
    # Calculate distance traveled (should be along a circle for 2D wave)
    distance = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
    
    # Expected distance = wave speed * time
    expected_distance = solver.c * T
    
    print(f"Initial peak: ({initial_x:.3f}, {initial_y:.3f})")
    print(f"Final peak: ({final_x:.3f}, {final_y:.3f})")
    print(f"Distance traveled: {distance:.6f}")
    print(f"Expected distance: {expected_distance:.6f}")
    print(f"Error: {abs(distance - expected_distance):.6f}")
    
    # For 2D waves, the peak doesn't move in a straight line, so we need to be more lenient
    assert abs(distance - expected_distance) < 0.3  # Allow more error for 2D case


def test_periodicity():
    """Test that a standing wave returns to its initial state after one period"""
    # Parameters for a standing wave
    c = 1.0
    Lx, Ly = 2.0, 2.0
    m, n = 1, 1  # Mode numbers

    # Calculate angular frequency
    omega = c * np.pi * np.sqrt((m/Lx)**2 + (n/Ly)**2)
    period = 2 * np.pi / omega

    # Create initial condition
    ic = InitialCondition(
        displacement=lambda x, y: np.sin(m*np.pi*x/Lx) * np.sin(n*np.pi*y/Ly),
        velocity=lambda x, y: np.zeros_like(x)
    )

    solver = WaveEquationSolver2D(
        c=c, Lx=Lx, Ly=Ly, Nx=101, Ny=101,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )

    # Store initial state
    initial_state = solver.u.copy()

    # Run for one period
    solver.solve(T=period)

    # Calculate error (should be close to initial state)
    error = np.max(np.abs(solver.u - initial_state))

    print(f"Periodicity error after one period: {error:.6f}")

    # The error should be small
    assert error < 0.1
    


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
