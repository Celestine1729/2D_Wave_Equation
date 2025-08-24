# August 2025, by Celestine1729
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from wave_solver.boundary_conditions import DirichletBC
from wave_solver.initial_conditions import InitialCondition
from wave_solver.wave_solver import WaveEquationSolver2D

def create_animation(solver: WaveEquationSolver2D, T: float, interval: int = 50) -> FuncAnimation:
    """
    Create an animation of the solution
    
    Args:
        solver: Wave equation solver instance
        T: Total simulation time
        interval: Animation frame interval in milliseconds
        
    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots()
    img = ax.imshow(solver.u.T, cmap='coolwarm', animated=True, 
                   extent=[0, solver.Lx, 0, solver.Ly], 
                   origin='lower', vmin=-0.1, vmax=0.1)
    plt.colorbar(img, ax=ax)
    ax.set_title("2D Wave Equation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Calculate the correct number of frames
    n_frames = int(T / solver.dt) + 1  # +1 to include initial state
    
    # Store current time
    current_time = [0]  # Use a list to make it mutable in the closure
    
    def update(frame):
        # Only step if we're not at the first frame
        if frame > 0:
            solver.step()
            current_time[0] += solver.dt
        
        img.set_array(solver.u.T)
        ax.set_title(f"Time = {current_time[0]:.2f}")
        return [img]
    
    return FuncAnimation(fig, update, frames=n_frames, 
                        interval=interval, blit=True, repeat=False)


def main():
    
    # Create and display an animation
    ic = InitialCondition(
        displacement=lambda x, y: np.exp(-((x-1)**2 + (y-1)**2) / 0.1)
    )
    
    solver = WaveEquationSolver2D(
        c=1.0, Lx=2.0, Ly=2.0, Nx=101, Ny=101,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )
    
    ani = create_animation(solver, T=3.0)
    plt.show()

if __name__ == "__main__":
    main()
