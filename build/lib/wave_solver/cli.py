#!/usr/bin/env python3
"""
Command-line interface for the 2D Wave Equation Solver.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from wave_solver.boundary_conditions import DirichletBC
from wave_solver.initial_conditions import InitialCondition
from wave_solver.wave_solver import WaveEquationSolver2D

def create_animation(solver: WaveEquationSolver2D, T: float, interval: int = 50, 
                    colormap: str = "coolwarm") -> FuncAnimation:
    """
    Create an animation of the solution
    
    Args:
        solver: Wave equation solver instance
        T: Total simulation time
        interval: Animation frame interval in milliseconds
        colormap: Matplotlib colormap name
        
    Returns:
        matplotlib animation object
    """
    fig, ax = plt.subplots()
    img = ax.imshow(solver.u.T, cmap=colormap, animated=True, 
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
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Solve the 2D wave equation with various parameters"
    )
    
    # Add command line arguments
    parser.add_argument("--wave-speed", "-c", type=float, default=1.0,
                        help="Wave speed (default: 1.0)")

    parser.add_argument("--colormap", "-cm", type=str, default="coolwarm",
                    choices=['viridis', 'plasma', 'inferno', 'magma', 'coolwarm',
                             'jet', 'rainbow', 'seismic', 'ocean', 'terrain'],
                    help="Color map for visualization (default: coolwarm)")

    parser.add_argument("--domain-size", "-L", type=float, default=2.0,
                        help="Domain size (default: 2.0)")
    parser.add_argument("--resolution", "-N", type=int, default=101,
                        help="Number of grid points (default: 101)")
    parser.add_argument("--time", "-T", type=float, default=3.0,
                        help="Total simulation time (default: 3.0)")
    parser.add_argument("--pulse-center", "-p", type=float, nargs=2, 
                        default=[1.0, 1.0], metavar=('X', 'Y'),
                        help="Center of Gaussian pulse (default: 1.0 1.0)")
    parser.add_argument("--pulse-width", "-w", type=float, default=0.1,
                        help="Width of Gaussian pulse (default: 0.1)")
    parser.add_argument("--no-animation", action="store_true",
                        help="Run without animation, just show final result")
    parser.add_argument("--save-animation", type=str,
                        help="Save animation to file (e.g., animation.mp4)")
    
    args = parser.parse_args()
    
    # Create initial condition
    cx, cy = args.pulse_center
    ic = InitialCondition(
        displacement=lambda x, y: np.exp(-((x-cx)**2 + (y-cy)**2) / args.pulse_width)
    )
    
    # Create solver
    solver = WaveEquationSolver2D(
        c=args.wave_speed,
        Lx=args.domain_size, Ly=args.domain_size,
        Nx=args.resolution, Ny=args.resolution,
        boundary_condition=DirichletBC(),
        initial_condition=ic
    )

    if args.no_animation:
            # Run simulation without animation
            print(f"Running simulation for {args.time} seconds...")
            solver.solve(T=args.time)
         
            # Plot final result
            plt.figure(figsize=(8, 6))
            plt.imshow(solver.u.T, cmap=args.colormap, 
                    extent=[0, args.domain_size, 0, args.domain_size],
                    origin='lower')
            plt.colorbar()
            plt.title(f"Wave equation solution at T={args.time}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
    else:
            # Create animation
            print(f"Creating animation for {args.time} seconds...")
            ani = create_animation(solver, T=args.time, colormap=args.colormap)
        
            # Save animation if requested
    if args.save_animation:
            print(f"Saving animation to {args.save_animation}...")
            ani.save(args.save_animation, writer='ffmpeg', fps=30)
        
    plt.show()

if __name__ == "__main__":
    main()
