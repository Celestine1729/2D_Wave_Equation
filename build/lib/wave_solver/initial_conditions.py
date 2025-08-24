from typing import Callable, Optional
import numpy as np

class InitialCondition:
    """Container for initial conditions"""
    
    def __init__(self, 
                 displacement: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 velocity: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None):
        """
        Initialize with displacement and optional velocity functions
        
        Args:
            displacement: Function f(x, y) that returns initial displacement
            velocity: Function g(x, y) that returns initial velocity (defaults to zero)
        """
        self.displacement = displacement
        self.velocity = velocity if velocity else lambda x, y: np.zeros_like(x)
