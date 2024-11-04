import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BPoly  # Bernstein polynomial (Bezier)

class FootSwingTrajectory:
    """Utility to generate foot swing trajectories"""
    def __init__(self,
                 p0: NDArray,      # Initial position
                 pf: NDArray,      # Final position
                 height: float):   # Swing height
        """Initialize trajectory generator"""
        self.p0 = p0
        self.pf = pf
        self.height = height
        
        # Current state
        self.p = np.zeros(3)  # Position
        self.v = np.zeros(3)  # Velocity
        self.a = np.zeros(3)  # Acceleration

    def compute_swing_trajectory_bezier(self, phase: float, swing_time: float):
        """Compute foot swing trajectory with a bezier curve"""
        # XY trajectory control points
        p1 = self.p0 + (self.pf - self.p0) * 0.333
        p2 = self.p0 + (self.pf - self.p0) * 0.666
        xy_bezier = BPoly.from_derivatives([[self.p0[0:2], p1[0:2], p2[0:2], self.pf[0:2]]], [0, 1])
        
        # Split Z trajectory into up and down
        if phase < 0.5:
            z_phase = phase * 2
            z_start = self.p0[2]
            z_end = self.p0[2] + self.height
        else:
            z_phase = phase * 2 - 1
            z_start = self.p0[2] + self.height
            z_end = self.pf[2]
            
        z1 = z_start + (z_end - z_start) * 0.333
        z2 = z_start + (z_end - z_start) * 0.666
        z_bezier = BPoly.from_derivatives([[z_start, z1, z2, z_end]], [0, 1])
        
        # Evaluate position
        t = phase if phase < 0.5 else phase * 2 - 1
        self.p[0:2] = xy_bezier(phase)
        self.p[2] = z_bezier(t)
        
        # Evaluate velocity
        self.v[0:2] = xy_bezier.derivative(1)(phase) / swing_time
        self.v[2] = z_bezier.derivative(1)(t) * (2 if phase < 0.5 else 2) / swing_time
        
        # Evaluate acceleration
        self.a[0:2] = xy_bezier.derivative(2)(phase) / (swing_time * swing_time)
        self.a[2] = z_bezier.derivative(2)(t) * (4 if phase < 0.5 else 4) / (swing_time * swing_time)