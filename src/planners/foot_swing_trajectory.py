import numpy as np
#import matplotlib.pyplot as plt
from numpy.typing import NDArray

class FootSwingTrajectory:
    """Utility to generate foot swing trajectories"""
    def __init__(self,
                 p0: NDArray,      # Initial position
                 pf: NDArray,      # Final position
                 height: float):   # Swing height
        """Initialize trajectory generator
        
        Args:
            p0: Initial foot position
            pf: Final foot position
            height: Maximum swing height
        """
        self.p0 = p0
        self.pf = pf
        self.height = height
        
        # Current state
        self.p = np.zeros(3)  # Position
        self.v = np.zeros(3)  # Velocity
        self.a = np.zeros(3)  # Acceleration

    def compute_swing_trajectory_bezier(self, phase: float, swing_time: float):
        """Compute foot swing trajectory with a bezier curve
        
        Args:
            phase: How far along we are in the swing (0 to 1)
            swing_time: How long the swing should take (seconds)
        """
        # XY trajectory
        self.p = self._cubic_bezier(self.p0, self.pf, phase)
        self.v = self._cubic_bezier_first_derivative(self.p0, self.pf, phase) / swing_time
        self.a = self._cubic_bezier_second_derivative(self.p0, self.pf, phase) / (swing_time * swing_time)
        
        # Z trajectory (split into two phases)
        if phase < 0.5:
            # First half: go up
            z_phase = phase * 2
            z_start = self.p0[2]
            z_end = self.p0[2] + self.height
            
            zp = self._cubic_bezier_1d(z_start, z_end, z_phase)
            zv = self._cubic_bezier_first_derivative_1d(z_start, z_end, z_phase) * 2 / swing_time
            za = self._cubic_bezier_second_derivative_1d(z_start, z_end, z_phase) * 4 / (swing_time * swing_time)
        else:
            # Second half: go down
            z_phase = phase * 2 - 1
            z_start = self.p0[2] + self.height
            z_end = self.pf[2]
            
            zp = self._cubic_bezier_1d(z_start, z_end, z_phase)
            zv = self._cubic_bezier_first_derivative_1d(z_start, z_end, z_phase) * 2 / swing_time
            za = self._cubic_bezier_second_derivative_1d(z_start, z_end, z_phase) * 4 / (swing_time * swing_time)
        
        # Update Z components
        self.p[2] = zp
        self.v[2] = zv
        self.a[2] = za
        
    def _cubic_bezier(self, p0: NDArray, pf: NDArray, t: float) -> NDArray:
        """Cubic Bezier curve interpolation"""
        p1 = p0 + (pf - p0) * 0.333
        p2 = p0 + (pf - p0) * 0.666
        return (
            p0 * (1-t)**3 +
            p1 * 3*(1-t)**2*t +
            p2 * 3*(1-t)*t**2 +
            pf * t**3
        )
    
    def _cubic_bezier_first_derivative(self, p0: NDArray, pf: NDArray, t: float) -> NDArray:
        """First derivative of cubic Bezier curve"""
        p1 = p0 + (pf - p0) * 0.333
        p2 = p0 + (pf - p0) * 0.666
        return (
            3 * (p1 - p0) * (1-t)**2 +
            6 * (p2 - p1) * (1-t)*t +
            3 * (pf - p2) * t**2
        )
    
    def _cubic_bezier_second_derivative(self, p0: NDArray, pf: NDArray, t: float) -> NDArray:
        """Second derivative of cubic Bezier curve"""
        p1 = p0 + (pf - p0) * 0.333
        p2 = p0 + (pf - p0) * 0.666
        return (
            6 * (p2 - 2*p1 + p0) * (1-t) +
            6 * (pf - 2*p2 + p1) * t
        )
    
    def _cubic_bezier_1d(self, x0: float, xf: float, t: float) -> float:
        """1D cubic Bezier curve interpolation"""
        x1 = x0 + (xf - x0) * 0.333
        x2 = x0 + (xf - x0) * 0.666
        return (
            x0 * (1-t)**3 +
            x1 * 3*(1-t)**2*t +
            x2 * 3*(1-t)*t**2 +
            xf * t**3
        )
    
    def _cubic_bezier_first_derivative_1d(self, x0: float, xf: float, t: float) -> float:
        """First derivative of 1D cubic Bezier curve"""
        x1 = x0 + (xf - x0) * 0.333
        x2 = x0 + (xf - x0) * 0.666
        return (
            3 * (x1 - x0) * (1-t)**2 +
            6 * (x2 - x1) * (1-t)*t +
            3 * (xf - x2) * t**2
        )
    
    def _cubic_bezier_second_derivative_1d(self, x0: float, xf: float, t: float) -> float:
        """Second derivative of 1D cubic Bezier curve"""
        x1 = x0 + (xf - x0) * 0.333
        x2 = x0 + (xf - x0) * 0.666
        return (
            6 * (x2 - 2*x1 + x0) * (1-t) +
            6 * (xf - 2*x2 + x1) * t
        )

# Example usage
if __name__ == "__main__":
    # Create trajectory
    p0 = np.array([0., 0., 0.])
    pf = np.array([0.3, 0., 0.])
    height = 0.1
    
    traj = FootSwingTrajectory(p0, pf, height)
    
    # Test trajectory
 
    
    phases = np.linspace(0, 1, 100)
    positions = []
    velocities = []
    accelerations = []
    
    for phase in phases:
        traj.compute_swing_trajectory_bezier(phase, swing_time=0.5)
        positions.append(traj.p.copy())
        velocities.append(traj.v.copy())
        accelerations.append(traj.a.copy())
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # plt.figure(figsize=(12, 4))
    # plt.subplot(131)
    # plt.plot(phases, positions)
    # plt.title('Position')
    # plt.legend(['x', 'y', 'z'])
    
    # plt.subplot(132)
    # plt.plot(phases, velocities)
    # plt.title('Velocity')
    
    # plt.subplot(133)
    # plt.plot(phases, accelerations)
    # plt.title('Acceleration')
    
    # plt.tight_layout()
    # plt.show()