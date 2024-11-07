import math
import numpy as np

class GaitScheduler:
    """
        Gait scheduler to generate swing and stance phases for each leg
    """
    def __init__(self, total_period: float, gait_type: str):
        self.n_legs = 4
        self.total_period = total_period # Total time for one gait cycle

        # Define gaits (offset percents, duty factor)
        gaits = {
            "trot": (np.array([0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5])),
            "walk": (np.array([0, 0.25, 0.25, 0.75]), np.array([0.75, 0.75, 0,75, 0.75])),
            "bound": (np.array([0, 0, 0.5, 0.5]), np.array([0.5, 0.5, 0.5, 0.5])),
            "stand": (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))
        }

        self.offset, self.duty = gaits[gait_type]
        self.phase = np.zeros(4)
        self.time = 0.0
    
    def update(self, dt: float):
        """
            Updates phases for each leg

            Params:
                dt: MPC update rate
        """
        self.time += dt
        total_phase = (self.time / self.total_period) % 1.0

        # Calculate phase for each leg based on offset
        self.phase = (total_phase - self.offset) % 1.0

    def get_current_contact_state(self) -> np.ndarray:
        """
            Get which legs are in stance (1) or swing (0)

            Returns:
                in_stance: np array of which feet are in stance at a given time (i.e. [1, 0, 1, 0])
        """
        in_stance = np.zeros(4)
        for leg in range(self.n_legs):
            if self.phase[leg] < self.duty[leg]:
                in_stance[leg] = 1
        return in_stance
    
    def predict_horizon_contact_state(self, dt: float, horizon_length: int) -> np.ndarray:
        """
            Predict future contact states for MPC Horizon

            Params:
                dt: MPC update rate
                horizon_length: MPC Time horizon
            
            Returns:
                horizon_states: np array of which feet are in stance throughout the entire horizon
        """ 
        horizon_states = np.zeros((horizon_length, self.n_legs))

        for k in range(horizon_length):
            # Simulate phase forward
            future_time = self.time + k * dt
            future_phase = (future_time / self.total_period) % 1.0
            future_leg_phases = (future_phase - self.offset) % 1.0

            # Check if legs are in stance
            for leg in range(self.n_legs):
                if future_leg_phases[leg] < self.duty[leg]:
                    horizon_states[k, leg] = 1

        return horizon_states 
    
    def get_stance_duration(self) -> float:
        """
            Get the stance duration for Raibert Heuristic

            Returns:
                stance duration (assuming same duty factor for all legs)        
        """
        return self.total_period * self.duty[0]

