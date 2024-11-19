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
            "trot": (np.array([0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5])), # FL, FR, RL, RR
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
                horizon_states: [4 x 10] which feet are in stance throughout the entire horizon
        """ 
        horizon_states = np.zeros((self.n_legs, horizon_length))

        # For each horizon step 
        for k in range(horizon_length):
            # Simulate phase forward
            future_time = self.time + k * dt
            
            # Get total phase progression 
            future_total_phase = (future_time / self.total_period) % 1.0

            # For each leg
            for leg in range(self.n_legs):
                # Get leg's phase and offset
                leg_phase = (future_total_phase - self.offset[leg]) % 1.0
                
                #Check if stance (phase < duty factor)
                if leg_phase < self.duty[leg]:
                    horizon_states[leg, k] = 1
                else:
                    horizon_states[leg, k] = 0

        return horizon_states 
    
    def get_stance_duration(self) -> float:
        """
            Get the stance duration for Raibert Heuristic

            Returns:
                stance duration (assuming same duty factor for all legs)        
        """
        return self.total_period * self.duty[0]
    
    def get_swing_phase(self, leg: int) -> float:
        """Get normalized phase (0-1) for leg in swing
        
        Args:
            leg: Index of leg
            
        Returns:
            Normalized phase through swing (0-1) or 0 if in stance
        """
        if self.phase[leg] < self.duty[leg]:  # If in stance
            return 0.0
            
        # Normalize swing phase to 0-1
        swing_phase = (self.phase[leg] - self.duty[leg]) / (1.0 - self.duty[leg])
        return swing_phase

    def get_swing_duration(self) -> float:
        """Get duration of swing phase"""
        return self.total_period * (1.0 - self.duty[0])
        
    def is_entering_swing(self, leg: int) -> bool:
        """Check if leg is just entering swing phase"""
        phase = self.phase[leg]
        return abs(phase - self.duty[leg]) < 1e-4  # Small threshold
        
    def is_entering_stance(self, leg: int) -> bool:
        """Check if leg is just entering stance phase"""
        phase = self.phase[leg]
        return phase < 1e-4  # Small threshold
        
    def time_until_next_swing(self, leg: int) -> float:
        """Get time until leg enters swing phase"""
        if self.phase[leg] < self.duty[leg]:
            return (self.duty[leg] - self.phase[leg]) * self.total_period
        else:
            return (1.0 - self.phase[leg] + self.duty[leg]) * self.total_period

    def time_since_swing_start(self, leg: int) -> float:
        """Get time since leg started swing phase"""
        if self.phase[leg] < self.duty[leg]:  # In stance
            return 0.0
        return (self.phase[leg] - self.duty[leg]) * self.total_period

