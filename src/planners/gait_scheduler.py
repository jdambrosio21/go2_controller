from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional

class GaitType(Enum):
    """Enumerated gait types matching C++ implementation"""
    STAND = 0
    STAND_CYCLE = 1
    STATIC_WALK = 2
    AMBLE = 3
    TROT_WALK = 4
    TROT = 5
    TROT_RUN = 6
    PACE = 7
    BOUND = 8
    ROTARY_GALLOP = 9
    TRAVERSE_GALLOP = 10
    PRONK = 11
    THREE_FOOT = 12
    CUSTOM = 13
    TRANSITION_TO_STAND = 14

# Define Gaits
@dataclass
class GaitData:
    """Storage for gait parameters and state, matching C++ GaitData"""
    # Current gait state
    _current_gait: GaitType = GaitType.STAND
    _next_gait: GaitType = GaitType.STAND
    gait_name: str = "STAND"

    # Gait descriptors
    period_time_nominal: float = 0.0      # Overall period time to scale
    initial_phase: float = 0.0           # Initial phase to offset
    switching_phase_nominal: float = 0.0  # Nominal phase to switch contacts
    overrideable: int = 0                # If gait params can be overridden

    # Enable flag for each foot
    gait_enabled: np.ndarray = np.zeros(4, dtype=np.int32)

    # Time based descriptors (one per foot)
    period_time: np.ndarray = np.zeros(4) # overall gait period scale
    time_stance: np.ndarray = np.zeros(4) # total stance time
    time_swing = np.ndarray = np.zeros(4) # total swing time
    time_stance_remaining: np.ndarray = np.zeros(4) # stance time remaining
    time_swing_remaining: np.ndarray = np.zeros(4) # swing time remaining

    # Phase based descriptors (one per foot)
    switching_phase: np.ndarray = np.zeros(4) # Phase to swtich to swing
    phase_varible: np.ndarray = np.zeros(4) # Current gait phase for each foot
    phase_offset: np.ndarray = np.zeros(4) # Nominal gait phase offsets
    phase_scale: np.ndarray = np.zeros(4) # phase scale relative to variable
    phase_stance: np.ndarray = np.zeros(4)  # stance subphase
    phase_swing: np.ndarray = np.zeros(4) # swing subphase

    # Scheduled contact states (one per foot)
    contact_state_scheduled: np.ndarray = np.zeros(4, dtype=np.int32) # contact state of the foot
    contact_state_prev: np.ndarray = np.zeros(4, dtype=np.int32) # previous contact state of the foot
    touchdown_scheduled: np.ndarray = np.zeros(4, dtype=np.int32) # scheduled touchdown flag
    liftoff_scheduled: np.ndarray = np.zeros(4, dtype=np.int32) # scheduled liftoff flag

    def zero(self):
        """Reset gait data to zeros (match C++ implementation)"""
        self._next_gait = self._current_gait

        # Reset all gait descriptors
        self.period_time_nominal = 0.0
        self.initial_phase = 0.0
        self.switching_phase_nominal = 0.0
        self.overrideable = 0.0

        # Reset all arrays to zeros
        self.gait_enabled.fill(0)
        self.period_time.fill(0.0)
        self.time_stance.fill(0.0)
        self.time_swing.fill(0.0)
        self.time_stance_remaining.fill(0.0)
        self.time_swing_remaining.fill(0.0)
        self.switching_phase.fill(0.0)
        self.phase_variable.fill(0.0)
        self.phase_offset.fill(0.0)
        self.phase_scale.fill(0.0)
        self.phase_stance.fill(0.0)
        self.phase_swing.fill(0.0)
        self.contact_state_scheduled.fill(0)
        self.contact_state_prev.fill(0)
        self.touchdown_scheduled.fill(0)
        self.liftoff_scheduled.fill(0)

class GaitScheduler:
    def __init__(self, dt: float):
        """Initialize the gait scheduler

        Args:
            dt: Timestep duration
        """
        self.dt = dt
        self.gait_data = GaitData()

        # Natural gait modifiers
        self.period_time_natural = 0.5
        self.switching_phase_natural = 0.5
        self.swing_time_natural = 0.25

        #Init
        self.initialize()
    
    def initialize(self):
        """Initalize the gait scheduler"""
        print("[GAIT] Initialize Gait Scheduler")

        # Start gait in trot since we use this the most
        self.gait_data._current_gait = GaitType.STAND

        # Zero all gait data
        self.gait_data.zero()

        # Create initial gait from the nominal initial
        self.create_gait()
        self.period_time_natural = self.gait_data.period_time_nominal
        self.switching_phase_natural = self.gait_data.switching_phase_nominal

    def step(self):
        """Executes the gait schedule step to calculate values for the defining"""

        # Modify the gait with settings
        self.modify_gait()

        if self.gait_data._current_gait != GaitType.STAND:
            # Track the reference phase variable
            self.gait_data.initial_phase = (
                self.gait_data.initial_phase + (self.dt / self.gait_data.period_time_nominal)
            ) % 1.0

        # Iterate over the feet
        for foot in range(4):
            # Store the previous contact state for the next timestep
            self.gait_data.contact_state_prev[foot] = self.gait_data.contact_state_scheduled[foot]

            if self.gait_data.gait_enabled[foot] == 1:
                # Compute monotonic time based phase increment
                if self.gait_data._current_gait == GaitType.STAND:
                    # Don't increment the phase when in stand mode
                    dphase = 0.0
                else:
                    dphase = GaitData.phase_scale[foot] * (self.dt / GaitData.period_time_nominal)

                # Find each foots current phase
                self.gait_data.phase_varible[foot] = (self.gait_data.phase_varible[foot] + dphase) % 1.0

                # Check the current contact state
                if self.gait_data.phase_varible[foot] <= self.gait_data.switching_phase[foot]:
                    # Foot is scheduled to be in stance
                    self.gait_data.contact_state_scheduled[foot] = 1

                    # Calculate stance subphase
                    self.gait_data.phase_stance[foot] = (
                        self.gait_data.phase_varible[foot] /self.gait_data.switching_phase[foot]
                    )

                    # Swing phase has not started since foot is in stance
                    self.gait_data.phase_swing[foot] = 0.0

                    # Calculate remaining time in stance
                    self.gait_data.time_stance_remaining[foot] = (
                        self.gait_data.period_time[foot] * 
                        (self.gait_data.switching_phase[foot] - self.gait_data.phase_varible[foot])
                    )

                    # Foot is in stance, no swing time remaining
                    self.gait_data.time_swing_remaining[foot] = 0.0

                    # Check for touchdown
                    # First contact signals scheduled touchdown
                    if self.gait_data.contact_state_prev[foot] == 0:
                        # Set touchdown flag to 1
                        self.gait_data.touchdown_scheduled[foot] = 1
                    else:
                        # Set touchdown flag to 0
                        self.gait_data.touchdown_scheduled[foot] = 0
                else:
                    # Foot is not scheduled to be in contact (in swing)
                    self.gait_data.contact_state_scheduled[foot] = 0

                    # Stance phase has been completed since foot is in swing
                    self.gait_data.phase_stance[foot] = 1.0

                    # Calculate swing subphase
                    self.gait_data.phase_swing[foot] = (
                        (self.gait_data.phase_variable[foot] - self.gait_data.switching_phase[foot]) /
                        (1.0 - self.gait_data.switching_phase[foot])
                        )
                    
                    # Foot is in swing, no stance time remaining
                    self.gait_data.time_stance_remaining[foot] = 0.0

                    # Calculate remaining time in swing
                    self.gait_data.time_swing_remaining[foot] = (
                        self.gait_data.period_time[foot] * (1.0 - self.gait_data.phase_varible[foot])
                    )

                    # First contact signifies scheduled touchdown
                    if self.gait_data.contact_state_prev[foot] == 1:
                        # Set liftoff




        
# Track Phases

# Implement Gait Transition Logic

#

# Output phases status and swing-stance timing for each leg