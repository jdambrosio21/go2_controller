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
    phase_variable: np.ndarray = np.zeros(4) # Current gait phase for each foot
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
                self.gait_data.phase_variable[foot] = (self.gait_data.phase_variable[foot] + dphase) % 1.0

                # Check the current contact state
                if self.gait_data.phase_variable[foot] <= self.gait_data.switching_phase[foot]:
                    # Foot is scheduled to be in stance
                    self.gait_data.contact_state_scheduled[foot] = 1

                    # Calculate stance subphase
                    if self.gait_data.switching_phase[foot] != 0:
                        self.gait_data.phase_stance[foot] = self.gait_data.phase_variable[foot] / self.gait_data.switching_phase[foot]
                    else:
                        self.gait_data.phase_stance[foot] = 0.0

                    # Swing phase has not started since foot is in stance
                    self.gait_data.phase_swing[foot] = 0.0

                    # Calculate remaining time in stance
                    self.gait_data.time_stance_remaining[foot] = (
                        self.gait_data.period_time[foot] * 
                        (self.gait_data.switching_phase[foot] - self.gait_data.phase_variable[foot])
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
                        self.gait_data.period_time[foot] * (1.0 - self.gait_data.phase_variable[foot])
                    )

                    # First contact signifies scheduled touchdown
                    if self.gait_data.contact_state_prev[foot] == 1:
                        # Set liftoff flag to 1
                        self.gait_data.liftoff_scheduled[foot] = 1
                    else:
                        # Set liftoff flag to 0
                        self.gait_data.liftoff_scheduled[foot] = 0
            else:
                # Leg is disabled
                self.gait_data.phase_variable[foot] = 0.0
                self.gait_data.contact_state_scheduled[foot] = 0


    def create_gait(self):
        """Create new gait from parameters, matching C++ implementation"""
        print(f"[GAIT] Transitioning gait from {self.gait_data.gait_name} to ", end="")
        
        # Get parameters for next gait
        if self.gait_data._next_gait == GaitType.STAND:
            self.gait_data.gait_name = "STAND"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 10.0
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 1.0
            self.gait_data.phase_offset = np.array([0.5, 0.5, 0.5, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 0
            
        elif self.gait_data._next_gait == GaitType.STAND_CYCLE:
            self.gait_data.gait_name = "STAND_CYCLE"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 1.0
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 1.0
            self.gait_data.phase_offset = np.array([0.5, 0.5, 0.5, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 0
            
        elif self.gait_data._next_gait == GaitType.STATIC_WALK:
            self.gait_data.gait_name = "STATIC_WALK"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 1.25
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.8
            self.gait_data.phase_offset = np.array([0.25, 0.0, 0.75, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.AMBLE:
            self.gait_data.gait_name = "AMBLE"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.5
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.6250
            self.gait_data.phase_offset = np.array([0.0, 0.5, 0.25, 0.75])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.TROT_WALK:
            self.gait_data.gait_name = "TROT_WALK"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.5
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.6
            self.gait_data.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.TROT:
            self.gait_data.gait_name = "TROT" 
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.5
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.5
            self.gait_data.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.TROT_RUN:
            self.gait_data.gait_name = "TROT_RUN"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.4
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.4
            self.gait_data.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.PACE:
            self.gait_data.gait_name = "PACE"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.35
            self.gait_data.initial_phase = 0.25
            self.gait_data.switching_phase_nominal = 0.5
            self.gait_data.phase_offset = np.array([0.0, 0.5, 0.0, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.BOUND:
            self.gait_data.gait_name = "BOUND"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.4
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.4
            self.gait_data.phase_offset = np.array([0.0, 0.0, 0.5, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.ROTARY_GALLOP:
            self.gait_data.gait_name = "ROTARY_GALLOP"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.4
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.2
            self.gait_data.phase_offset = np.array([0.0, 0.8571, 0.3571, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.TRAVERSE_GALLOP:
            self.gait_data.gait_name = "TRAVERSE_GALLOP"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.5
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.2
            self.gait_data.phase_offset = np.array([0.0, 0.8571, 0.3571, 0.5])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.PRONK:
            self.gait_data.gait_name = "PRONK"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            self.gait_data.period_time_nominal = 0.5
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.5
            self.gait_data.phase_offset = np.array([0.0, 0.0, 0.0, 0.0])
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.THREE_FOOT:
            self.gait_data.gait_name = "THREE_FOOT"
            self.gait_data.gait_enabled = np.array([0, 1, 1, 1])  # Front left disabled
            self.gait_data.period_time_nominal = 0.4
            self.gait_data.initial_phase = 0.0
            self.gait_data.switching_phase_nominal = 0.666
            self.gait_data.phase_offset = np.array([0.0, 0.666, 0.0, 0.333])
            self.gait_data.phase_scale = np.array([0.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 1
            
        elif self.gait_data._next_gait == GaitType.TRANSITION_TO_STAND:
            self.gait_data.gait_name = "TRANSITION_TO_STAND"
            self.gait_data.gait_enabled = np.array([1, 1, 1, 1])
            
            # Special transition case - uses current gait timing
            old_period_time = self.gait_data.period_time_nominal
            self.gait_data.period_time_nominal = 3 * old_period_time
            self.gait_data.initial_phase = 0.0
            
            # Compute transition phases
            self.gait_data.switching_phase_nominal = (
                (self.gait_data.period_time_nominal + old_period_time * 
                (self.gait_data.switching_phase_nominal - 1)) / 
                self.gait_data.period_time_nominal
            )
            
            # Compute transition offsets for each leg
            for i in range(4):
                self.gait_data.phase_offset[i] = (
                    (self.gait_data.period_time_nominal + old_period_time * 
                    (self.gait_data.phase_variable[i] - 1)) / 
                    self.gait_data.period_time_nominal
                )
                
            self.gait_data.phase_scale = np.array([1.0, 1.0, 1.0, 1.0])
            self.gait_data.overrideable = 0
            
        print(self.gait_data.gait_name)
        
        # Update gait
        self.gait_data._current_gait = self.gait_data._next_gait
        
        # Calculate auxiliary data
        self.calc_auxiliary_gait_data()

    def calc_auxiliary_gait_data(self):
        """Calculate auxilary gait data from parameters"""
        # Set the gait parameters for each foot
        for foot in range(4):
            if self.gait_data.gait_enabled[foot] == 1:
                # The scaled period time for each foot
                self.gait_data.period_time[foot] = (
                    self.gait_data.period_time_nominal / self.gait_data.phase_scale[foot]
                    )
                
            # Phase at which to switch the foot from stance to swing
            self.gait_data.switching_phase[foot] = self.gait_data.switching_phase_nominal

            # Init phase variables according to offset
            self.gait_data.phase_variable[foot] = (
                self.gait_data.initial_phase + 
                self.gait_data.phase_offset[foot])
            
            # Find the total stance time over the gait cycle
            self.gait_data.time_stance[foot] = (self.gait_data.period_time[foot] * self.gait_data.switching_phase[foot]
            )

            # Find the total swing time over the gait cycle
            self.gait_data.time_swing[foot] = (
                self.gait_data.period_time[foot] * (1.0 - self.gait_data.switching_phase[foot])
            )
        
        else:
            # Leg is disabled
            self.gait_data.period_time[foot] = 0.0
            self.gait_data.switching_phase[foot] = 0.0
            self.gait_data.phase_variable[foot] = 0.0
            self.gait_data.time_stance[foot] = 0.0 # Foot is never in stance
            self.gait_data.time_swing[foot] = float('inf') # Foot is always in swing

    def modify_gait(self):
        """Modify gait parameters"""
        # For now, just checking if we need to transition
        # May need to implement gait modification logic
        if self.gait_data._current_gait != self.gait_data._next_gait:
            self.create_gait()