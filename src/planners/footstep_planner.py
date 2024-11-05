import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from planners.gait_scheduler import GaitScheduler
from numpy.typing import NDArray

@dataclass
class FootstepPlan:
    """Class for keeping track of a single footstep plan"""
    position: NDArray    # Desired foot position in world frame
    start_time: float       # When to start moving foot
    contact_time: float     # When foot should touch down
    lift_time: float        # When foot should lift off


class FootstepPlanner:
    def __init__(self, 
                 foot_radius: float = 0.2, 
                 default_height: float = -0.3):
        """Initialize footstep planner

        Args:
            foot_radius: Vector from CoM to foot in XY plane
            default_height: Default foot Z-height in stance
        """

        # Default stance parameters
        self.default_height = default_height

        # Default foot positions in robot frame (should probably use pinocchio for this)
        self.default_stance = np.array([
            [foot_radius, -foot_radius, default_height], #FL
            [foot_radius, foot_radius, default_height], #FR
            [-foot_radius, -foot_radius, default_height], #BL
            [-foot_radius, foot_radius, default_height], #BR
        ])

    def plan_footsteps(self,
                       gait_scheduler: GaitScheduler,
                       current_pos: NDArray,
                       current_yaw: float,
                       desired_vel: NDArray,
                       dt: float,
                       planning_horizon: float) -> List[List[FootstepPlan]]:
        """Plan footsteps over time horizon using Raibert Heuristic

        Args:
            gait_scheduler: Current gait scheduler
            current_pos: Current robot COM position (x,y,z)
            current_yaw: Current robot yaw
            desired_vel: Desired COM velocity (x,y,z)
            dt: Timestep duration
            planning_horizon: How far to plan ahead

        Returns:
            List of FootstepPlans for each leg
        """

        # Get number of timesteps in horizon
        n_steps = int(planning_horizon / dt)

        # Initialize plans for each leg
        footstep_plans = [[] for _ in range(4)]

        # Get stance durations from gait scheduler
        stance_duration = gait_scheduler.gait_data.time_stance

        # Create rotation matrix from yaw
        R = np.array([
            [np.cos(current_yaw), -np.sin(current_yaw), 0],
            [np.sin(current_yaw),  np.cos(current_yaw), 0],
            [0,                    0,                   1]
        ])
        
        # Plan for each leg
        for leg in range(4):
            # Only plan if leg is enabled
            if gait_scheduler.gait_data.gait_enabled[leg] == 1:
                # Get phase and timing info
                phase = gait_scheduler.gait_data.phase_variable[leg]

                # Check if were entering swing phase
                if (gait_scheduler.gait_data.contact_state_scheduled[leg] == 1 
                    and gait_scheduler.gait_data.time_stance_remaining[leg] < planning_horizon):
                        
                    # Get timing
                    start_time = gait_scheduler.gait_data.time_stance_remaining[leg] + gait_scheduler.dt
                    contact_time = gait_scheduler.gait_data.time_swing[leg]
                    lift_time = contact_time + stance_duration[leg]

                    # Get reference position (location on ground beneath the hip) (prob use pinocchio)
                    p_ref = current_pos + R @ self.default_stance[leg]

                    # Apply Raibert Heuristic
                    # p_des = p_ref + v_com * Δt/2
                    delta_t = stance_duration[leg]
                    p_des = p_ref + desired_vel * (delta_t / 2)

                    # Keep Z at default height
                    p_des[2] = self.default_height

                    # Create footstep plan
                    plan = FootstepPlan(
                        position=p_des,
                        start_time=start_time,
                        contact_time=contact_time,
                        lift_time=lift_time
                    )

                    footstep_plans[leg].append(plan)
        
        return footstep_plans
                    
    def get_foot_positions(self,
                           gait_scheduler: GaitScheduler,
                           footstep_plans: List[List[FootstepPlan]],
                           current_time: float) -> List[NDArray]:
        """Get current desired foot positions.

        Args:
            gait_scheduler: Current gait scheduler
            footstep_plans: List of footstep plans for each leg
            current_time: Current time

        Returns:
            List of desired foot positions for each leg
        """

        positions = []

        for leg in range(4):
            # Find active plan for this leg
            active_plan = None
            for plan in footstep_plans[leg]:
                if plan.start_time <= current_time <= plan.lift_time:
                    active_plan = plan
                    break

            if active_plan is None:
                # No active plan, use current position
                pos = np.array([0.0, 0.0, self.default_height]) # Get actual position with pincchio
            else:
                pos = active_plan.position

            positions.append(pos)

        return positions
    
# Example usage 
if __name__ == "__main__":
    from gait_scheduler import GaitScheduler, GaitType
    
    # Create scheduler
    gait_scheduler = GaitScheduler(dt=0.02)
    footstep_planner = FootstepPlanner()
    
    # Test planning
    current_pos = np.array([0., 0., 0.5])
    current_yaw = 0.0
    desired_vel = np.array([0.5, 0., 0.])  # Moving forward at 0.5 m/s
    
    # Plan footsteps
    plans = footstep_planner.plan_footsteps(
        gait_scheduler=gait_scheduler,
        current_pos=current_pos,
        current_yaw=current_yaw,
        desired_vel=desired_vel,
        dt=0.02,
        planning_horizon=0.5
    )
    print(plans)
    
    # Get foot positions for current time
    positions = footstep_planner.get_foot_positions(
        gait_scheduler=gait_scheduler,
        footstep_plans=plans,
        current_time=0.0
    )
    print(positions)