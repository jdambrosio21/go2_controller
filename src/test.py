import numpy as np
from numpy.typing import NDArray
from planners.gait_scheduler import GaitScheduler, GaitType
from planners.footstep_planner import FootstepPlanner
from planners.foot_swing_trajectory import FootSwingTrajectory
from controllers.convex_mpc import ConvexMPC, MPCParams

def create_reference_trajectory(
    current_state: NDArray,  # [pos, ori, lin_vel, ang_vel, g]
    desired_vel: NDArray,    # [vx, vy, vz]
    desired_yaw_rate: float,
    horizon_steps: int,
    dt: float
) -> NDArray:
    """Create simple reference trajectory"""
    trajectory = np.zeros((13, horizon_steps + 1))  # +1 for initial state
    
    # Start from current state
    trajectory[:, 0] = current_state
    
    # For each future timestep
    for i in range(1, horizon_steps + 1):
        time = i * dt
        # Position increases with velocity
        trajectory[0:3, i] = current_state[0:3] + desired_vel * time
        # Yaw increases with yaw rate
        trajectory[5, i] = current_state[5] + desired_yaw_rate * time
        # Constant desired velocity
        trajectory[6:9, i] = desired_vel
        # Constant desired yaw rate
        trajectory[11, i] = desired_yaw_rate
        
    return trajectory

def test_controller():
    # Initialize components
    dt = 0.02
    horizon_steps = 10
    
    gait_scheduler = GaitScheduler(dt=dt)
    footstep_planner = FootstepPlanner()
    mpc = ConvexMPC(MPCParams(dt=dt, horizon_steps=horizon_steps))
    
    # Initial state [pos, ori, lin_vel, ang_vel, g]
    current_state = np.zeros(13)
    current_state[2] = 0.3  # Initial height
    
    # Desired movement
    desired_vel = np.array([0.5, 0.0, 0.0])  # Move forward at 0.5 m/s
    desired_yaw_rate = 0.0
    
    # Simulation loop
    sim_time = 3.0  # Simulate for 3 seconds
    for t in np.arange(0, sim_time, dt):
        # 1. Update gait scheduler
        gait_scheduler.step()
        
        # 2. Create reference trajectory
        reference_trajectory = create_reference_trajectory(
            current_state,
            desired_vel,
            desired_yaw_rate,
            horizon_steps,
            dt
        )
        
        # 3. Plan footsteps
        footstep_plans = footstep_planner.plan_footsteps(
            gait_scheduler=gait_scheduler,
            current_pos=current_state[0:3],
            current_yaw=current_state[5],
            desired_vel=desired_vel,
            dt=dt,
            planning_horizon=horizon_steps * dt
        )
        
        # 4. Collect foot positions for each step in the horizon
        foot_positions_horizon = []
        for i in range(horizon_steps):
            future_time = t + i * dt
            foot_positions_step = footstep_planner.get_foot_positions(
                gait_scheduler=gait_scheduler,
                footstep_plans=footstep_plans,
                current_time=future_time
            )
            foot_positions_horizon.append(foot_positions_step)
        
        # 5. Get contact schedule
        contact_schedule = np.zeros((4, horizon_steps))  # Initialize full schedule
        for i in range(horizon_steps):
            contact_schedule[:, i] = gait_scheduler.gait_data.contact_state_scheduled
        
        # 6. Solve MPC with updated foot_positions_horizon
        forces = mpc.solve(
            x0=current_state,
            x_ref=reference_trajectory,
            contact_schedule=contact_schedule,
            foot_positions=foot_positions_horizon
        )
        
        if forces is None:
            print(f"MPC failed at t={t}")
            continue
            
        # 7. Print some debug info
        print(f"Time: {t:.2f}")
        print(f"Position: {current_state[0:3]}")
        print(f"Contact states: {contact_schedule}")
        print(f"Forces: {forces}")
        print("---")
        
        # 8. Simple forward simulation (you'll replace this with actual robot)
        # Just update position based on velocity
        current_state[0:3] += desired_vel * dt
        
if __name__ == "__main__":
    test_controller()