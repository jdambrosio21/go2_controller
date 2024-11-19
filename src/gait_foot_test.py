import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from planners.gait_scheduler import GaitScheduler
from planners.footstep_planner import FootstepPlanner
from mpl_toolkits.mplot3d import Axes3D

def create_reference_trajectory(duration, dt, desired_vel):
    """Create simple reference trajectory"""
    steps = int(duration / dt)
    
    # Initialize arrays
    com_pos = np.zeros((steps, 3))
    com_vel = np.zeros((steps, 3))
    
    # Set constant velocity and integrate for position
    com_vel[:] = desired_vel
    for i in range(1, steps):
        com_pos[i] = com_pos[i-1] + com_vel[i-1] * dt
        
    return com_pos, com_vel

def main():
    # Parameters
    duration = 10.0  # seconds to simulate
    dt = 0.02       # timestep
    desired_vel = np.array([0.5, 0, 0])  # 0.5 m/s forward

    # Define legs list
    legs = ['FL', 'FR', 'RL', 'RR']  # Add this line
    
    # Initialize planners
    gait_scheduler = GaitScheduler(total_period=0.5, gait_type="trot")  # 0.5s period trot
    footstep_planner = FootstepPlanner("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf", k_raibert=0.03)
    
    # Create reference trajectory
    com_pos, com_vel = create_reference_trajectory(duration, dt, desired_vel)
    steps = com_pos.shape[0]
    
    # Storage for foot positions and contact states
    foot_positions = np.zeros((4, 3, steps))  # (num_feet, xyz, timesteps)
    contact_states = np.zeros((4, steps))     # (num_feet, timesteps)
    
    # Initial robot state
    q_current = np.zeros(19)  # [base_pos(3), base_quat(4), joint_pos(12)]
    q_current[7:] = np.array([0.0, 0.6, -1.2,   # FL hip, thigh, knee
                             0.0, 0.6, -1.2,   # FR
                             0.0, 0.6, -1.2,   # RL
                             0.0, 0.6, -1.2])  # RR
    
    # Get initial foot positions from robot model
    current_feet = footstep_planner.quadruped.get_foot_positions(q_current)

    # Initialize foot history for plotting
    foot_history = {leg: [] for leg in ['FL', 'FR', 'RL', 'RR']}
    
    # Simulate and collect data
    print("Simulating...")
    for i in range(steps):
        q_current[0:3] = com_pos[i]  # Update base position
        
        # Update gait scheduler
        gait_scheduler.update(dt)
        
        # Get contact states
        contact_states[:, i] = gait_scheduler.get_current_contact_state()
        
        # Update COM
        current_com = com_pos[i]

        # Plan next footsteps
        q_current = np.zeros(19)  # Dummy state vector
        next_footholds = footstep_planner.plan_footsteps(
            com_state=(com_pos[i], com_vel[i], np.zeros(3)),
            desired_vel=desired_vel,
            q_current=q_current,
            gait_scheduler=gait_scheduler
        )
        
        # Plan footsteps
        next_footholds = footstep_planner.plan_footsteps(
            com_state=(current_com, com_vel[i], np.zeros(3)),
            desired_vel=desired_vel,
            q_current=q_current,
            gait_scheduler=gait_scheduler
        )
        
        # Update foot positions and store history
        for leg in range(4):
            if contact_states[leg, i] == 0:  # Swing
                foot_positions[leg, :, i] = next_footholds[leg]
                if i > 0 and contact_states[leg, i-1] == 1:  # Just entered swing
                    current_feet[leg] = next_footholds[leg]
            else:  # Stance - keep current position
                foot_positions[leg, :, i] = current_feet[leg]
            
            # Store for plotting
            foot_history[legs[leg]].append(foot_positions[leg, :, i])
    
    # Plotting
    print("Creating plots...")
    
    # Figure 1: Contact Schedule
    plt.figure(figsize=(12, 4))
    legs = ['FL', 'FR', 'RL', 'RR']
    time = np.arange(steps) * dt
    
    for leg in range(4):
        plt.plot(time, contact_states[leg] + leg*1.1, 'b-', label=legs[leg])
        plt.fill_between(time, leg*1.1, contact_states[leg] + leg*1.1, alpha=0.3)
    
    plt.yticks(np.arange(4)*1.1 + 0.5, legs)
    plt.xlabel('Time (s)')
    plt.title('Contact Schedule (Blue = Stance)')
    plt.grid(True)
    
    # Figure 2: Top-down view of footsteps and COM
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'y']
    
    # Plot COM trajectory
    plt.plot(com_pos[:, 0], com_pos[:, 1], 'k-', label='COM')
    
    # Plot foot positions
    for leg in range(4):
        for i in range(0, steps, 10):  # Plot every 10th step for clarity
            if contact_states[leg, i] == 1:  # Stance
                plt.plot(foot_positions[leg, 0, i], foot_positions[leg, 1, i], 
                        'o', color=colors[leg], alpha=0.5)
            else:  # Swing
                plt.plot(foot_positions[leg, 0, i], foot_positions[leg, 1, i], 
                        'x', color=colors[leg], alpha=0.5)
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Top-down View (o=stance, x=swing)')
    plt.legend(['COM'] + legs)
    
    # Figure 3: 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot COM trajectory
    ax.plot(com_pos[:, 0], com_pos[:, 1], com_pos[:, 2], 'k-', label='COM')
    
    # Plot foot trajectories
    for leg in range(4):
        ax.plot(foot_positions[leg, 0, :], 
                foot_positions[leg, 1, :], 
                foot_positions[leg, 2, :], 
                color=colors[leg], label=legs[leg])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()