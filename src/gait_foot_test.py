import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from planners.gait_scheduler import GaitScheduler
from planners.footstep_planner import FootstepPlanner
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

def create_reference_trajectory(duration, dt, desired_vel):
    """Create a simple reference trajectory for CoM."""
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
    duration = 10.0
    dt = 0.02
    desired_vel = np.array([0.5, 0, 0])
    mpc_horizon = 10  # MPC prediction horizon
    legs = ['FL', 'FR', 'RL', 'RR']

    # Initialize planners
    gait_scheduler = GaitScheduler(total_period=0.5, gait_type="trot")
    footstep_planner = FootstepPlanner(
        "/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf", 
        k_raibert=0.03
    )
    
    # Create reference trajectory
    com_pos, com_vel = create_reference_trajectory(duration, dt, desired_vel)
    steps = com_pos.shape[0]
    
    # Initialize storage arrays
    foot_positions = np.zeros((4, 3, steps))
    contact_states = np.zeros((4, steps))
    horizon_footsteps_history = []  # Store MPC predictions
    
    # Initial state setup
    q_current = np.zeros(19)
    q_current[3:7] = [0, 0, 0, 1]  # Identity quaternion
    stand_up_joint_pos = np.array([
        0.00571868, 0.608813, -1.21763,     # FR
        -0.00571868, 0.608813, -1.21763,    # FL
        0.00571868, 0.608813, -1.21763,     # RR
        -0.00571868, 0.608813, -1.21763     # RL
    ])
    q_nom = stand_up_joint_pos
    current_feet = footstep_planner.quadruped.get_foot_positions(q_current)
    
    print("Simulating...")
    for i in range(steps):
        # Update state
        q_current[0:3] = com_pos[i]
        
        # Update gait scheduler
        gait_scheduler.update(dt)
        contact_states[:, i] = gait_scheduler.get_current_contact_state()
        
        # Create reference trajectory for MPC horizon
        x_ref = np.zeros((13, mpc_horizon + 1))  # +1 because we need current state too
        for j in range(mpc_horizon + 1):
            idx = min(i + j, steps - 1)  # Prevent index out of bounds
            x_ref[3:6, j] = com_pos[idx]    # Position
            x_ref[9:12, j] = com_vel[idx]   # Velocity
            x_ref[12, j] = 9.81             # Gravity

        # Get MPC horizon prediction
        horizon_footsteps = footstep_planner.plan_horizon_footsteps(
            dt,
            mpc_horizon,
            x_ref,
            q_nom,
            q_current,
            gait_scheduler
        )
        horizon_footsteps_history.append(horizon_footsteps)
        
        # Update foot positions
        for leg in range(4):
            if contact_states[leg, i] == 0:  # Swing
                foot_positions[leg, :, i] = horizon_footsteps[leg*3:(leg+1)*3, 0]
                current_feet[leg] = horizon_footsteps[leg*3:(leg+1)*3, 0]
            else:  # Stance
                foot_positions[leg, :, i] = current_feet[leg]
    
    # Call animation with stored history
    animate_results(com_pos, foot_positions, contact_states, dt, horizon_footsteps_history)

def plot_results(com_pos, foot_positions, contact_states, dt, foot_history):
    """Improved visualization of contact schedules and foot positions with a proper legend."""
    steps = com_pos.shape[0]
    time = np.arange(steps) * dt
    legs = ['FL', 'FR', 'RL', 'RR']
    colors = ['r', 'g', 'b', 'y']

    # Contact schedule plot
    plt.figure(figsize=(12, 4))
    for leg in range(4):
        plt.plot(time, contact_states[leg] + leg * 1.1, 'b-', label=legs[leg])
        plt.fill_between(time, leg * 1.1, contact_states[leg] + leg * 1.1, alpha=0.3)
    plt.yticks(np.arange(4) * 1.1 + 0.5, legs)
    plt.xlabel('Time (s)')
    plt.title('Contact Schedule (Blue = Stance)')
    plt.grid(True)

    # Top-down foot positions and COM trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(com_pos[:, 0], com_pos[:, 1], 'k-', label='COM', linewidth=2)
    for leg in range(4):
        for i in range(0, steps, 10):  # Reduce frequency for clarity
            if contact_states[leg, i] == 1:  # Stance
                plt.scatter(foot_positions[leg, 0, i], foot_positions[leg, 1, i], 
                            c=colors[leg], marker='o', s=60, alpha=0.8)
            else:  # Swing
                plt.scatter(foot_positions[leg, 0, i], foot_positions[leg, 1, i], 
                            c=colors[leg], marker='x', s=30, alpha=0.8)

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Top-down Footsteps (o=stance, x=swing)')
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Stance', markerfacecolor='k', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Swing', markerfacecolor='k', markersize=10),
        Line2D([0], [0], color='k', label='COM', linewidth=2),
    ]
    for i, leg in enumerate(legs):
        legend_elements.append(Line2D([0], [0], marker='o', color=colors[i], label=leg, markersize=10))
    
    plt.legend(handles=legend_elements, loc='best')

    # # 3D Trajectory plot
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(com_pos[:, 0], com_pos[:, 1], com_pos[:, 2], 'k-', label='COM', linewidth=2)
    # for leg in range(4):
    #     ax.plot(foot_positions[leg, 0, :], 
    #             foot_positions[leg, 1, :], 
    #             foot_positions[leg, 2, :], 
    #             color=colors[leg], label=legs[leg])
    # ax.set_xlabel('X (m)')
    # ax.set_yl
    plt.show()

def animate_results(com_pos, foot_positions, contact_states, dt, horizon_footsteps_history):
    import matplotlib.animation as animation
    
    steps = com_pos.shape[0]
    legs = ['FL', 'FR', 'RL', 'RR']
    colors = ['r', 'g', 'b', 'y']
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    def update(frame):
        ax.clear()
        
        # Plot COM trajectory history
        ax.plot(com_pos[:frame, 0], com_pos[:frame, 1], 'k-', label='COM History', alpha=0.5)
        
        # Plot predicted COM trajectory for horizon
        horizon_length = 10  # Your MPC horizon length
        future_indices = range(frame, min(frame + horizon_length, steps))
        if len(future_indices) > 0:
            ax.plot(com_pos[future_indices, 0], com_pos[future_indices, 1], 
                   'k--', label='Predicted COM', alpha=0.5)
        
        # Plot foot position history
        for leg in range(4):
            # Historical foot positions
            foot_x = foot_positions[leg, 0, :frame]
            foot_y = foot_positions[leg, 1, :frame]
            if frame > 0:
                ax.plot(foot_x, foot_y, color=colors[leg], alpha=0.3, label=f'{legs[leg]} History')
            
            # Current foot position
            if contact_states[leg, frame] == 1:  # Stance
                ax.scatter(foot_positions[leg, 0, frame], foot_positions[leg, 1, frame],
                          c=colors[leg], marker='o', s=100, label=f'{legs[leg]} Stance')
            else:  # Swing
                ax.scatter(foot_positions[leg, 0, frame], foot_positions[leg, 1, frame],
                          c=colors[leg], marker='x', s=100, label=f'{legs[leg]} Swing')
            
            # Plot MPC predicted footsteps
            if frame < len(horizon_footsteps_history):
                horizon_steps = horizon_footsteps_history[frame]
                # Plot predicted foot positions for this leg
                predicted_x = horizon_steps[leg*3, :]
                predicted_y = horizon_steps[leg*3 + 1, :]
                ax.plot(predicted_x, predicted_y, color=colors[leg], 
                       linestyle='--', alpha=0.5)
                # Mark predicted touchdown positions
                ax.scatter(predicted_x, predicted_y, color=colors[leg], 
                         marker='.', alpha=0.3, s=50)
        
        # Draw support polygon for stance feet
        stance_feet = []
        for leg in range(4):
            if contact_states[leg, frame] == 1:
                stance_feet.append([foot_positions[leg, 0, frame], 
                                 foot_positions[leg, 1, frame]])
        if len(stance_feet) >= 3:  # Need at least 3 points for a polygon
            stance_feet = np.array(stance_feet)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(stance_feet)
            for simplex in hull.simplices:
                ax.plot(stance_feet[simplex, 0], stance_feet[simplex, 1], 
                       'k-', alpha=0.3)
        
        # Set consistent view
        ax.set_xlim(com_pos[frame, 0] - 0.8, com_pos[frame, 0] + 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Time: {frame*dt:.2f}s')
        
        # Add legend (show only unique entries)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Keep aspect ratio equal
        ax.set_aspect('equal')
        
    ani = animation.FuncAnimation(fig, update, frames=steps, 
                                interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    
    # Optionally save animation
    # ani.save('footstep_planning.gif', writer='pillow')
    import matplotlib.animation as animation
    
    steps = com_pos.shape[0]
    legs = ['FL', 'FR', 'RL', 'RR']
    colors = ['r', 'g', 'b', 'y']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    def update(frame):
        ax.clear()
        
        # Plot COM trajectory up to current frame
        ax.plot(com_pos[:frame, 0], com_pos[:frame, 1], 'k-', label='COM')
        
        # Plot foot position history up to current frame
        for leg in range(4):
            foot_x = foot_positions[leg, 0, :frame]
            foot_y = foot_positions[leg, 1, :frame]
            if frame > 0:
                ax.plot(foot_x, foot_y, color=colors[leg], alpha=0.3)
            
            # Current foot position
            if contact_states[leg, frame] == 1:
                ax.scatter(foot_positions[leg, 0, frame], foot_positions[leg, 1, frame],
                          c=colors[leg], marker='o', s=60)
            else:
                ax.scatter(foot_positions[leg, 0, frame], foot_positions[leg, 1, frame],
                          c=colors[leg], marker='x', s=30)
        
        # Set consistent view
        ax.set_xlim(com_pos[frame, 0] - 1, com_pos[frame, 0] + 1)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Time: {frame*dt:.2f}s')
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='k', label='COM'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='Stance'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='k', label='Swing')
        ]
        ax.legend(handles=legend_elements)
        
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    plt.show()
if __name__ == "__main__":
    main()
