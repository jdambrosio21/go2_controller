import numpy as np
from controllers.go2_controller import Go2Controller

def test_controller():
    # Init controller with urdf path
    controller = Go2Controller("~/unitree_ros/robots/go2_description/go2_description.urdf")

    # Test params
    dt = 0.02
    sim_time = 2.0 # simulate 2 seconds
    n_steps = int(sim_time/dt)

    # Initial state
    q = np.zeros(7)
    q[0:3] = [0, 0, 0.33]
    q[7] = 1.0

    dq = np.zeros(6)

    desired_vel = np.array([0.5, 0, 0])

    # Lists to store data
    forces_history = []
    foot_pos_history = []
    com_pos_history = []

    # Sim loop
    for i in range(n_steps):
        # Get MPC forces
        forces = controller.run(
            q=q,
            dq=dq,
            desired_vel=desired_vel,
            dt=dt
        )

        # Store data
        forces_history.append(forces)
        com_pos_history.append(q[0:3])

        if i % 50 == 0:
            print(f"Step {i}")
            print(f"CoM Pos: {q[0:3]}")
            print(f"Forces: {forces}")
            print("---")

            q[0:3] += dq[0:3] * dt

        return {
            'forces': forces_history,
            'com_positions': com_pos_history
        }
    
if __name__ == "__main__":
    results = test_controller()

    # plot results 
    import matplotlib.pyplot as plt

    # Plot CoM traj
    com_pos = np.array(results['com_positions'])
    plt.figure()
    plt.plot(com_pos[:, 0], label = 'x')
    plt.plot(com_pos[:, 1], label = 'y')
    plt.plot(com_pos[:, 2], label = 'z')
    plt.legend()
    plt.title('CoM Position')

    # Plot forces
    forces = np.array(results['forces'])
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(forces[:, i*3:i*3 + 3])
        plt.title(f'Foot {i} Forces')
        plt.legend(['x', 'y', 'z'])

    plt.show()