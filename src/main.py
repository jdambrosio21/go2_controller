import time
import sys
import numpy as np
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from controllers.go2_controller import Go2Controller

def execute_stand_up(controller):
    """Execute standing sequence before main control"""
    print("Starting stand up sequence...")
    running_time = 0.0
    
    while running_time < 1.5:
        step_start = time.perf_counter()
        
        # Your original standup logic
        phase = np.tanh(running_time / 1.2)
        controller.execute_stand_up(phase)
        
        running_time += 0.02
        
        # Maintain timing
        time_until_next = 0.02 - (time.perf_counter() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    print("Stand up complete!")
    

def main():
    print("Starting Go2 Controller")

    # Initialize DDS Communication
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    
    # Wait for user confirmation
    input("Press Enter to start controller...") 

    try:
        # Initialize controller with URDF path (should probably use relative path)
        controller = Go2Controller("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf")

        # First enter standup sequence
        print("Starting stand up sequence...")
        execute_stand_up(controller)
        print("Stand up complete. Starting main control loop...")

        # Main control loop
        while True:
            try:
                # The run_control_loop method now handles all the timing
                # and frequency management internally
                controller.run_control_loop()

            except KeyboardInterrupt:
                print("\nStopping controller carefully...")
                # Should probabaly add some method here
                break

            except Exception as e:
                print(f"Error in control loop: {e}")
                # Return to standing if theres an error
                print("Attempting to return to standing...")
                controller.execute_stand_up()
    
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        raise

    finally:
        print("Shutting down controller...")
        # Should probably add clean up code

if __name__ == "__main__":
    main()
