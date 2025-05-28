import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

from plot_trajectories import plot_trajectories_from_values

"""
Main script for IMU-based motion estimation and trajectory plotting.
This script loads IMU data, and plots the estimated the trajectory by
ZED's visual-inertial SLAM system.

It also has a dead reckoning function that estimates the trajectory
from IMU data using linear acceleration and angular velocity,
subtracts gravity, and integrates the motion over time.
"""

def load_sequential_data(file, imu=False):
    """ Load all sequential data"""
    data = []
    with open(file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if imu: # Extra check for IMU data
                if entry['is_available']:
                    data.append(entry)
            else:
                data.append(entry)
    
    return data

def imu_dead_reckoning(timestamps, linear_acc, angular_vel):
    
    positions = []
    velocities = []
    orientations = []

    # Initialize state
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

    for idx in range(len(timestamps)):
        
        # Extract linear acceleration and angular velocity
        l_acc = linear_acc[idx]
        ang_vel = angular_vel[idx]
        dt = timestamps[idx] - timestamps[idx - 1] if idx > 0 else 0.067  # Default to 0.067s (15 Frames/s) for the first iteration
        if dt <= 0:
            continue

        # Update orientation
        delta_angle = ang_vel * dt
        delta_rotation = R.from_rotvec(delta_angle)
        orientation = orientation * delta_rotation

        # Rotate acceleration to world frame
        acc_world = orientation.apply(l_acc)

        # Subtract gravity (assuming gravity is along Z-axis)
        acc_world[2] -= 9.81

        # Update velocity and position
        velocity += acc_world * dt
        position += velocity * dt

        # Store the state
        positions.append(position.copy())
        velocities.append(velocity.copy())
        orientations.append(orientation.as_quat())

    return np.array(positions), np.array(velocities), np.array(orientations)

def plot_IMU_data(positions, timestamps, quaternions, angular_vel, linear_acc):
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], label='Trajectory')
    ax.set_title("3D Pose Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 2D trajectory plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(positions[:, 0], positions[:, 2], label='Trajectory')
    ax.set_title("2D Pose Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Orientation (quaternion components over time)
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(["x", "y", "z", "w"]):
        plt.plot(timestamps, quaternions[:, i], label=f"q_{label}")
    plt.title("Quaternion Components Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Quaternion Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Linear Acceleration and Angular Velocity
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Angular velocity
    for i, label in enumerate(["x", "y", "z"]):
        axs[0].plot(timestamps, angular_vel[:, i], label=f"ω_{label}")
    axs[0].set_title("Angular Velocity Over Time")
    axs[0].set_ylabel("rad/s")
    axs[0].legend()
    axs[0].grid(True)

    # Linear acceleration
    for i, label in enumerate(["x", "y", "z"]):
        axs[1].plot(timestamps, linear_acc[:, i], label=f"a_{label}")
    axs[1].set_title("Linear Acceleration Over Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("m/s²")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_vislam_trajectory_on_map(seq_num, positions):

    if seq_num == "00":
        initial_utm = (426069.90, 4581718.85)
        angle_deg = -17  # Initial orientation in degrees

    # Rotate the positions to match the initial orientation
    initial_orientation = R.from_euler('y', angle_deg, degrees=True)  # Assuming initial orientation is along Y-axis
    positions = initial_orientation.apply(positions)
     
    # Convert positions to UTM world coordinates
    positions[:, 0] = initial_utm[0] + positions[:, 0]
    positions[:, 1] = 1.8 + positions[:, 1]  # Assuming a fixed height of 1.8m for the ZED glasses
    positions[: ,2] = initial_utm[1] + positions[:, 2]

    # Convert to (X, Z, Y) format
    positions = positions[:, [0, 2, 1]] 
    
    # Plot trajectory
    plot_trajectories_from_values([positions], seq=seq_num, labels=['ZED-VIO Estimation'])
    
    # Saving the trajectory in a .txt file
    save_dir = f"../datasets/predicted/trajectories/{seq_num}"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, "ZED_VIO_estimation.txt"), positions, fmt="%.16f") 

    return positions
 

def plot_imu_trajectory_on_map(seq_num, positions, save=False):
    
    # Define basic parameters
    if seq_num == '00':
        initial_utm = (426069.90, 4581718.85)  
        last_frame_utm = initial_utm  
        angle_deg = -17  # Initial orientation in degrees
   
   # Rotate the positions to match the initial orientation
    initial_orientation = R.from_euler('y', angle_deg, degrees=True)  # Assuming initial orientation is along Y-axis
    positions = initial_orientation.apply(positions)
     
    # Center positions so last frame is at origin
    positions -= positions[-1]  # Center the trajectory around the last frame

    # Convert positions to UTM coordinates
    positions[:,0] = last_frame_utm[0] + positions[:, 0]
    positions[:,1] = 1.8 + positions[:, 1]  # Assuming a fixed height of 1.8m for the ZED glasses
    positions[:,2] = last_frame_utm[1] + positions[:, 2]

     # Convert to (X, Z, Y) format
    positions = positions[:, [0, 2, 1]] 
    
    # Plot trajectory
    plot_trajectories_from_values([positions], seq=seq_num, labels=['ZED Estimation'])

    # Saving the trajectory in a .txt file
    if save:
        save_dir = f"../datasets/predicted/trajectories/{seq_num}"
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, "ZED_IMU_estimation.txt"), positions, fmt="%.16f") 

    return positions

def main_dead_reckoning(seq_num='00'): 
    """
    Main function to load IMU data and perform dead reckoning.
    Note: Not working yet.
    """

    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    # Load IMU data
    imu_data = load_sequential_data(imu_file)
    if imu_data is None or len(imu_data) == 0:
        print(f"No valid IMU data found in {imu_file}.")
        return
    
    # Extract timestamps, positions, quaternions, angular velocities, and linear accelerations
    timestamps = np.array([d['timestamp'] for d in imu_data])
    angular_vel = np.array([d['angular_velocity'] for d in imu_data])
    linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    
    # Convert timestamps to seconds relative to the first one
    timestamps = (timestamps - timestamps[0]) * 1e-9  # ns → s
    
    # Perform dead reckoning
    positions, velocities, orientations = imu_dead_reckoning(timestamps, linear_acc, angular_vel)

    print(positions, velocities, orientations)
    # Plot trajectory on map
    #plot_trajectory_on_map(seq_num, positions)

def main(seq_num='00', vislam= False, imu=False):
    """
    Main function to load VISlam data or IMU data and plot the estimated trajectory on a map.
    """

    vislam_file = f'../datasets/BIEL/{seq_num}/vislam_data.txt'
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    if vislam:
        vislam_data = load_sequential_data(vislam_file)

        # Extract positions and quaternions
        positions = np.array([d['pose']['translation'] for d in vislam_data])
        quaternions = np.array([d['pose']['quaternion'] for d in vislam_data])

        # Plot trajectory on map
        plot_vislam_trajectory_on_map(seq_num, positions)

    if imu:
        imu_data = load_sequential_data(imu_file, imu=True)
        if imu_data is None or len(imu_data) == 0:
            print(f"No valid IMU data found in {imu_file}.")
            return
        
        # Extract timestamps, positions, quaternions, angular velocities, and linear accelerations
        timestamps = np.array([d['timestamp'] for d in imu_data])
        positions = np.array([d['pose']['translation'] for d in imu_data])
        quaternions = np.array([d['pose']['quaternion'] for d in imu_data])
        angular_vel = np.array([d['angular_velocity'] for d in imu_data])
        linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    
        # Convert timestamps to seconds relative to the first one
        timestamps = (timestamps - timestamps[0]) * 1e-9  # ns → s

        # Plot trajectory on map
        plot_imu_trajectory_on_map(seq_num, positions)

        plot_IMU_data(positions, timestamps, quaternions, angular_vel, linear_acc)

if __name__ == "__main__":

    seq_num = '00'  # Change this to the desired sequence number
    
    main(seq_num, vislam=True, imu=True) 
    # main_dead_reckoning(seq_num)  # Run dead reckoning -> NOT WORKING YET
    


