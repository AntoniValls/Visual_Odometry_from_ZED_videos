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
    """
    Perform dead reckoning using IMU data.
    
    Args:
        timestamps: Array of timestamps in seconds
        linear_acc: Array of linear accelerations [N x 3]
        angular_vel: Array of angular velocities [N x 3]
    
    Returns:
        positions, velocities, orientations
    """

    positions = []
    velocities = []
    orientations = []

    # Initialize state
    position = np.array([0.0, 0.0, 0.0])
    velocity = np.array([0.0, 0.0, 0.0])
    orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

    # Gravity vector in world frame (negative Y is down)
    gravity = np.array([0.0, -9.81, 0.0])


    for idx in range(len(timestamps)):
        
        # Extract linear acceleration and angular velocity
        l_acc = linear_acc[idx]
        ang_vel = angular_vel[idx]
        
        # Calculate time step
        if idx > 0:
            dt = timestamps[idx] - timestamps[idx - 1]
        else:
            dt = 1/15 # Hz; Default to (15 FPS)

        # Update orientation using angular velocity
        # For small rotations: delta_angle ≈ ang_vel * dt
        delta_angle = ang_vel * dt
        delta_rotation = R.from_rotvec(delta_angle)
        orientation = orientation * delta_rotation # THE ORIENTATION QUATERNION MUST BE WRONG

        # Transform acceleration from body frame to world frame
        acc_world = orientation.apply(l_acc)

        # Remove gravity from world frame acceleration
        acc_world_corrected = acc_world - gravity
        
        #  # Apply low-pass filter to acceleration to reduce noise
        # if idx > 0:
        #     alpha = 0.8  # Filter coefficient
        #     acc_world_corrected = alpha * acc_world_corrected + (1 - alpha) * prev_acc
        # prev_acc = acc_world_corrected.copy()

        # # Zero small accelerations to reduce drift
        # acc_threshold = 0.5  # m/s²
        # acc_magnitude = np.linalg.norm(acc_world_corrected)
        # if acc_magnitude < acc_threshold:
        #     acc_world_corrected *= 0.1  # Significantly reduce small accelerations
        
        # Update velocity and position
        velocity = acc_world_corrected * dt
        print(f"Velocity at step {idx}: {velocity}")
        position += velocity * dt + 0.5 * acc_world_corrected * dt**2
        
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
    imu_data = load_sequential_data(imu_file, imu=True)
    if imu_data is None or len(imu_data) == 0:
        print(f"No valid IMU data found in {imu_file}.")
        return
    
    # Extract timestamps, positions, quaternions, angular velocities, and linear accelerations
    timestamps = np.array([d['timestamp'] for d in imu_data])
    angular_vel = np.array([d['angular_velocity'] for d in imu_data])
    linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    
    # Convert angular velocity to rad/s
    angular_vel = np.deg2rad(angular_vel)

    # Convert timestamps to seconds relative to the first one
    timestamps = (timestamps - timestamps[0]) * 1e-9  # ns → s
    
    # Perform dead reckoning
    positions, velocities, orientations = imu_dead_reckoning(timestamps, linear_acc, angular_vel)

    print(f"Dead reckoning completed for sequence {seq_num}")
    print(f"Final position: {positions[-1]}")
    print(f"Total displacement: {np.linalg.norm(positions[-1]):.2f} meters")
    
    # Plot the trajectory
    plot_vislam_trajectory_on_map(seq_num, positions)
    
    return positions

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

        # Convert angular velocity to rad/s
        angular_vel = np.deg2rad(angular_vel)

        # Convert timestamps to seconds relative to the first one
        timestamps = (timestamps - timestamps[0]) * 1e-9  # ns → s

        # Plot trajectory on map
        plot_imu_trajectory_on_map(seq_num, positions)

        plot_IMU_data(positions, timestamps, quaternions, angular_vel, linear_acc)


def debug_imu_data(imu_data_file):
    """
    Debug IMU data to identify issues causing kilometric advances.
    """
    # Load your IMU data
    imu_data = load_sequential_data(imu_data_file, imu=True)
    
    timestamps = np.array([d['timestamp'] for d in imu_data])
    angular_vel = np.array([d['angular_velocity'] for d in imu_data])
    linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    linear_acc_uncalibrated = np.array([d['linear_acceleration_uncalibrated'] for d in imu_data])
    
    # Convert timestamps to seconds
    timestamps = (timestamps - timestamps[0]) * 1e-9
    
    print("=== IMU DATA ANALYSIS ===")
    print(f"Data points: {len(imu_data)}")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    print(f"Average sample rate: {len(timestamps) / timestamps[-1]:.1f} Hz")
    
    # Check acceleration magnitudes
    acc_magnitude = np.linalg.norm(linear_acc, axis=1)
    acc_uncal_magnitude = np.linalg.norm(linear_acc_uncalibrated, axis=1)
    
    print(f"\n=== ACCELERATION ANALYSIS ===")
    print(f"Calibrated acceleration:")
    print(f"  Mean magnitude: {np.mean(acc_magnitude):.3f} m/s²")
    print(f"  Std magnitude: {np.std(acc_magnitude):.3f} m/s²")
    print(f"  Min/Max: {np.min(acc_magnitude):.3f} / {np.max(acc_magnitude):.3f}")
    
    print(f"Uncalibrated acceleration:")
    print(f"  Mean magnitude: {np.mean(acc_uncal_magnitude):.3f} m/s²")
    print(f"  Std magnitude: {np.std(acc_uncal_magnitude):.3f} m/s²")
    print(f"  Min/Max: {np.min(acc_uncal_magnitude):.3f} / {np.max(acc_uncal_magnitude):.3f}")
    
    # Check if gravity is already removed in calibrated data
    print(f"\n=== GRAVITY CHECK ===")
    mean_acc_cal = np.mean(linear_acc, axis=0)
    mean_acc_uncal = np.mean(linear_acc_uncalibrated, axis=0)
    print(f"Mean calibrated acceleration: [{mean_acc_cal[0]:.3f}, {mean_acc_cal[1]:.3f}, {mean_acc_cal[2]:.3f}]")
    print(f"Mean uncalibrated acceleration: [{mean_acc_uncal[0]:.3f}, {mean_acc_uncal[1]:.3f}, {mean_acc_uncal[2]:.3f}]")
    
    # If mean magnitude is close to 9.81, gravity is NOT removed
    # If mean magnitude is close to 0, gravity IS already removed
    gravity_removed_cal = np.abs(np.mean(acc_magnitude) - 9.81) > 5.0
    gravity_removed_uncal = np.abs(np.mean(acc_uncal_magnitude) - 9.81) > 5.0
    
    print(f"Gravity likely removed in calibrated data: {gravity_removed_cal}")
    print(f"Gravity likely removed in uncalibrated data: {gravity_removed_uncal}")
    
    # Check angular velocity
    gyro_magnitude = np.linalg.norm(angular_vel, axis=1)
    print(f"\n=== ANGULAR VELOCITY ANALYSIS ===")
    print(f"Mean magnitude: {np.mean(gyro_magnitude):.3f} rad/s")
    print(f"Max magnitude: {np.max(gyro_magnitude):.3f} rad/s")
    
    # Plot data for visual inspection
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Acceleration plots
    axes[0, 0].plot(timestamps, linear_acc[:, 0], 'r-', label='X')
    axes[0, 0].plot(timestamps, linear_acc[:, 1], 'g-', label='Y')
    axes[0, 0].plot(timestamps, linear_acc[:, 2], 'b-', label='Z')
    axes[0, 0].set_title('Calibrated Linear Acceleration')
    axes[0, 0].set_ylabel('m/s²')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(timestamps, linear_acc_uncalibrated[:, 0], 'r-', label='X')
    axes[0, 1].plot(timestamps, linear_acc_uncalibrated[:, 1], 'g-', label='Y')
    axes[0, 1].plot(timestamps, linear_acc_uncalibrated[:, 2], 'b-', label='Z')
    axes[0, 1].set_title('Uncalibrated Linear Acceleration')
    axes[0, 1].set_ylabel('m/s²')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Acceleration magnitude
    axes[1, 0].plot(timestamps, acc_magnitude, 'k-')
    axes[1, 0].axhline(y=9.81, color='r', linestyle='--', label='Gravity')
    axes[1, 0].set_title('Calibrated Acceleration Magnitude')
    axes[1, 0].set_ylabel('m/s²')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(timestamps, acc_uncal_magnitude, 'k-')
    axes[1, 1].axhline(y=9.81, color='r', linestyle='--', label='Gravity')
    axes[1, 1].set_title('Uncalibrated Acceleration Magnitude')
    axes[1, 1].set_ylabel('m/s²')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Angular velocity
    axes[2, 0].plot(timestamps, angular_vel[:, 0], 'r-', label='X')
    axes[2, 0].plot(timestamps, angular_vel[:, 1], 'g-', label='Y')
    axes[2, 0].plot(timestamps, angular_vel[:, 2], 'b-', label='Z')
    axes[2, 0].set_title('Angular Velocity')
    axes[2, 0].set_ylabel('degrees/s')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(timestamps, gyro_magnitude, 'k-')
    axes[2, 1].set_title('Angular Velocity Magnitude')
    axes[2, 1].set_ylabel('degrees/s')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'gravity_removed_calibrated': gravity_removed_cal,
        'gravity_removed_uncalibrated': gravity_removed_uncal,
        'mean_acc_magnitude_cal': np.mean(acc_magnitude),
        'mean_acc_magnitude_uncal': np.mean(acc_uncal_magnitude)
    }

# Usage example:
def main_debug_IMU(seq_num='00'):
    """
    Debug version of dead reckoning with analysis.
    """
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    # First, analyze the IMU data
    print("=== DEBUGGING IMU DATA ===")
    debug_imu_data(imu_file)
    
    
    return

if __name__ == "__main__":

    seq_num = '00'  # Change this to the desired sequence number
    
    # main(seq_num, vislam=True, imu=True) 
    main_dead_reckoning(seq_num)  # Run dead reckoning -> NOT WORKING YET
    # main_debug_IMU(seq_num)  # Run debug dead reckoning

