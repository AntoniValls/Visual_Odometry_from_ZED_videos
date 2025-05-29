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

        # We now only care about the rotation along axis Y (yaw)
        y_angle = ang_vel[1] * dt

        position = position + velocity * dt + 0.5 * l_acc * dt**2





        # Update orientation using angular velocity
        # For small rotations: delta_angle ≈ ang_vel * dt
        delta_angle = ang_vel * dt
        delta_rotation = R.from_rotvec(delta_angle)
        orientation = orientation * delta_rotation 

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
import numpy as np

def imu_dead_reckoning(timestamps, linear_acc, angular_vel):
    """
    Perform dead reckoning using IMU data for X-Z plane trajectory.
   
    Args:
        timestamps: Array of timestamps in seconds
        linear_acc: Array of linear accelerations [N x 3] (ax, ay, az)
        angular_vel: Array of angular velocities [N x 3] (wx, wy, wz)
        
    Returns:
        trajectory: Dictionary containing:
            - 'position': [N x 2] array of (x, z) positions
            - 'velocity': [N x 2] array of (vx, vz) velocities  
            - 'heading': [N] array of heading angles (rotation about Y-axis)
            - 'timestamps': timestamps array
    """
    
    # Convert to numpy arrays if not already
    timestamps = np.array(timestamps)
    linear_acc = np.array(linear_acc)
    angular_vel = np.array(angular_vel)
    
    n_samples = len(timestamps)
    
    # Initialize output arrays
    position = np.zeros((n_samples, 2))  # [x, z]
    velocity = np.zeros((n_samples, 2))  # [vx, vz]
    heading = np.zeros(n_samples)        # rotation about Y-axis
    
    # Initial conditions (can be modified as needed)
    position[0] = [0.0, 0.0]
    velocity[0] = [0.0, 0.0]
    heading[0] = 0.0
    
    # Integration loop
    for i in range(1, n_samples):
        dt = timestamps[i] - timestamps[i-1]
        
        # Update heading using angular velocity about Y-axis (wy)
        heading[i] = heading[i-1] + angular_vel[i-1, 1] * dt
        
        # Get acceleration in body frame (ax, az)
        acc_body = np.array([linear_acc[i-1, 0], linear_acc[i-1, 2]])
        
        # Transform acceleration from body frame to world frame
        # Rotation matrix for rotation about Y-axis
        cos_h = np.cos(heading[i-1])
        sin_h = np.sin(heading[i-1])
        
        # Rotation matrix from body to world frame (2D rotation in X-Z plane)
        R = np.array([[cos_h, sin_h],
                      [-sin_h, cos_h]])
        
        acc_world = R @ acc_body
        
        # Remove gravity (assuming gravity acts in negative Z direction)
        # Note: This assumes the IMU is roughly level initially
        acc_world[1] += 9.81  # Remove gravity from Z component
        
        # Integrate acceleration to get velocity
        velocity[i] = velocity[i-1] + acc_world * dt
        
        # Integrate velocity to get position
        position[i] = position[i-1] + velocity[i-1] * dt + 0.5 * acc_world * dt**2
    
    return {
        'position': position,
        'velocity': velocity,
        'heading': heading,
        'timestamps': timestamps
    }

# Example usage and plotting function
def plot_trajectory(trajectory):
    """Plot the estimated trajectory"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # X-Z trajectory plot
    ax1.plot(trajectory['position'][:, 0], trajectory['position'][:, 1], 'b-', linewidth=2)
    ax1.scatter(trajectory['position'][0, 0], trajectory['position'][0, 1], 
                color='green', s=100, label='Start', zorder=5)
    ax1.scatter(trajectory['position'][-1, 0], trajectory['position'][-1, 1], 
                color='red', s=100, label='End', zorder=5)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Z Position (m)')
    ax1.set_title('X-Z Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Position vs time
    ax2.plot(trajectory['timestamps'], trajectory['position'][:, 0], 'r-', label='X')
    ax2.plot(trajectory['timestamps'], trajectory['position'][:, 1], 'b-', label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Velocity vs time
    ax3.plot(trajectory['timestamps'], trajectory['velocity'][:, 0], 'r-', label='Vx')
    ax3.plot(trajectory['timestamps'], trajectory['velocity'][:, 1], 'b-', label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Heading vs time
    ax4.plot(trajectory['timestamps'], np.degrees(trajectory['heading']), 'g-')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading (degrees)')
    ax4.set_title('Heading vs Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Function to process actual IMU data
def process_imu_data(imu_data):
    """
    Process IMU data and perform dead reckoning with ground truth comparison.
    
    Args:
        imu_data: List of dictionaries containing IMU measurements
    
    Returns:
        Dictionary with dead reckoning results and ground truth data
    """
    # Extract timestamps, positions, quaternions, angular velocities, and linear accelerations
    timestamps = np.array([d['timestamp'] for d in imu_data])
    positions = np.array([d['pose']['translation'] for d in imu_data])
    quaternions = np.array([d['pose']['quaternion'] for d in imu_data])
    angular_vel = np.array([d['angular_velocity'] for d in imu_data])
    linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    
    # Perform dead reckoning
    dr_result = imu_dead_reckoning(timestamps, linear_acc, angular_vel)
    
    # Extract ground truth X-Z positions
    ground_truth_xz = positions[:, [0, 2]]  # X and Z components
    
    return {
        'dead_reckoning': dr_result,
        'ground_truth_xz': ground_truth_xz,
        'ground_truth_full': positions,
        'quaternions': quaternions
    }

def plot_comparison(results):
    """Plot dead reckoning results compared to ground truth"""
    import matplotlib.pyplot as plt
    
    dr = results['dead_reckoning']
    gt_xz = results['ground_truth_xz']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # X-Z trajectory comparison
    ax1.plot(gt_xz[:, 0], gt_xz[:, 1], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(dr['position'][:, 0], dr['position'][:, 1], 'r--', linewidth=2, label='Dead Reckoning')
    ax1.scatter(gt_xz[0, 0], gt_xz[0, 1], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(gt_xz[-1, 0], gt_xz[-1, 1], color='red', s=100, label='End', zorder=5)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Z Position (m)')
    ax1.set_title('X-Z Trajectory Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Position error over time
    pos_error = np.linalg.norm(dr['position'] - gt_xz, axis=1)
    ax2.plot(dr['timestamps'], pos_error, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error vs Time')
    ax2.grid(True, alpha=0.3)
    
    # X and Z position comparison
    ax3.plot(dr['timestamps'], gt_xz[:, 0], 'g-', linewidth=2, label='GT X')
    ax3.plot(dr['timestamps'], dr['position'][:, 0], 'r--', linewidth=2, label='DR X')
    ax3.plot(dr['timestamps'], gt_xz[:, 1], 'b-', linewidth=2, label='GT Z')
    ax3.plot(dr['timestamps'], dr['position'][:, 1], 'm--', linewidth=2, label='DR Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Heading vs time
    ax4.plot(dr['timestamps'], np.degrees(dr['heading']), 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading (degrees)')
    ax4.set_title('Estimated Heading vs Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    final_error = pos_error[-1]
    mean_error = np.mean(pos_error)
    max_error = np.max(pos_error)
    
    print(f"\nDead Reckoning Performance:")
    print(f"Final position error: {final_error:.3f} m")
    print(f"Mean position error: {mean_error:.3f} m")
    print(f"Maximum position error: {max_error:.3f} m")

# Example usage with your IMU data
if __name__ == "__main__":
    
    imu_file = f'../datasets/BIEL/00/imu_data.txt'
    imu_data = load_sequential_data(imu_file, imu=True)
    results = process_imu_data(imu_data)
    plot_comparison(results)

 
# if __name__ == "__main__":

#     seq_num = '00'  # Change this to the desired sequence number
    
#     main(seq_num, vislam=True, imu=True) 
#     # main_dead_reckoning(seq_num)  # Run dead reckoning -> NOT WORKING YET
#     # main_debug_IMU(seq_num)  # Run debug dead reckoning

