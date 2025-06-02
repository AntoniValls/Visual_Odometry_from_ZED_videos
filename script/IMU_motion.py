import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
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

def plot_3d_trajectory(positions, is_relative=False, seq_num=""):
    """
    Plot simple 3D trajectory without external maps
    
    Args:
        positions: Array of positions
        is_relative: True if positions are relative to previous frame
        seq_num: Sequence number for saving/labeling
    """
    # Convert relative positions to cumulative if needed
    if is_relative:
        # Correct the format
        positions = positions[:, [1, 2, 0]]  
        positions[:, 0] *= -1
        positions = np.cumsum(positions, axis=0)

    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, label='ZED-VIO Trajectory')
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, marker='s', label='End')
    
    # Add arrows to show direction (every 10th point)
    step = max(1, len(positions) // 20)  # Show ~20 arrows max
    for i in range(0, len(positions) - step, step):
        direction = positions[i + step] - positions[i]
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                 direction[0], direction[1], direction[2],
                 length=0.1, normalize=True, alpha=0.6, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'ZED Camera 3D Trajectory - Sequence {seq_num}')
    ax.legend()
    
    # Make axes equal
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    
    return positions

def plot_vislam_trajectory_on_map(seq_num, positions, is_relative=False):
    """
    Plot VISLAM trajectory on map
    
    Args:
        seq_num: Sequence number
        positions: Array of positions
        is_relative: True if positions are relative to previous frame
    """
    if seq_num == "00":
            initial_utm = (426069.90, 4581718.85)
            angle_deg = -17  # Initial orientation in degrees
    
    # Convert relative positions to cumulative if needed
    if is_relative:
        # Correct the format
        positions = positions[:, [1, 2, 0]]  
        positions[:, 0] *= -1
        positions = np.cumsum(positions, axis=0)
        angle_deg = -10

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

def plot_imu_trajectory_on_map(seq_num, positions, save=False, label="ZED IMU Estimation"):
    
    # Define basic parameters
    if seq_num == '00':
        initial_utm = (426069.90, 4581718.85)  
        last_frame_utm = initial_utm  
        angle_deg = -17  # Initial orientation in degrees
   
   # Rotate the positions to match the initial orientation
    initial_orientation = R.from_euler('y', angle_deg, degrees=True)  # Assuming initial orientation is along Y-axis
    positions = initial_orientation.apply(positions)
     
    # Convert positions to UTM coordinates
    positions[:,0] = last_frame_utm[0] + positions[:, 0]
    positions[:,1] = 1.8 + positions[:, 1]  # Assuming a fixed height of 1.8m for the ZED glasses
    positions[:,2] = last_frame_utm[1] + positions[:, 2]

     # Convert to (X, Z, Y) format
    positions = positions[:, [0, 2, 1]] 
    
    # Plot trajectory
    plot_trajectories_from_values([positions], seq=seq_num, labels=[label])

    # Saving the trajectory in a .txt file
    if save:
        save_dir = f"../datasets/predicted/trajectories/{seq_num}"
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, "ZED_IMU_estimation.txt"), positions, fmt="%.16f") 

    return positions

def process_vislam_trajectory(vislam_file, seq_num, is_relative=False, plot_3d=False):
    """
    Process and plot VISLAM trajectory
    
    Args:
        vislam_file: Path to the VISLAM data file
        seq_num: Sequence number for the dataset
        is_relative: True if positions are relative to previous frame, False if cumulative
    """
    vislam_data = load_sequential_data(vislam_file)
    # Extract positions and quaternions
    positions = np.array([d['pose']['translation'] for d in vislam_data])
    quaternions = np.array([d['pose']['quaternion'] for d in vislam_data])
    
    if plot_3d:
        # Plot simple 3D trajectory
        plot_3d_trajectory(positions, is_relative=is_relative, seq_num=seq_num)
    else:
        # Plot trajectory on map
        plot_vislam_trajectory_on_map(seq_num, positions, is_relative=is_relative)

    return

def dead_reckoning(imu_data_file, name='00'):

    imu_data = load_sequential_data(imu_data_file, imu=True) if os.path.exists(imu_data_file) else None

    timestamps = np.array([d['timestamp'] for d in imu_data])
    angular_vel = np.array([d['angular_velocity'] for d in imu_data])
    linear_acc = np.array([d['linear_acceleration'] for d in imu_data])
    
    # Define gravity vector
    gravity = np.array([0, -9.81, 0])  

    # Convert timestamps to seconds
    timestamps = (timestamps - timestamps[0]) * 1e-9

    # Create initial homogeneous matrix
    homo_matrix = np.eye(4)

    # Initialize trajectory
    trajectory = np.zeros((len(timestamps), 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    num_frames = len(timestamps)
    
    for i in  tqdm(range(num_frames-1) , desc=f"Dead reckoning with IMU"):
        dt = 15 # Hz 

        acc = linear_acc[i][[0,2]]
        gyro = angular_vel[i][[0,2]]

        # Remove gravity
        acc = acc - gravity

        # Update rotation interval (integrate angular velocity)
        delta_r = cv2.Rodrigues(gyro * dt)[0]

        # Update velocity and position intervals
        delta_v = acc * dt
        delta_p = delta_v * dt + 0.5 * acc * dt**2

        print(delta_r, delta_v, delta_p)
        # Create transformation matrix
        Transformation_matrix = np.eye(4)
        Transformation_matrix[:3, :3] = delta_r
        Transformation_matrix[:3, 3] = delta_p.T

        # Update global pose
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        trajectory[i+1, :, :] = homo_matrix[:3, :]

    positions = trajectory[:,:,3]

    return positions
  
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
    print(f"Mean magnitude: {np.mean(gyro_magnitude):.3f} degree/s")
    print(f"Max magnitude: {np.max(gyro_magnitude):.3f} degree/s")
    
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

def main_debug_IMU(seq_num='00'):
    """
    Debug version of dead reckoning with analysis.
    """
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    # First, analyze the IMU data
    print("=== DEBUGGING IMU DATA ===")
    debug_imu_data(imu_file)
    
    return

def main_dead_reckoning(seq_num='00'):
    """
    Main function for dead reckoning using IMU data.
    This function loads the IMU data, processes it, and plots the estimated trajectory.
    """
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    if not os.path.exists(imu_file):
        print(f"IMU data file {imu_file} does not exist.")
        return
    
    positions = dead_reckoning(imu_file, name=seq_num)
    
    # Plot the trajectory on a map
    plot_imu_trajectory_on_map(seq_num, positions, save=True, label="Dead Reckoning")

    return

def main(seq_num='00', vislam= False, imu=False):
    """
    Main function to load VISlam data or IMU data and plot the estimated trajectory on a map.
    """

    vislam_file = f'../datasets/BIEL/{seq_num}/vislam_data.txt'
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
    
    if vislam:
        process_vislam_trajectory(vislam_file, seq_num, is_relative=False, plot_3d=False)

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
        
        # Convert relative positions to cumulative
        positions = np.cumsum(positions, axis=0)

        # Convert angular velocity to rad/s
        angular_vel = np.deg2rad(angular_vel)

        # Convert timestamps to seconds relative to the first one
        timestamps = (timestamps - timestamps[0]) * 1e-9  # ns → s

        # Plot trajectory on map
        plot_imu_trajectory_on_map(seq_num, positions)

        plot_IMU_data(positions, timestamps, quaternions, angular_vel, linear_acc)

if __name__ == "__main__":

    seq_num = '00'  # Change this to the desired sequence number
    
    # main(seq_num, vislam=False, imu=True) 
    main_dead_reckoning(seq_num)  # Run dead reckoning -> NOT WORKING YET
    # main_debug_IMU(seq_num)  # Run debug dead reckoning

