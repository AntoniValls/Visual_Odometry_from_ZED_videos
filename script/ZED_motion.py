import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils import GT_reader, generate_angles_near_deg
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
from plot_trajectories import plot_trajectories_from_values
from error_stats import interpolate_gt, compute_error_statistics

"""
Main script for ZED feaetures-based motion estimation and trajectory plotting.
This script loads VI-SLAM and IMU data, and plots the estimated the trajectory.

It also has a dead reckoning function that estimates the trajectory
from IMU data using linear acceleration and angular velocity,
subtracts gravity, and integrates the motion over time (but it doesn't work)
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

############################################################# IMU #######################################################################
def plot_imu_trajectory_on_map(seq_num, positions, save=False, label="ZED IMU Estimation"):
    
    # Define basic parameters
    max_lat, min_lat, max_lon, min_lon, zone_number, initial_utm, angle_deg = GT_reader(seq_num)

    last_frame_utm = initial_utm  
   
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

def dead_reckoning(imu_data_file, name='00'):
    """NOT WORKING"""

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

#################################### VI-SLAM ###################################

def quaternion_to_rotation_matrix(q):
    """Computess the rotation matrix based on the input quaternion."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    rotation_matrix = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    return rotation_matrix


def rotation_matrix_z(theta):
    """ Computes a rotation matrix around the z axis.""" 
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0, 0, 1]
    ])

    return rot_z
def plot_3d_trajectory(positions, seq_num="00"):
    """
    Plot simple 3D trajectory without external maps
    
    Args:
        positions: Array of positions
        is_relative: True if positions are relative to previous frame
        seq_num: Sequence number for saving/labeling
    """
    # Convert relative positions to cumulative if needed
    #if is_relative:
        # Correct the format
        #positions = positions[:, [0, 2, 1]]  
        # positions[:, 0] *= -1
        #positions = np.cumsum(positions, axis=0)
        #print(positions)

    
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

    plt.plot(positions[:, 1])
    plt.show()
    
    return positions

def process_vislam_trajectory(vislam_file, seq_num, plot_3d=False, max_iterations=2):

    """
    Process ZED VISLAM trajectory
    
    Args:
        - vislam_file: Path to the VISLAM data file. This file contains relative positions and orientations 
        with a RIGHT_HANDED_Y_DOWN_Z_FWD
        - seq_num: Sequence number
        - max_iterations: Maximum number of telescopic search iterations
    
    It computes the error trajectory (on beginning of the sequence) for a set of angles, with the initial angle read from GT_reader(),
    and finally selects the one with less error using telescopic search.
    """

    # Load the data
    save_dir = f"../datasets/predicted/trajectories/{seq_num}"
    vislam_data = load_sequential_data(vislam_file)

   # Extract positions and quaternions
    og_positions = np.array([d['pose']['translation'] for d in vislam_data])
    og_quaternions = np.array([d['pose']['quaternion'] for d in vislam_data])

    # Obtain parameters
    max_lat, min_lat, max_lon, min_lon, zone_number, initial_utm, angle_deg = GT_reader(seq_num)

    # Telescopic search loop
    current_angle = angle_deg
    iteration = 1
    spread = 180
    
    while iteration < max_iterations:
        print(f"Telescopic search iteration {iteration + 1}, spread {spread}, centered at angle: {current_angle}")
    
        # Generate angles near current angle:
        #angles = generate_angles_near_deg(current_angle, spread=spread)
        angles = [0]
        # Run for all angles
        all_rmse = []
        for angle in angles:
            positions = og_positions.copy()
            quaternions = og_quaternions.copy()

            # Create initial homogeneous matrix
            homo_matrix = np.eye(4)
            # homo_matrix[0, 3], homo_matrix[1, 3] = initial_utm
            # homo_matrix[2, 3] = 1.8
            # rotation_matrix = R.from_euler('z', angle, degrees=True).as_matrix()
            # homo_matrix[:3, :3] = rotation_matrix

            # Computes the error for the initial 25% of frames --> More accurate with the drift issue
            subset = int(len(positions) * 0.25)
            positions = positions[:subset] 
            quaternions = quaternions[:subset]    
            
            # Initialize trajectory
            trajectory = np.zeros((subset, 3, 4))
            trajectory[0] = homo_matrix[:3, :]

            # Main processing loop
            for i in range(subset - 1):
                translation_vector = positions[i]
                rotation_matrix = R.from_quat(quaternions[i]).as_matrix()

                # Create transformation - homogeneous matrix (4X4)
                Transformation_matrix = np.eye(4)
                Transformation_matrix[:3, :3] = rotation_matrix
                Transformation_matrix[:3, 3] = translation_vector.T

                # Update global pose
                homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))

                trajectory[i+1, :, :] = homo_matrix[:3, :]

            # Slice the position
            points = trajectory[:, :, 3]
            plot_3d_trajectory(points, seq_num=seq_num)
            plot_trajectories_from_values([points], seq=seq_num, labels=["ZED's Estimation"])

            # Compute error
            gt_file = os.path.join(save_dir, "GT.txt")
            gt = np.loadtxt(gt_file)
            interpolated_gt = interpolate_gt(gt, points)
            stats, errors = compute_error_statistics(points, interpolated_gt) 
            all_rmse.append(stats['rmse'])
            print(f"Run for angle: {angle}. Obtained RMSE: {stats['rmse']}")
        
        # Find best angle
        best_idx = np.argmin(all_rmse)
        best_angle = angles[best_idx]
        best_rmse = all_rmse[best_idx]

        print(f"Best angle in iteration {iteration + 1}: {best_angle} with RMSE: {best_rmse}")

        # Check convergence
        has_converged = True if (np.max(all_rmse) - np.min(all_rmse)) < 0.001 else False
        
        if (not has_converged) and (iteration < max_iterations - 1):
            # Rerun with best angle as new center and a smaller spread
            current_angle = best_angle
            spread = spread / (iteration + 1)
            iteration += 1
            print(f"Best angle {best_angle} is at boundary. Expanding search...")
        else:
            # Either not at boundary or max iterations reached
            angle_deg = best_angle
            if not has_converged:
                print(f"Reached maximum iterations ({max_iterations}). Using best angle: {angle_deg}")
            else:
                print(f"Converged! Best angle found: {angle_deg}")
            break
    
    print(f"Final best initial angle is {angle_deg}!")
        
    # Now re-run for the final best angle!
    # Create initial homogeneous matrix
    homo_matrix = np.eye(4)
    homo_matrix[0, 3], homo_matrix[1, 3] = initial_utm
    homo_matrix[2, 3] = 1.8
    rotation_matrix = R.from_euler('z', angle, degrees=True).as_matrix()
    homo_matrix[:3, :3] = rotation_matrix

    # Initialize trajectory
    trajectory = np.zeros((len(og_positions), 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    # Main processing loop
    for i in range(len(og_positions) - 1):
        translation_vector = og_positions[i]
        rotation_matrix = R.from_quat(og_quaternions[i]).as_matrix()

        # Create transformation - homogeneous matrix (4X4)
        Transformation_matrix = np.eye(4)
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        # Update global pose
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        trajectory[i+1, :, :] = homo_matrix[:3, :]

    # Slice the position
    points = trajectory[:, :, 3]

    if plot_3d:
        plot_3d_trajectory(points, seq_num=seq_num)

    plot_trajectories_from_values([points], seq=seq_num, labels=["ZED's Estimation"])
    
    # Saving the trajectory in a .txt file
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, "ZED_VIO_2_estimation.txt"), points, fmt="%.16f") 

    return points, angle_deg

def main(seq_num='00', vislam=False, imu=False):
    """
    Main function to load VISlam data or IMU data and plot the estimated trajectory on a map.

    Args:
        - seq_num (str): sequence identifier
        - vislam (bool): process ZED's VI-SLAM predictions from python API
        - imu (bool): process ZED's IMU features from python API
    """

    # Load the data
    vislam_file = f'../datasets/BIEL/{seq_num}/vislam_data.txt'
    imu_file = f'../datasets/BIEL/{seq_num}/imu_data.txt'
        
    # VISLAM processing
    if vislam:
        process_vislam_trajectory(vislam_file, seq_num, plot_3d=False)

    # IMU processing
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

    # seqs = [str(i).zfill(2) for i in range(2,21)]
    seqs = ["00"]
    for seq_num in seqs:
        main(seq_num, vislam=True, imu=False) 
    # main_debug_IMU(seq_num)  # Run debug dead reckoning

