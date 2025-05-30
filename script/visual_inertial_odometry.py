import os
import sys
import tilemapbase
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import *
from motion_estimation import motion_estimation
from feature_matching import FeatureMatcher
from sde import StereoDepthEstimator
from IMU_motion import load_sequential_data
from segmentation_utils import street_segmentation
from scipy.spatial.transform import Rotation, Slerp
from pyproj import Proj
from tqdm import tqdm
from collections import deque

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM 

class VisualInertialOdometry:

    def __init__(self, config, intrinsic_matrix):
        self.config = config
        self.intrinsic_matrix = intrinsic_matrix
        
        # IMU-related parameters
        self.gravity = np.array([0, -9.81, 0])  # Gravity vector in world frame
        
        # State variables
        self.current_velocity = np.zeros(3)  # Current velocity in world frame
        self.last_timestamp = None
        
        # Sliding window for optimization
        self.window_size = config.get('window_size', 10)
        self.pose_window = deque(maxlen=self.window_size)
        self.imu_window = deque(maxlen=self.window_size * 10)
        
        # Fusion weights imu_weight + visual_weight = 1
        self.imu_weight = config.get('imu_weight', 0.2)

        return
    
    def extract_yaw_rotation(self, R):
        """Extract only yaw rotation from rotation matrix"""
        # Get yaw angle from rotation matrix
        yaw = np.arctan2(R[0, 2], R[0, 0])
        
        # Create yaw-only rotation matrix
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [cos_yaw, 0, sin_yaw],
            [0, 1, 0],
            [-sin_yaw, 0, cos_yaw]
        ])
        return R_yaw

    def preintegrate_imu(self, imu_measurements, dt_total):
        """Preintegrate IMU measurements between two keyframes"""
        if not imu_measurements:
            return np.zeros(3), np.zeros(3), np.eye(3)
        
        delta_p = np.zeros(3)  # Position change
        delta_v = np.zeros(3)  # Velocity change
        delta_R = np.eye(3)    # Rotation change
        
        dt_per_measurement = dt_total / len(imu_measurements) if len(imu_measurements) > 0 else dt_total # 15 Hz

        for imu_data in imu_measurements:
            dt = imu_data.get('dt', dt_per_measurement)
            
            # Bias-corrected measurements
            acc = np.array(imu_data['linear_acceleration']) - np.diagonal(np.array(imu_data['linear_acceleration_covariance']).reshape(3,3))
            gyro = np.array(imu_data['angular_velocity']) - np.diagonal(np.array(imu_data['angular_velocity_covariance']).reshape(3,3))

            # Remove gravity from acceleration
            acc = acc - self.gravity

            # Rotate acceleration to world frame
            acc_world = delta_R @ acc 
            
            print(f"IMU Acceleration (world frame): {acc_world}, Gyro: {gyro}, dt: {dt}")
            # Update rotation (integrate angular velocity)
            if np.linalg.norm(gyro) > 1e-8:  # Avoid numerical issues
                delta_R = delta_R @ cv2.Rodrigues(gyro * dt)[0]
            
            delta_R = self.extract_yaw_rotation(delta_R)
            
            # Update velocity and position
            delta_v += acc_world * dt
            delta_p += delta_v * dt + 0.5 * acc_world * dt**2

        return delta_p, delta_v, delta_R

    def fuse_rotations_slerp(self, R_visual, R_imu):
        """Fuse two rotation matrices using SLERP"""
        r_visual = Rotation.from_matrix(R_visual)
        r_imu = Rotation.from_matrix(R_imu)

        # SLERP for rotation
        key_rots = Rotation.from_matrix([R_visual, R_imu])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate at imu_weight position
        r_fused = slerp(self.imu_weight)  
        
        return r_fused.as_matrix()
    
    def fuse_rotations_log(self, R_visual, R_imu):
        """Fuse two rotation matrices using logarithmic map"""
        r_visual = Rotation.from_matrix(R_visual)
        r_imu = Rotation.from_matrix(R_imu)

        # Logarithmic map to get rotation vectors
        log_visual = r_visual.as_rotvec()
        log_imu = r_imu.as_rotvec()

        # Weighted average of rotation vectors
        log_fused = (1 - self.imu_weight) * log_visual + self.imu_weight * log_imu
        
        # Convert back to rotation matrix
        r_fused = Rotation.from_rotvec(log_fused)
        
        return r_fused.as_matrix()
    
    def fuse_visual_imu(self, R_visual, t_visual, R_imu, t_imu, fusion_method="log"):
        """Fuse visual and IMU estimates using weighted combination"""
        
        try:
            if fusion_method == "slerp":
                R_fused = self.fuse_rotations_slerp(R_visual, R_imu)
            elif fusion_method == "log":
                R_fused = self.fuse_rotations_log(R_visual, R_imu)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        except Exception as e:
            print(f"Rotation fusion failed: {e}, using visual rotation only")
            R_fused = R_visual

        # Simple weighted average for translation
        t_fused = (1 - self.imu_weight) * t_visual.flatten() + self.imu_weight * t_imu
        
        return R_fused, t_fused.reshape(-1, 1)

    def estimate_bias(self, imu_measurements, stationary_threshold=0.1):
        """Estimate IMU biases during stationary periods"""
        if len(imu_measurements) < 50:  # Need sufficient measurements
            return
        
        acc_measurements = np.array([imu['linear_acceleration'] for imu in imu_measurements])
        gyro_measurements = np.array([imu['angular_velocity'] for imu in imu_measurements])
        
        # Check if IMU is stationary (low variance)
        if (np.var(acc_measurements, axis=0).max() < stationary_threshold and
            np.var(gyro_measurements, axis=0).max() < stationary_threshold):
            
            # Update biases
            self.imu_bias_acc = np.mean(acc_measurements, axis=0) - np.array([0, 0, 9.81])
            self.imu_bias_gyro = np.mean(gyro_measurements, axis=0)

def motion_estimation_with_imu(keypoint_left_first, keypoint_left_next, intrinsic_matrix, 
                              config, depth, imu_measurements=None, timestamp=None, vio=None):
    """Enhanced motion estimation with IMU integration"""
    
    # Original visual motion estimation
    try:
        R_visual, t_visual, inlier_pts1, inlier_pts2 = motion_estimation(
            keypoint_left_first, keypoint_left_next, intrinsic_matrix, config, depth)
    except Exception as e:
        print(f"Visual motion estimation failed: {e}")
        # Return identity transformation if visual fails
        return np.eye(3), np.zeros((3, 1)), keypoint_left_first[:10], keypoint_left_next[:10]
    
    # If no VIO system or IMU data, return visual estimate
    if vio is None or not imu_measurements or timestamp is None:
        return R_visual, t_visual, inlier_pts1, inlier_pts2
    
    # IMU preintegration
    if vio.last_timestamp is not None:
        dt = (timestamp - vio.last_timestamp) * 1e-9  # Convert to seconds 
        if dt > 0:  # Valid time difference
            delta_p_imu, delta_v_imu, delta_R_imu = vio.preintegrate_imu(imu_measurements, dt)
            # Print debug information
            print("Visual motion estimation:") 
            print(f"R_visual:\n{R_visual}\nt_visual:\n{t_visual}")
            print("IMU preintegration results:")
            print(f"delta_p_imu: {delta_p_imu}, delta_v_imu: {delta_v_imu}, delta_R_imu:\n{delta_R_imu}")
            # Sensor fusion
            R_fused, t_fused = vio.fuse_visual_imu(R_visual, t_visual, delta_R_imu, delta_p_imu)
            print("Fused motion estimation:")
            print(f"R_fused:\n{R_fused}\nt_fused:\n{t_fused}")
            # Update velocity estimate  
            vio.current_velocity += delta_v_imu
            
            vio.last_timestamp = timestamp
            return R_fused, t_fused, inlier_pts1, inlier_pts2
    
    # First frame or invalid timestamp
    vio.last_timestamp = timestamp
    return R_visual, t_visual, inlier_pts1, inlier_pts2

def visual_inertial_odometry(data_handler, config, precomputed_depth_maps=True, 
                           plot=True, plotframes=False, verbose=True):
    '''
    Enhanced visual odometry with IMU integration.
    
    Parameters:
    - data_handler: interface for accessing stereo images and camera calibration
    - config: configuration dictionary with parameters (detector type, thresholds, etc.)
    - imu_data: list of IMU measurements with timestamps
    - precomputed_depth_maps: whether to use precomputed depth maps
    - plot: whether to display trajectory and map overlays
    - plotframes: whether to show current frame side-by-side with trajectory
    - verbose: whether to print debug and info messages
    '''
    
    # Declare Necessary Variables
    name = config['data']['type']
    detector = config['parameters']['detector']
    depth_model = config['parameters']['depth_model']

    # Load IMU data if available
    imu_file = f'../datasets/BIEL/{name}/imu_data.txt'
    imu_data = load_sequential_data(imu_file, imu=True) if os.path.exists(imu_file) else None
    use_imu = imu_data is not None
    
    num_frames = data_handler.frames

    if plot:
        if name == "KITTI":
            max_lat = 48.987
            min_lat = 48.980
            max_lon = 8.3967
            min_lon = 8.388
            zone_number = 32
        elif name == "01" or name == "00":
            max_lat = 41.384280
            min_lat = 41.381470
            max_lon = 2.117390
            min_lon = 2.114900
            zone_number = 31

        if plotframes:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            _, ax1 = plt.subplots(figsize=(10, 10))

        # Setup map visualization
        proj_utm = Proj(proj="utm",zone=zone_number, ellps="WGS84",preserve_units=False)
        extent_utm = ExtentUTM(min_lon, max_lon, min_lat, max_lat, zone_number, proj_utm)
        extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False)
        tilemapbase.start_logging()
        tilemapbase.init(create=True)
        tiles = tilemapbase.tiles.build_OSM()
        plotter1 = tilemapbase.Plotter(extent_utm_sq, tiles, width=600)
        plotter1.plot(ax1, tiles)

    # Initialize pose parameters
    if name == "KITTI":
        initial_point = (455395.37362745, 5425694.47262261)
        angle_deg = 29.98458135624834
        zone_number = 32
    elif name == "01":
        initial_point = (426069.90, 4581718.85)
        angle_deg = 15 
        zone_number = 31
    elif name == "00":
        initial_point = (426070.04, 4581718.85)
        angle_deg = -12
        zone_number = 31

    # Create initial homogeneous matrix
    homo_matrix = np.eye(4)
    homo_matrix[0, 3], homo_matrix[2, 3] = initial_point
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = Rotation.from_euler('y', angle_rad).as_matrix()
    homo_matrix[:3, :3] = rotation_matrix

    # Initialize trajectory
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    # Initialize VIO system if IMU data is available
    vio = None
    imu_groups = []
    image_timestamps = []
    
    if use_imu:
        left_intrinsic_matrix, *_ = decomposition(data_handler.P0)
        vio = VisualInertialOdometry(config, left_intrinsic_matrix)
        
        # Extract timestamps and group IMU measurements
        image_timestamps = [i for i in range(num_frames)]  # Placeholder timestamps
        if hasattr(data_handler, 'timestamps'):
            image_timestamps = data_handler.timestamps
        
        # Group IMU measurements between consecutive frames
        for i in range(len(image_timestamps) - 1):
            t_start = image_timestamps[i]
            t_end = image_timestamps[i + 1]
            
            imu_segment = [imu for imu in imu_data 
                          if t_start <= imu['timestamp'] < t_end] # len = 1, i.e., one IMU read for image
            imu_groups.append(imu_segment)
    
        print(f"VIO initialized with {len(imu_data)} IMU measurements")

    if data_handler.low_memory:
        data_handler.reset_frames()
        next_image = next(data_handler.left_images)

    # Load OSM street data
    initial_point_latlon = utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    if plot:
        edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
        road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
        walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)

    # Initialize components
    if not precomputed_depth_maps and depth_model != "ZED":
        sde = StereoDepthEstimator(config, data_handler.P0, data_handler.P1)
    
    featureMatcher = FeatureMatcher(config)

    # Main processing loop
    iterator = range(num_frames - 1)
    if not verbose:
        iterator = tqdm(iterator, desc=f"Processing frames {'with IMU' if use_imu else 'visual only'}")

    for i in iterator:
        # Load images
        if data_handler.low_memory:
            image_left = next_image
            image_right = next(data_handler.right_images)
            next_image = next(data_handler.left_images)
        else:
            image_left = data_handler.left_images[i]
            image_right = data_handler.right_images[i]
            next_image = data_handler.left_images[i+1]
        
        # Load or compute depth map
        if precomputed_depth_maps:           
            if depth_model == "ZED":
                depth_map_path = os.path.join(f"../datasets/BIEL/{name}/depths/depth_map_{i}.npy")
            else:
                depth_map_path = os.path.join(f"../datasets/predicted/depth_maps/{name}/{depth_model}/", f"depth_map_{i}.npy")
            
            if i == 0:
                print(f"Loading cached depth maps from {os.path.dirname(depth_map_path)}")
            depth = np.load(depth_map_path)
        else:
            depth, _ = sde.estimate_depth(image_left, image_right)
        
        # Feature matching
        keypoint_left_first, keypoint_left_next = featureMatcher.compute(image_left, next_image, i)

        # Enhanced motion estimation with IMU
        left_intrinsic_matrix, *_ = decomposition(data_handler.P0)
        
        # Get IMU measurements for this frame interval
        current_imu_measurements = imu_groups[i] if use_imu and i < len(imu_groups) else None
        current_timestamp = image_timestamps[i+1] if use_imu and i+1 < len(image_timestamps) else None
        
        try:
            rotation_matrix, translation_vector, *_ = motion_estimation_with_imu(
                keypoint_left_first, keypoint_left_next, left_intrinsic_matrix, config, depth,
                current_imu_measurements, current_timestamp, vio)
        except Exception as e:
            if verbose:
                print(f"Motion estimation failed at frame {i}: {e}")
            # Use identity transformation as fallback
            rotation_matrix = np.eye(3)
            translation_vector = np.zeros((3, 1))

        # Create transformation matrix
        Transformation_matrix = np.eye(4)
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        # Update global pose
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        trajectory[i+1, :, :] = homo_matrix[:3, :]
        
        # Periodic IMU bias estimation
        if use_imu and i > 0 and i % 50 == 0:
            recent_imu = [imu for group in imu_groups[max(0, i-10):i] for imu in group]
            vio.estimate_bias(recent_imu)
            if verbose:
                print(f"Updated IMU biases - Acc: {vio.imu_bias_acc}, Gyro: {vio.imu_bias_gyro}")
        
        # Visualization
        if plot:
            xs = trajectory[:i+2, 0, 3]
            zs = trajectory[:i+2, 2, 3]

            if i == 0:
                label = f"{detector} + IMU" if use_imu else detector
                ax1.plot(xs, zs, c='crimson', label=label)
            else:
                ax1.plot(xs, zs, c='crimson')

            ax1.set_title("Estimated Trajectory")
            ax1.legend()

            if plotframes:
                ax2.imshow(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB))
                ax2.set_title(f"Current Frame: {i}")
                ax2.axis("off")

            plt.pause(1e-32)

        # Periodic status updates
        if i % 10 == 0 and verbose:
            distance_to_ini = np.sqrt((homo_matrix[0, 3] - initial_point[0])**2 + 
                            (homo_matrix[2, 3] - initial_point[1])**2)
            imu_status = f", IMU velocity: {np.linalg.norm(vio.current_velocity if vio else 0):.3f}m/s" if use_imu else ""
            print(f"Frame {i}: Distance from start: {distance_to_ini:.2f}m, "
                  f"Translation: {np.linalg.norm(translation_vector):.4f}{imu_status}")

    # Final visualization and output
    if plot:
        suffix = "_VIO" if use_imu else ""
        plt.savefig(f'../datasets/predicted/figures/{name}_{detector}{suffix}.png')
        plt.show()

    return trajectory

# Backward compatibility - keep original function name
def visual_odometry(data_handler, config, precomputed_depth_maps=True, plot=True, plotframes=False, verbose=True):
    """Original visual odometry function - calls VIO without IMU data"""
    return visual_inertial_odometry(data_handler, config, imu_data=None, 
                                   precomputed_depth_maps=precomputed_depth_maps,
                                   plot=plot, plotframes=plotframes, verbose=verbose)