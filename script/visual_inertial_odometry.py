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
from BielGlasses.script.ZED_motion import load_sequential_data
from segmentation_utils import street_segmentation
from scipy.spatial.transform import Rotation, Slerp
from pyproj import Proj
from tqdm import tqdm
from collections import deque

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'modules/mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM 

"""
Extended version of the visual_odometry.py code for implementing info extracted from the IMU. It uses the gyroscope and acceleration data
to compute the affine transformation matrices and averages them with the ones obtained by the simple VO model.

NOTE: It doesn't work well, either because the IMU data is shitty or because the implementation is wrong (possibly the second one).
"""

class VisualInertialOdometry:
   
    def __init__(self, config, intrinsic_matrix):
        self.config = config
        self.intrinsic_matrix = intrinsic_matrix
        
        # IMU-related parameters
        self.gravity = np.array([0, -9.81, 0])  # Gravity vector in IMU frame
        
        # State variables
        self.current_velocity = np.zeros(3)  # Current velocity in world frame
        self.current_position = np.zeros(3)  # Current position in world frame
        self.current_orientation = np.eye(3)  # Current orientation matrix
        self.last_timestamp = None
        
        # IMU biases (will be estimated)
        self.imu_bias_acc = np.zeros(3)
        self.imu_bias_gyro = np.zeros(3)
        
        # Fusion weights imu_weight + visual_weight = 1
        self.imu_weight = config.get('imu_weight', 0.05)  # Reduced IMU weight initially
        
        # Debug info
        self.debug_info = []

    def transform_imu_to_camera(self, imu_measurement):
        """
        Transform IMU measurements from IMU frame to camera frame.
        This is a critical step that's often overlooked.
        
        Typical transformation for forward-facing camera:
        - IMU X (forward) -> Camera Z (forward)  
        - IMU Y (left) -> Camera -X (left)
        - IMU Z (up) -> Camera -Y (up)
        """
        # This transformation matrix should be calibrated for your specific setup
        # For now, assuming a common configuration
        # T_cam_imu = np.array([
        #     [0, -1, 0],   # IMU Y -> Camera -X
        #     [0, 0, -1],   # IMU Z -> Camera -Y  
        #     [1, 0, 0]     # IMU X -> Camera Z
        # ])
        T_cam_imu = np.eye(3)
        
        acc_cam = T_cam_imu @ np.array(imu_measurement['linear_acceleration'])
        gyro_cam = T_cam_imu @ np.array(imu_measurement['angular_velocity'])
        
        return {
            'linear_acceleration': acc_cam,
            'angular_velocity': gyro_cam,
            'timestamp': imu_measurement['timestamp']
        }
    
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
        
        # Initialize preintegration variables
        delta_p = np.zeros(3)  # Position change in world frame
        delta_v = np.zeros(3)  # Velocity change in world frame
        delta_R = np.eye(3)    # Rotation change
        
        # Keep initial state constant during integration
        initial_R = self.current_orientation.copy()
        initial_v = self.current_velocity.copy()
        
        # Integration state (can be modified)
        current_R = initial_R.copy()
        accumulated_v = initial_v.copy()  # Track velocity changes separately

        dt_per_measurement = dt_total / len(imu_measurements) if len(imu_measurements) > 0 else dt_total
        
        print(f"\n=== IMU Preintegration Debug ===")
        print(f"Total dt: {dt_total:.4f}s, Measurements: {len(imu_measurements)}")
        print(f"dt per measurement: {dt_per_measurement:.4f}s")
        
        for idx, imu_data in enumerate(imu_measurements):
            # Transform IMU data to camera frame
            imu_cam = self.transform_imu_to_camera(imu_data)
            
            dt = imu_data.get('dt', dt_per_measurement)
            
            acc = np.array(imu_cam['linear_acceleration']) 
            gyro = np.array(imu_cam['angular_velocity']) 
            
            # Add low-pass filtering for noise reduction
            if hasattr(self, 'prev_acc'):
                # Simple exponential smoothing
                alpha = 0.3  # Adjust based on your needs
                acc = alpha * acc + (1 - alpha) * self.prev_acc
                gyro = alpha * gyro + (1 - alpha) * self.prev_gyro
            
            self.prev_acc = acc.copy()
            self.prev_gyro = gyro.copy()

            # Print raw measurements for debugging
            if idx == 0:
                print(f"Raw acc: {imu_data['linear_acceleration']}")
                print(f"Raw gyro: {imu_data['angular_velocity']}")
                print(f"Transformed acc: {acc}")
                print(f"Transformed gyro: {gyro}")
                print(f"Bias acc: {self.imu_bias_acc}")
                print(f"Bias gyro: {self.imu_bias_gyro}")
                
            # Integrate rotation first
            if np.linalg.norm(gyro) > 1e-6:
                # Small angle approximation for rotation
                gyro_norm = np.linalg.norm(gyro)
                if gyro_norm * dt < 0.1:  # Small angle
                    skew_gyro = np.array([
                        [0, -gyro[2], gyro[1]],
                        [gyro[2], 0, -gyro[0]],
                        [-gyro[1], gyro[0], 0]
                    ])
                    dR = np.eye(3) + skew_gyro * dt + 0.5 * (skew_gyro @ skew_gyro) * dt**2
                else:
                    # Use Rodrigues formula for larger rotations
                    dR = cv2.Rodrigues(gyro * dt)[0]
                
                dR = self.extract_yaw_rotation(dR)
                delta_R = delta_R @ dR
                current_R = current_R @ dR
            
            # Transform acceleration to world frame and remove gravity
            acc_world = current_R @ acc
            gravity_world = np.array([0, -9.81, 0])  
            acc_world_corrected = acc_world - gravity_world

            # Proper integration of position and velocity    
            delta_p += accumulated_v * dt + 0.5 * acc_world_corrected * dt**2
            delta_v += acc_world_corrected * dt
            accumulated_v += acc_world_corrected * dt
            
            if idx == 0:
                print(f"Acc world: {acc_world}")
                print(f"Acc world corrected: {acc_world_corrected}")
                print(f"Delta v step: {acc_world_corrected * dt}")
                print(f"Delta p step: {accumulated_v * dt + 0.5 * acc_world_corrected * dt**2}")

        print(f"Final Δp: {delta_p}")
        print(f"Final Δv: {delta_v}")
        print(f"Final ΔR det: {np.linalg.det(delta_R):.6f}")
        print("=== End IMU Debug ===\n")

        # Update internal state
        # self.current_velocity += delta_v
        self.current_orientation = current_R
        self.current_position += delta_p
        
        # Store debug info
        self.debug_info.append({
            'delta_p': delta_p.copy(),
            'delta_v': delta_v.copy(),
            'delta_R': delta_R.copy(),
            'dt_total': dt_total,
            'num_measurements': len(imu_measurements)
        })

        return delta_p, delta_v, delta_R

    def fuse_rotations_slerp(self, R_visual, R_imu):
        """Fuse two rotation matrices using SLERP"""
        try:
            # SLERP for rotation
            key_rots = Rotation.from_matrix([R_visual, R_imu])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            
            # Interpolate at imu_weight position
            r_fused = slerp(self.imu_weight)  
            
            return r_fused.as_matrix()
        except Exception as e:
            print(f"SLERP fusion failed: {e}")
            return R_visual
    
    def fuse_rotations_log(self, R_visual, R_imu):
        """Fuse two rotation matrices using logarithmic map"""
        try:
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
        except Exception as e:
            print(f"Log fusion failed: {e}")
            return R_visual
    
    def fuse_visual_imu(self, R_visual, t_visual, R_imu, t_imu, fusion_method="log"):
        """Fuse visual and IMU estimates using weighted combination"""
        
        print(f"\n=== Fusion Debug ===")
        print(f"Visual t: {t_visual.flatten()}")
        print(f"IMU t: {t_imu}")
        print(f"Visual t norm: {np.linalg.norm(t_visual):.6f}")
        print(f"IMU t norm: {np.linalg.norm(t_imu):.6f}")
        print(f"Visual R: \n{R_visual}")   
        print(f"IMU R: \n{R_imu}")

        # Check for reasonable scale differences
        visual_scale = np.linalg.norm(t_visual)
        imu_scale = np.linalg.norm(t_imu)
        
        if visual_scale > 0 and imu_scale > 0:
            scale_ratio = visual_scale / imu_scale
            print(f"Scale ratio (visual/imu): {scale_ratio:.2f}")
            
            # If IMU scale is much smaller, it might need scaling
            if scale_ratio > 100:
                print("WARNING: IMU translation seems too small, possibly coordinate frame issue")
            elif scale_ratio < 0.01:
                print("WARNING: IMU translation seems too large, possibly integration error")
        
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
        
        print(f"Fused t: {t_fused}")   
        print(f"Fused R: \n{R_fused}")
        print(f"Fusion weights: visual={1-self.imu_weight:.2f}, imu={self.imu_weight:.2f}")
        print("=== End Fusion Debug ===\n")

        return R_fused, t_fused.reshape(-1, 1) 

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
    if vio is None or not imu_measurements:
        return R_visual, t_visual, inlier_pts1, inlier_pts2
    
    # IMU preintegration
    if vio.last_timestamp is not None and timestamp is not None:
        # Calculate time difference
        if hasattr(timestamp, '__iter__'):
            current_time = timestamp
        else:
            current_time = timestamp
            
        if hasattr(vio.last_timestamp, '__iter__'):
            last_time = vio.last_timestamp
        else:
            last_time = vio.last_timestamp
            
        # Handle different timestamp formats
        if isinstance(current_time, (int, float)) and isinstance(last_time, (int, float)):
            dt = (current_time - last_time) * 1e-9

        else:
            dt = 1.0/15.0  # Default fallback 15Hz
            
        print(f"Timestamp info: current={current_time}, last={last_time}, dt={dt:.4f}s")
        
        if dt > 0 and dt < 1.0:  # Reasonable time difference (less than 1 second)
            # Let VO declare the velocity
            vio.current_velocity = t_visual.flatten() / dt

            delta_p_imu, delta_v_imu, delta_R_imu = vio.preintegrate_imu(imu_measurements, dt)
           
            # Sensor fusion
            R_fused, t_fused = vio.fuse_visual_imu(R_visual, t_visual, delta_R_imu, delta_p_imu)
          
            vio.last_timestamp = timestamp
            return R_fused, t_fused, inlier_pts1, inlier_pts2
        else:
            print(f"Invalid time difference: {dt:.4f}s, using visual only")
    
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
    
    if use_imu:
        print(f"Loaded {len(imu_data)} IMU measurements")
        # Print first few IMU measurements for debugging
        for i, imu in enumerate(imu_data[:3]):
            print(f"IMU {i}: acc={imu['linear_acceleration']}, gyro={imu['angular_velocity']}, t={imu.get('timestamp', 'N/A')}")
    
    num_frames = data_handler.frames

    if name == "KITTI":
        # Hardcoded for the KITTI dataset max_lat, min_lat, max_lon, min_lon
        angle_deg = 29.98458135624834
        max_lat = 48.987
        min_lat = 48.980
        max_lon = 8.3967
        min_lon = 8.388
        zone_number = 32

    else:
        # Load the sequence parameters
        max_lat, min_lat, max_lon, min_lon, zone_number, initial_point, angle_deg = GT_reader(name)
    
    if plot:

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
        image_timestamps = list(range(num_frames))  # Default frame numbers
        if hasattr(data_handler, 'timestamps'):
            image_timestamps = data_handler.timestamps
        elif hasattr(data_handler, 'image_timestamps'):
            image_timestamps = data_handler.image_timestamps
            
        print(f"Image timestamps type: {type(image_timestamps[0]) if image_timestamps.any() else 'None'}")
        print(f"First few timestamps: {image_timestamps[:5] if len(image_timestamps) >= 5 else image_timestamps}")
        
        # Group IMU measurements between consecutive frames
        for i in range(len(image_timestamps) - 1):
            t_start = image_timestamps[i]
            t_end = image_timestamps[i + 1]
            
            # Handle different timestamp formats
            if isinstance(t_start, (int, float)) and len(imu_data) > 0:
                if hasattr(imu_data[0], 'get') and 'timestamp' in imu_data[0]:
                    imu_segment = [imu for imu in imu_data 
                                  if t_start <= imu['timestamp'] < t_end]
                else:
                    # Fallback: distribute IMU measurements evenly
                    imu_per_frame = len(imu_data) // (num_frames - 1)
                    start_idx = i * imu_per_frame
                    end_idx = min((i + 1) * imu_per_frame, len(imu_data))
                    imu_segment = imu_data[start_idx:end_idx]
            else:
                # Fallback: single IMU measurement per frame
                if i < len(imu_data):
                    imu_segment = [imu_data[i]]
                else:
                    imu_segment = []
                    
            imu_groups.append(imu_segment)
            
        print(f"IMU groups created: {len(imu_groups)} groups")
        print(f"Measurements per group (first 5): {[len(group) for group in imu_groups[:5]]}")
        
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
            translation_norm = np.linalg.norm(translation_vector)
            print(f"Frame {i}: Distance from start: {distance_to_ini:.2f}m, "
                  f"Translation: {translation_norm:.4f}m{imu_status}")
                  
            # Additional debugging for IMU
            if use_imu and vio and i > 0:
                print(f"  IMU position: {vio.current_position}")
                print(f"  IMU velocity: {vio.current_velocity}")
                print(f"  IMU orientation: {vio.current_orientation}")
                if vio.debug_info:
                    last_debug = vio.debug_info[-1]
                    print(f"  Last IMU delta_p: {last_debug['delta_p']}")

    # Final visualization and output
    if plot:
        suffix = "_VIO" if use_imu else ""
        plt.savefig(f'../datasets/predicted/figures/{name}_{detector}{suffix}.png')
        plt.show()

    return trajectory

# Backward compatibility - keep original function name
def visual_odometry(data_handler, config, precomputed_depth_maps=True, plot=True, plotframes=False, verbose=True):
    """Original visual odometry function - calls VIO without IMU data"""
    return visual_inertial_odometry(data_handler, config, 
                                   precomputed_depth_maps=precomputed_depth_maps,
                                   plot=plot, plotframes=plotframes, verbose=verbose)