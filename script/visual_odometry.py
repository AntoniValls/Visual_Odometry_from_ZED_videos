import os
import sys
import tilemapbase
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from motion_estimation import motion_estimation
from feature_matching import FeatureMatcher
from sde import StereoDepthEstimator
from segmentation_utils import street_segmentation
from scipy.spatial.transform import Rotation
from pyproj import Proj
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM 

# Obtain the UTM projection for the area of interest
#   * UTM projection, zone 32 corresponds to Germany (KITTI)
#   * UTM projection, zone 30 corresponds to Malaga
#   * UTM projection, zone 31 corresponds to Barcelona

def visual_odometry(data_handler, config, precomputed_depth_maps=True, plot=True, plotframes=False, verbose=True):
    '''
    Estimates camera trajectory from stereo image sequences using visual odometry.
    This function computes depth maps, detects and matches features, estimates motion,
    and reconstructs the camera path in 3D space. It supports optional plotting of the
    estimated trajectory, ground truth, and OpenStreetMap overlays.

    Parameters:
    - data_handler: interface for accessing stereo images and camera calibration
    - config: configuration dictionary with parameters (detector type, thresholds, etc.)
    - plot: whether to display trajectory and map overlays
    - plotframes: whether to show current frame side-by-side with trajectory
    - verbose: whether to print debug and info messages
    
    Here the coordinate system is Right Handed, Y-down, Z-FWD
    '''
    # Declare Necessary Variables
    name = config['data']['type']
    detector = config['parameters']['detector']
    depth_model = config['parameters']['depth_model']

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
            # Create a side-by-side plot with trajectory and current frame
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            # Create only one plot
            _, ax1 = plt.subplots(figsize=(10, 10))

        # Use ExtentUTM for map visualization
        proj_utm = Proj(proj="utm",zone=zone_number, ellps="WGS84",preserve_units=False)
        extent_utm = ExtentUTM(min_lon, max_lon, min_lat, max_lat, zone_number, proj_utm)
        extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False) # square aspect ratio
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

    if data_handler.low_memory:
        data_handler.reset_frames()
        next_image = next(data_handler.left_images)

    # Load OSM street data 
    initial_point_latlon =  utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    if plot:
        edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
        road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
        walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)

    # Initialize components
    if not precomputed_depth_maps and depth_model != "ZED":
        sde = StereoDepthEstimator(config, data_handler.P0, data_handler.P1)
    
    featureMatcher= FeatureMatcher(config)
    
    # Main processing loop
    iterator = range(num_frames - 1)
    if not verbose:
        iterator = tqdm(iterator, desc="Processing frames")

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
            else: # Simple or Complex
                depth_map_path = os.path.join(f"../datasets/predicted/depth_maps/{name}/{depth_model}/", f"depth_map_{i}.npy")
            
            if i == 0:
                print(f"Loading cached depth maps from {os.path.dirname(depth_map_path)}")
            depth = np.load(depth_map_path)
        else:
            depth, _ = sde.estimate_depth(image_left, image_right)
        
        # Feature matching
        keypoint_left_first, keypoint_left_next = featureMatcher.compute(image_left, next_image, i)

         # Motion estimation
        left_instrinsic_matrix, *_ = decomposition(data_handler.P0)
        rotation_matrix, translation_vector, *_= motion_estimation(
            keypoint_left_first, keypoint_left_next, left_instrinsic_matrix, config, depth)

        # Create transformation - homogeneous matrix (4X4)
        Transformation_matrix = np.eye(4)
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        # Update global pose
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        trajectory[i+1, :, :] = homo_matrix[:3, :]
        
        # Visualization            
        if plot:
            # Define the trajectory
            xs = trajectory[:i+2, 0, 3]
            zs = trajectory[:i+2, 2, 3]

            # Plot the estimated trajectory
            if i == 0:
                ax1.plot(xs, zs, c='crimson', label=detector)
            else:
                ax1.plot(xs, zs, c='crimson')

            ax1.set_title("Estimated Trajectory")
            ax1.legend()

            if plotframes:
                # Second subplot - current frame
                ax2.imshow(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB))
                ax2.set_title(f"Current Frame: {i}")
                ax2.axis("off")  # Hide axes for the image   

            plt.pause(1e-32)

        # Periodic status updates
        if i % 10 == 0 and verbose:
            distance_to_ini = np.sqrt((homo_matrix[0, 3] - initial_point[0])**2 + 
                            (homo_matrix[2, 3] - initial_point[1])**2)
            print(f"Frame {i}: Distance from start: {distance_to_ini:.2f}m, ",
                f"Current translation: {np.linalg.norm(translation_vector):.4f}")

    
    # Visualization and output
    if plot:
        plt.savefig(f'../datasets/predicted/figures/{name}_{detector}.png')
        plt.show()

    return trajectory
