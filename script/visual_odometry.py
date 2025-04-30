import os
import sys
import tilemapbase
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from gps_utils import align_trajectories
from segmentation_utils import street_segmentation
from scipy.spatial.transform import Rotation
from pyproj import Proj

current_dir = os.path.dirname(__file__)
mapinmeters_path = os.path.abspath(os.path.join(current_dir, '..', 'mapinmeters'))
sys.path.append(mapinmeters_path)
from mapinmeters.extentutm import ExtentUTM # type: ignore

# Obtain the UTM projection for the area of interest
#   * UTM projection, zone 32 corresponds to Germany (KITTI)
#   * UTM projection, zone 30 corresponds to Malaga
#   * UTM projection, zone 31 corresponds to Barcelona

def visual_odometry(data_handler, config, mask=None, plot=True, plotframes=False, verbose=True):
    '''
    Estimates camera trajectory from stereo image sequences using visual odometry.
    This function computes depth maps, detects and matches features, estimates motion,
    and reconstructs the camera path in 3D space. It supports optional plotting of the
    estimated trajectory, ground truth, and OpenStreetMap overlays.

    Parameters:
    - data_handler: interface for accessing stereo images and camera calibration
    - config: configuration dictionary with parameters (detector type, thresholds, etc.)
    - mask: optional mask to filter out parts of the image
    - plot: whether to display trajectory and map overlays
    - plotframes: whether to show current frame side-by-side with trajectory
    - verbose: whether to print debug and info messages
    '''
    # Declare Necessary Variables
    name = config['data']['type']
    detector = config['parameters']['detector']
    subset = config['parameters']['subset']
    threshold = config['parameters']['distance_threshold']
    plot_GT = config['data']['ground_truth']
    max_depth_value = config['parameters']['max_depth']
    rgb_value = config['parameters']['rgb']
    rectified = config["parameters"]["rectified"]


    if subset is not None:
        num_frames = subset
    else:
        num_frames = data_handler.frames

    if plot:
        if name == "KITTI":
            # Hardcoded for the KITTI dataset max_lat, min_lat, max_lon, min_lon
            max_lat = 48.987
            min_lat = 48.980
            max_lon = 8.3967
            min_lon = 8.388
            zone_number = 32

        elif name == "BIEL":
            # Harcoded for the IRI dataset max_lat, min_lat, max_lon, min_lon
            max_lat = 41.384280
            min_lat = 41.381470
            max_lon = 2.117390
            min_lon = 2.114900
            zone_number = 31

        if plotframes:
            # Create a side-by-side plot with trajectory and current frame
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        else:
            # Create only one plot
            _, ax1 = plt.subplots(figsize=(20, 10))

        # Use ExtentUTM
        proj_utm = Proj(proj="utm",zone=zone_number,ellps="WGS84",preserve_units=False)
        extent_utm = ExtentUTM(min_lon, max_lon, min_lat, max_lat, zone_number, proj_utm)
        extent_utm_sq = extent_utm.to_aspect(1.0, shrink=False) # square aspect ratio
        tilemapbase.start_logging()
        tilemapbase.init(create=True)
        tiles = tilemapbase.tiles.build_OSM()
        plotter1 = tilemapbase.Plotter(extent_utm_sq, tiles, width=600)
        plotter1.plot(ax1, tiles)

        # Plot the GT
        if plot_GT:

            # Special case for the IRI sequences, as ground truth has a different format, later should be homogenaized.
            # trajectory_file = '../datasets/poses/00_IRI.txt'
            # latitudes = []
            # longitudes = []
            # with open(trajectory_file, 'r') as file:
            #     # Skip the header line
            #     next(file)
            #     for line in file:
            #         # Split the line by tab
            #         parts = line.strip().split('\t')
            #         # Extract latitude and longitude
            #         lat = float(parts[1])
            #         lon = float(parts[2])
            #         latitudes.append(lat)
            #         longitudes.append(lon)

            # xs,zs = proj_utm(longitudes,latitudes)

            # if plot:
            #     plt.plot(xs,zs,c='darkorange',label='Ground Truth')
            #     plt.title("Ground Truth vs Estimated Trajectory")

            xt = data_handler.ground_truth[:, 0, 3]
            yt = data_handler.ground_truth[:, 1, 3]
            zt = data_handler.ground_truth[:, 2, 3]

            if verbose:
                print(f"X data from ground truth: {xt}")
                print(f"Y data from ground truth: {yt}")
                print(f"Z data from ground truth: {zt}")

            # Correcting the ground truth to the UTM projection
            ground_truth = np.array([xt,zt]).T
            ground_truth_utm = align_trajectories(ground_truth)

            # Plotting range for better visualisation
            ax1.plot(ground_truth_utm[:,0], ground_truth_utm[:,1], c='darkorange', label='Ground Truth')

    # Harcoded first value and angle (in UTM)
    if name == "KITTI": #00
        initial_point = (455395.37362745, 5425694.47262261)
        angle_deg = 29.98458135624834
        zone_number = 32
    elif name == "BIEL":
        initial_point = (426069.90, 4581718.85)
        angle_deg = 15 # The first frame is oriented to the north (x=0)
        zone_number = 31

    # Create a homogeneous matrix
    homo_matrix = np.eye(4)
    homo_matrix[0, 3], homo_matrix[2, 3] = initial_point

    # Relate it to a rotatiion matrix
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = Rotation.from_euler('y', angle_rad).as_matrix()
    homo_matrix[:3, :3] = rotation_matrix

    # 'trajectory' keeps track of the orientation and position at each frame
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    # From projection matrix retrieve the left camera's intrinsic matrix
    left_instrinsic_matrix, _, _ = decomposition(data_handler.P0)

    if data_handler.low_memory:
        data_handler.reset_frames()
        next_image = next(data_handler.left_images)

    # Load OSM street data for the area around the initial point
    initial_point_latlon =  utm_to_latlon(initial_point[0], initial_point[1], zone_number)
    zone = f"+proj=utm +zone={zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # Extract useful data
    edges, road_area, walkable_area, *_ = street_segmentation(initial_point_latlon, zone)

    # Plot the edges, roads and walkable areas
    if plot:
        edges.plot(ax=ax1, linewidth=1, edgecolor="dimgray", label='Graph from OSM')
        road_area.plot(ax=ax1, color="paleturquoise", alpha=0.7)
        walkable_area.plot(ax=ax1, color="lightgreen", alpha=0.7)

    # Loop to iterate all the frames
    for i in range(num_frames - 1,):

        # using generator retrieveing images
        if data_handler.low_memory:
            image_left = next_image
            image_right = next(data_handler.right_images)
            next_image = next(data_handler.left_images)

        # If you set the low memory to False, all your images will be stored in your RAM and you can access like a normal array.
        else:
            image_left = data_handler.left_images[i]
            image_right = data_handler.right_images[i]
            next_image = data_handler.left_images[i+1]

        # Estimating the depth map of an image_left
        depth = stereo_depth(image_left,
                             image_right,
                             P0=data_handler.P0,
                             P1=data_handler.P1,
                             rgb_value=rgb_value,
                             rectified=rectified)

        # Keypoints and Descriptors of two sequential images of the left camera
        keypoint_left_first, descriptor_left_first = feature_extractor(
            image_left, detector, mask)
        keypoint_left_next, descriptor_left_next = feature_extractor(
            next_image, detector, mask)

        # Use feature detector to match features
        matches = feature_matching(descriptor_left_first,
                                   descriptor_left_next,
                                   detector=detector,
                                   distance_threshold=threshold)

        # Visualize the matches between left and right images.
        if not plot:
            visualize_matches(image_left,next_image,keypoint_left_first,keypoint_left_next,matches)

        # Estimate motion between sequential images of the left camera
        rotation_matrix, translation_vector, _, _ = motion_estimation(
            matches, keypoint_left_first, keypoint_left_next, left_instrinsic_matrix, depth, max_depth_value)

        if verbose:
            print(f"Transaltion vector: \n{translation_vector}")
            print(f"Rotation Matrix (Rodrigues): \n{rotation_matrix}")

        # Initialise a homogeneous matrix (4X4)
        Transformation_matrix = np.eye(4)

        # Build the Transformation matrix using rotation matrix and translation vector from motion estimation function
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        # Transformation wrt. world coordinate system
        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        print(initial_point)
        distance_to_ini = np.sqrt((homo_matrix[0, 3] - initial_point[0])**2 + (homo_matrix[2, 3] - initial_point[1])**2)
        print(f"Current point to be in the area is:\nUTM: ({homo_matrix[0, 3]}, {homo_matrix[2, 3]})\nLat, Lon: {utm_to_latlon(homo_matrix[0, 3], homo_matrix[2, 3], zone_number)}")
        print(f"Distance to the initial point: {distance_to_ini:.2f} meters")
        
        ###########################################################
        ###            CORRECTION OF THE TRAJECTORY             ###
        ###########################################################

        # # Check if the current position is within a street and calculate the distance
        # current_position = Point(homo_matrix[0, 3], homo_matrix[2, 3])
        # current_point = gpd.GeoSeries([current_position],crs=f"EPSG:326{zone_number}") 
        # inside_street, distance = in_street_checker(edges, walkable_area, current_position,current_point)

        # # Correct the homo_matrix if the trajectory is outside the street.
        # if not inside_street:
        #     nearest_edge = edges.geometry.distance(current_position).idxmin()
        #     line = edges.geometry[nearest_edge]
        #     projected_point = nearest_points(line, current_position)[0]
        #     homo_matrix[0, 3] = homo_matrix[0,3] - (homo_matrix[0,3] - projected_point.x)
        #     homo_matrix[2, 3] = homo_matrix[2,3] - (homo_matrix[2,3] - projected_point.y)
        #     print(f"After correction X: {projected_point.x}, Y: {projected_point.y}")

        # Append the pose of camera in the trajectory array
        trajectory[i+1, :, :] = homo_matrix[:3, :]

        if i % 10 == 0:
            print(f'{i} frames have been computed')

        if i == num_frames - 2:
            print('All frames have been computed')

        if plot:
            
            # Define the trajectory
            xs = trajectory[:i+2, 0, 3]
            ys = trajectory[:i+2, 2, 3]
            zs = trajectory[:i+2, 1, 3]

            # Plot the estimated trajectory
            if i == 0:
                ax1.plot(xs, ys, c='crimson', label=detector)
            else:
                ax1.plot(xs, ys, c='crimson')
            ax1.set_title("Estimated Trajectory")
            ax1.legend()

            if plotframes:
                # Second subplot - current frame
                ax2.imshow(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB))
                ax2.set_title(f"Current Frame: {i}")
                ax2.axis("off")  # Hide axes for the image   

            plt.pause(1e-32)

    if plot:
        plt.show()
        plt.savefig('../figures/foo.png')


    return trajectory
