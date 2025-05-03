import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import geopandas as gpd
import os, sys
import torch

current_dir = os.path.dirname(__file__)
lightglue_path = os.path.abspath(os.path.join(current_dir, '..', 'LightGlue'))
sys.path.append(lightglue_path)
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

# Root Mean Squared Error function
def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the root mean squared error between true and predicted values.

    :params y_true: array of true values
    :params y_pred: array of predicted values
    :return: root mean squared error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

############################################ UTM <-> lat, lon ################################################

def latlon_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.
    Returns: (easting, northing, zone_number, zone_letter)
    """
    # Determine the UTM zone number
    zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'

    # Create a transformer for lat/lon to UTM
    proj_str = f"+proj=utm +zone={zone_number} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("epsg:4326", proj_str, always_xy=True)
    
    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone_number, hemisphere

def utm_to_latlon(easting, northing, zone_number, hemisphere='north'):
    """
    Convert UTM coordinates to latitude and longitude.
    """
    proj_str = f"+proj=utm +zone={zone_number} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(proj_str, "epsg:4326", always_xy=True)
    
    lon, lat = transformer.transform(easting, northing)
    return (lat, lon)

############################################ Stereo Depth Estimation #########################################

def disparity_mapping(left_image, right_image, rgb_value):
    '''
    Takes a stereo pair of images from the sequence and
    computes the disparity map for the left image.

    :params left_image: image from left camera
    :params right_image: image from right camera

    '''

    if rgb_value:
        num_channels = 3
    else:
        num_channels = 1

    # Empirical values collected from a OpenCV website
    num_disparities = 6*16
    block_size = 7

    # Using SGBM matcher(Hirschmuller algorithm) (Read about this!)
    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * num_channels * block_size ** 2,
                                    P2=32 * num_channels * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )

    if rgb_value:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Disparity map
    left_image_disparity_map = matcher.compute(
        left_image, right_image).astype(np.float32)/16

    return left_image_disparity_map


# Decompose camera projection Matrix
def decomposition(p):
    '''
    :params p: camera projection matrix

    '''
    # Decomposing the projection matrix
    intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(
        p)

    # Scaling and removing the homogenous coordinates
    translation_vector = (translation_vector / translation_vector[3])[:3]

    return intrinsic_matrix, rotation_matrix, translation_vector


# Calculating depth information
def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified):
    '''

    :params left_disparity_map: disparity map of left camera
    :params left_intrinsic: intrinsic matrix for left camera
    :params left_translation: translation vector for left camera
    :params right_translation: translation vector for right camera

    '''
    # Focal length of x axis for left camera
    focal_length = left_intrinsic[0][0]

    # Calculate baseline of stereo pair
    if rectified:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]

    # Avoid instability and division by zero
    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    # depth_map = f * b/d
    depth_map = np.ones(left_disparity_map.shape)
    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map


def stereo_depth(left_image, right_image, P0, P1, config):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. 

    :params left_image: image from left camera
    :params right_image: image from right camera
    :params P0: Projection matrix for the left camera
    :params P1: Projection matrix for the right camera

    '''
    rgb_value = config['parameters']['rgb']
    rectified = config["parameters"]["rectified"]

    # First we compute the disparity map
    disp_map = disparity_mapping(left_image,
                                 right_image,
                                 rgb_value)

    # Then decompose the left and right camera projection matrices
    l_intrinsic, _, l_translation = decomposition(
        P0)
    _, _, r_translation = decomposition(
        P1)

    # Calculate depth map for left camera
    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation, rectified)

    return depth


############################################ Stereo Depth Estimation #########################################


######################################### Feature Extraction and Matching ####################################

def feature_extractor(image, detector, mask=None):
    """
    provide keypoints and descriptors

    :params image: image from the dataset

    """
    if detector == 'sift':
        create_detector = cv2.SIFT_create()
    elif detector == 'orb':
        create_detector = cv2.ORB_create()
    else:
        raise ValueError("Detector not supported - Only sift, orb and lightglue are supported")

    keypoints, descriptors = create_detector.detectAndCompute(image, mask)

    return keypoints, descriptors


def BF_matching(first_descriptor, second_descriptor, k=2,  distance_threshold=1.0):
    """
    Brute-Force match features between two images.

    """
    # Using BFMatcher to match the features
    feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)

    # Obtain the matches    
    matches = feature_matcher.knnMatch(
        first_descriptor, second_descriptor, k=k) # Returns the k best matches

    # Filtering out the weak features
    filtered_matches = []
    for match1, match2 in matches:
        if match1.distance <= distance_threshold * match2.distance:
            filtered_matches.append(match1)

    if len(filtered_matches) < 4:
        filtered_matches= sorted(matches[0], key=lambda x:x.distance)[:4]
        print("Matches not filtered!")

    return filtered_matches

def feature_matching(image_left, next_image, mask, config, data_handler, plot, idx, show=False):

    name = config['data']['type']
    detector = config['parameters']['detector']
    threshold = config['parameters']['threshold']

    # In BF the threshold is a distance threshold for the matches
    # In LightGlue the threshold is the max number of keypoints for the SuperPoint() detector

    if detector != 'lightglue':

        # Keypoints and Descriptors of two sequential images of the left camera
        keypoint_left_first, descriptor_left_first = feature_extractor(
            image_left, detector, mask)
        keypoint_left_next, descriptor_left_next = feature_extractor(
            next_image, detector, mask)

        # Use feature detector to match features
        matches = BF_matching(descriptor_left_first,
                                descriptor_left_next,
                                distance_threshold=threshold)
        # Visualize and save the matches between left and right images.
        if not plot:
            if idx % 100 == 0:
                show_matches = cv2.drawMatches(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB), keypoint_left_first, cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB), keypoint_left_next, matches, None, flags=2)
                plt.figure(figsize=(15, 5), dpi=100)
                plt.imshow(show_matches)
                plt.title(f"Matches using {detector} extractor and BFMatcher. Frames {idx} and {idx+1}.")
                if show:
                    plt.show()
                save_dir = f"../datasets/predicted/matches/{name}{detector}_{threshold}"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"matches_{idx}.png"))
    
    else:
        # LightGlue feature matching [If this works, the other feature extractors can be removed, and the code can be simplified]
        image0 = load_image(data_handler.sequence_dir + 'image_0/' + data_handler.left_camera_images[idx])
        image1 = load_image(data_handler.sequence_dir + 'image_0/' + data_handler.left_camera_images[idx+1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        extractor = SuperPoint(max_num_keypoints=threshold).eval().to(device)  # load the extractor
        matcher = LightGlue(features="superpoint").eval().to(device)

        descriptor_left_first = extractor.extract(image0.to(device))
        descriptor_left_next = extractor.extract(image1.to(device))
        matches01 = matcher({
            "image0": descriptor_left_first, 
            "image1": descriptor_left_next
            })
        
        descriptor_left_first, descriptor_left_next, matches01 = [
            rbd(x) for x in [descriptor_left_first, descriptor_left_next, matches01]
        ]  # remove batch dimension

        # Obtain the keypoints
        pre_keypoint_left_first, pre_keypoint_left_next, matches = descriptor_left_first["keypoints"], descriptor_left_next["keypoints"], matches01["matches"]

        # Filter the keypoints that are matched
        keypoint_left_first, keypoint_left_next = pre_keypoint_left_first[matches[..., 0]], pre_keypoint_left_next[matches[..., 1]]

        # Visualize and save the matches between left and right images.
        if not plot:
            if idx % 100 == 0:
                # Plot the matches and the stopping layer
                _ = viz2d.plot_images([image0, image1])
                viz2d.plot_matches(keypoint_left_first, keypoint_left_next, color="lime", lw=0.2)
                viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
                plt.title(f"Matches using LightGlue. Frames {idx} and {idx+1}")
                if show:
                    plt.show()
                save_dir = f"../datasets/predicted/matches/{name}/{detector}_{threshold}"
                os.makedirs(save_dir, exist_ok=True)
                viz2d.save_plot(os.path.join(save_dir, f"matches_{idx}.png"))
        
    return keypoint_left_first, descriptor_left_first, keypoint_left_next, descriptor_left_next, matches


######################################### Feature Extraction and Matching ####################################


######################################### Motion Estimation ####################################
def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, config):
    """
    Estimating motion of the left camera from sequential imgaes 

    """

    max_depth = config['parameters']['max_depth']
    detector = config['parameters']['detector']
    
    # Initialize the rotation matrix and translation vector
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    if detector != 'lightglue':
        # Only considering keypoints that are matched for two sequential frames
        image1_points = np.float32(
            [firstImage_keypoints[m.queryIdx].pt for m in matches])
        image2_points = np.float32(
            [secondImage_keypoints[m.trainIdx].pt for m in matches])
    else:
        # This step is already done in the feature matching function
        image1_points = np.float32(firstImage_keypoints.cpu())
        image2_points = np.float32(secondImage_keypoints.cpu())    

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)] # From the depth map 

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

        # HERE ADD A FUNCTION THAT USES FAR POINTS FOR ROTATION AND CLOSE POINTS FOR TRANSLATION

    # Deleting the false depth points if possible
    if len(image2_points) - len(outliers) > 4:
        image1_points = np.delete(image1_points, outliers, 0)
        image2_points = np.delete(image2_points, outliers, 0)
    else:    
        print("Outliers not removed!")

    # Apply RRANSAC Algorithm: matching robust to outliers and obtaing rotation and translation
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    # Convert the rotation vector to a rotation matrix
    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

######################################### Motion Estimation ####################################


######################################### In street checker ####################################
import geopandas as gpd
from shapely.geometry import Point

def in_street_checker(edges, walkable_area, crossing_area, point, point_area):
    '''
    Function that checks if the point is on the graph and the distance to the closest edge of the graph
    '''
    # Ensure all geometries are in the same CRS
    if walkable_area.crs != edges.crs:
        walkable_area = walkable_area.to_crs(edges.crs)
    if point_area.crs != edges.crs:
        point_area = point_area.to_crs(edges.crs)

    # Reset the index of point_area to ensure alignment
    point_area = point_area.reset_index(drop=True)

    # Convert walkable_area to a GeoDataFrame
    walkable_area_gdf = gpd.GeoDataFrame(geometry=walkable_area.geometry, crs=walkable_area.crs)

    # Perform a spatial join to check if the point is inside the walkable area
    point_gdf = gpd.GeoDataFrame(geometry=point_area)

    # Check if the point is inside the walkable area
    is_inside = walkable_area_gdf.geometry.apply(lambda geom: geom.contains(point_area.iloc[0])).any()

    # Check if the point is inside the crossing area
    is_inside_crossing = crossing_area.geometry.apply(lambda geom: geom.contains(point_area.iloc[0])).any()

    # Update is_inside to be true if the point is inside either the walkable area or the crossing area
    is_inside = is_inside or is_inside_crossing

    # Calculate the minimum distance to the nearest crossing
    distance = crossing_area.distance(point).min()

    return is_inside, distance

######################################### In street checker ####################################