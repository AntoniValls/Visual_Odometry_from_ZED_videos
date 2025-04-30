import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import geopandas as gpd

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


def stereo_depth(left_image, right_image, P0, P1, rgb_value, rectified):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. 

    :params left_image: image from left camera
    :params right_image: image from right camera
    :params P0: Projection matrix for the left camera
    :params P1: Projection matrix for the right camera

    '''
    # First we compute the disparity map
    disp_map = disparity_mapping(left_image,
                                 right_image,
                                 rgb_value)

    # Then decompose the left and right camera projection matrices
    l_intrinsic, l_rotation, l_translation = decomposition(
        P0)
    r_intrinsic, r_rotation, r_translation = decomposition(
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

    keypoints, descriptors = create_detector.detectAndCompute(image, mask)

    return keypoints, descriptors


def feature_matching(first_descriptor, second_descriptor, detector, k=2,  distance_threshold=1.0):
    """
    Match features between two images

    """

    if detector == 'sift':
        feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
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


def visualize_matches(first_image, second_image, keypoint_one, keypoint_two, matches):
    """
    Visualize corresponding matches in two images

    """
   
    show_matches = cv2.drawMatches(
        first_image, keypoint_one, second_image, keypoint_two, matches, None, flags=2)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.imshow(show_matches)
    plt.show()

######################################### Feature Extraction and Matching ####################################


######################################### Motion Estimation ####################################
def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, max_depth):
    """
    Estimating motion of the left camera from sequential imgaes 

    """
    print(max_depth)

    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    # Only considering keypoints that are matched for two sequential frames
    image1_points = np.float32(
        [firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32(
        [secondImage_keypoints[m.trainIdx].pt for m in matches])

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)] # From the depth map (somehow this gets negative values)

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    # Deleting the false depth points if possible
    if len(image2_points) - len(outliers) > 4:
        image1_points = np.delete(image1_points, outliers, 0)
        image2_points = np.delete(image2_points, outliers, 0)

    # Apply Ransac Algorithm 
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