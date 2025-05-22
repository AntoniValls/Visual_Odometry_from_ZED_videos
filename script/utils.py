import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import os, sys


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


def decomposition(p):
    '''
    Camera projection matrix decomposition

    Args:
        p: camera projection matrix

    Returns: instrinsic matrix, rotation matrix and translation vector
    '''

    # Decomposing the projection matrix
    intrinsic_matrix, rotation_matrix, translation_vector, *_ = cv2.decomposeProjectionMatrix(
        p)

    # Scaling and removing the homogenous coordinates
    translation_vector = (translation_vector / translation_vector[3])[:3]

    return intrinsic_matrix, rotation_matrix, translation_vector

def motion_estimation(firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, config, depth):
    """
    Estimating motion of the left camera from sequential images with drift compensation
    """

    max_depth = config['parameters']['max_depth']
    detector = config['parameters']['detector']
    name = config['data']['type']

    image1_points = np.float32(firstImage_keypoints)
    image2_points = np.float32(secondImage_keypoints)    

    # Define the instrinsic camera parameters
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    # Hardcode the distorsion coefficients [k1, k2, p1, p2, k3]
    # D1 = np.array([-0.164644, 0.012281, 0.007764, 0.000446, 0.000032])  # (Left camera)
    # D2 = np.array([-0.166799, 0.012723, 0.008387, 0.000536, -0.000078])  # (Right camera)

    points_3D = np.zeros((0, 3))
    outliers = []

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    # Deleting the false depth points
    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    # Apply Ransac Algorithm to remove outliers
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

