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


def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, idx, config, image_left, next_image, plot = False):
    """
    Estimating motion of the left camera from sequential images with drift compensation
    """

    max_depth = config['parameters']['max_depth']
    detector = config['parameters']['detector']
    name = config['data']['type']

    if detector != 'lightglue':
        # Only considering keypoints that are matched for two sequential frames
        image1_points = np.float32(
            [firstImage_keypoints[m.queryIdx].pt for m in matches])
        image2_points = np.float32(
            [secondImage_keypoints[m.trainIdx].pt for m in matches])
    else:
        # This step is already done in the feature matching function
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

    # Collect depth values from image1_points
    depths = []
    for i, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]
        if z > 0 and np.isfinite(z):  # Filter invalid depths
            depths.append(z)

    # Compute depth distribution thresholds
    if len(depths) == 0:
        return np.eye(3), np.zeros((3, 1)), image1_points, image2_points

    depths = np.array(depths)
    lower_percentile = np.percentile(depths, 5)
    upper_percentile = max_depth

    # # If we are in a werid looking zone we rely on the ZED maps
    # if lower_percentile > 2.5: # Bad SDE detected! Heuristic to 00
    #     depth = np.load(os.path.join(f"../datasets/BIEL/{name}/depths/depth_map_{idx}.npy"))

    #     # Collect depth values from image1_points
    #     depths = []
    #     for i, (u, v) in enumerate(image1_points):
    #         z = depth[int(v), int(u)]
    #         if z > 0 and np.isfinite(z):  # Filter invalid depths
    #             depths.append(z)

    #     # Compute depth distribution thresholds
    #     if len(depths) == 0:
    #         return np.eye(3), np.zeros((3, 1)), image1_points, image2_points

    #     depths = np.array(depths)
    #     prev_lower_percentile = lower_percentile
    #     lower_percentile = np.percentile(depths, 5)
    #     upper_percentile = np.percentile(depths, 85)
    #     print(f"Relied on ZED: from l={prev_lower_percentile:.2f} to l={lower_percentile:.2f}. Frame {idx}.")
    
    # Use distribution-based filtering
    pts3D = []
    pts2D_first = []
    pts2D_next = []

    for i, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

         # Skip if invalid depth
        if z <= 0 or np.isnan(z) or not np.isfinite(z):
            continue

        # Filter based on depth distribution
        if z >= upper_percentile:
            continue

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        pt3D = np.array([x, y, z])

        pts3D.append(pt3D)
        pts2D_first.append(image1_points[i])
        pts2D_next.append(image2_points[i])

    # Return identity if insufficient points
    print(f"Percentage of used points = {len(pts3D)/10:.2f}%", end="\r")
    if len(pts3D) < 4:
        return np.eye(3), np.zeros((3, 1)), image1_points, image2_points

    # Convert to NumPy arrays
    pts3D = np.array(pts3D, dtype=np.float32)
    pts2D_first = np.array(pts2D_first, dtype=np.float32)
    pts2D_next = np.array(pts2D_next, dtype=np.float32)

    # Perspective-n-Point (PnP) pose computation
    # Apply RANSAC Algorithm: matching robust to outlier
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3D, pts2D_next, intrinsic_matrix, None
    )

    if not success or inliers is None or len(inliers) < 4:
        return np.eye(3), np.zeros((3, 1)), image1_points, image2_points
    
    # Convert the rotation vector to a rotation matrix
    rotation_matrix = cv2.Rodrigues(rvec)[0]

    # Drift compensation strategies
    t_norm = np.linalg.norm(tvec)
    
    # if t_norm > 0.3:
    #     tvec = tvec * (0.3 / t_norm)
    inliers_2D_first = pts2D_first[inliers[:, 0]]
    inliers_2D_next = pts2D_next[inliers[:, 0]]

    if plot and idx % 500 == 0:
        _ = viz2d.plot_images([cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB), cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)])
        viz2d.plot_matches(inliers_2D_first, inliers_2D_next, color="lime", lw=0.2)
        # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        plt.title(f"Selected matches using LightGlue. Frames {idx} and {idx+1}. Max_depth = {max_depth}")
        plt.show()
        plt.close()

    return rotation_matrix, tvec, image1_points, image2_points, lower_percentile, upper_percentile

