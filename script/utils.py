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

    # Using SGBM matcher(Hirschmuller algorithm)
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

def improved_disparity_mapping(left_img, right_img, rgb_value=False):
    """
    Improved disparity estimation for low-texture scenes (e.g., sidewalks).
    Uses pre-processing + SGBM/WLS filtering for smoother results.
    """
    # Convert to grayscale if needed
    if rgb_value:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        num_channels = 3
    else:
        left_gray = left_img
        right_gray = right_img
        num_channels = 1

    # --- Pre-processing: Enhance texture ---
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)

    # --- Stereo Matching ---
    # SGBM Parameters (tuned for sidewalks)
    window_size = 5
    min_disp = 0
    num_disp = 16 * 5  # Must be divisible by 16

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * num_channels * window_size ** 2,
        P2=32 * num_channels * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute left disparity
    left_disp = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # --- Post-processing: WLS Filter ---
    # Reduces noise while preserving edges
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    right_disp = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0

    # WLS filter
    lmbda = 8000
    sigma = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)

    return filtered_disp

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

def compute_depth(disparity_map, focal_length, baseline):
    """Convert disparity to depth with sanity checks."""
    disparity_map[disparity_map <= 0] = 0.1  # Avoid division by zero
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map

# Calculating depth information
def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified_bl=True):
    '''

    :params left_disparity_map: disparity map of left camera
    :params left_intrinsic: intrinsic matrix for left camera
    :params left_translation: translation vector for left camera
    :params right_translation: translation vector for right camera

    '''
    # Focal length of x axis for left camera
    focal_length = left_intrinsic[0][0]

    # Calculate baseline of stereo pair
    if rectified_bl:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]
    
    # Calculate depth map
    depth_map = compute_depth(left_disparity_map, focal_length, baseline)

    return depth_map

def refine_depth_map(depth_map):
    """Fill holes and smooth the depth map."""
    # Fill invalid disparities (optional)
    depth_map_filled = cv2.inpaint(
        depth_map.astype(np.float32),
        (depth_map == 0).astype(np.uint8),
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # Median blur to reduce noise
    depth_map_smoothed = cv2.medianBlur(depth_map_filled, 5)
    return depth_map_smoothed

def plot_depth_results(left_img, right_img, depth_map, disparity_map):
    """Visualize results."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Input images
    axs[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Left Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Right Image")
    axs[0, 1].axis("off")

    # Disparity map
    disp_plot = axs[1, 0].imshow(disparity_map, cmap="viridis")
    axs[1, 0].set_title("Disparity Map")
    axs[1, 0].axis("off")
    plt.colorbar(disp_plot, ax=axs[1, 0])

    # Depth map
    depth_plot = axs[1, 1].imshow(depth_map, cmap="plasma", vmin=np.percentile(depth_map, 5), vmax=np.percentile(depth_map, 95))
    axs[1, 1].set_title("Depth Map")
    axs[1, 1].axis("off")
    plt.colorbar(depth_plot, ax=axs[1, 1])
    plt.tight_layout()
    plt.show()

def stereo_depth(left_image, right_image, P0, P1, config, stereo_complex=True, plot=False):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. 

    :params left_image: image from left camera
    :params right_image: image from right camera
    :params P0: Projection matrix for the left camera
    :params P1: Projection matrix for the right camera

    '''
    rgb_value = config['parameters']['rgb']

    # Do a more refined pipeline for difficult images with low texture
    if stereo_complex:
        # Decompose projection matrices 
        K_left, _, _ = decomposition(P0)
        focal_length = K_left[0, 0]
        baseline = abs(P1[0, 3] / P1[0, 0])  # Baseline from projection matrices

        # Compute disparity with pre-processing
        disparity_map = improved_disparity_mapping(left_image, right_image, rgb_value)

        # Compute depth
        depth_map = compute_depth(disparity_map, focal_length, baseline)
        # depth_map = refine_depth_map(depth_map)  # Optional refinement

    else:
        # First we compute the disparity map
        disparity_map = disparity_mapping(left_image,
                                    right_image,
                                    rgb_value)

        # Then decompose the left and right camera projection matrices
        l_intrinsic, _, l_translation = decomposition(
            P0)
        _, _, r_translation = decomposition(
            P1)

        # Calculate depth map for left camera
        depth_map = depth_mapping(disparity_map, l_intrinsic, l_translation, r_translation)

    if plot:
         plot_depth_results(left_image, right_image, depth_map, disparity_map)

    return depth_map, disparity_map

######################################### Feature Extraction and Matching ####################################

def feature_extractor(image, detector, mask=None):
    """
    Provide keypoints and descriptors using SIFT or ORB detectors.

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


def BF_matching(first_descriptor, second_descriptor, k=2):
    """
    Brute-Force match features between two images.

    """
    # Using BFMatcher to match the features
    feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)

    # Obtain the matches    
    matches = feature_matcher.knnMatch(
        first_descriptor, second_descriptor, k=k) # Returns the k best matches

    return matches

def feature_matching(image_left, next_image, mask, config, data_handler, plot, idx, show=False):
    """
    Perform feature matching between two consecutive images using either a brute-force
    approach or the LightGlue deep learning-based method. The function detects keypoints
    in both images, computes descriptors, matches the features, and applies filtering
    based on a given threshold. Results can be cached for future use to avoid recomputing
    matches.
    """
    name = config['data']['type']
    detector = config['parameters']['detector']
    threshold = config['parameters']['threshold']
    rectified = config['parameters']['rectified']
    if rectified:
        cache_dir = f"../datasets/predicted/prefiltered_matches/{name}/{detector}/rectified/"
    else:
        cache_dir = f"../datasets/predicted/prefiltered_matches/{name}/{detector}/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"matches_{idx}.npz")

    if os.path.exists(cache_path):
        # --- Load cached matches ---
        if idx == 0:
            print(f"Loading cached matches from {cache_dir}")
        data = np.load(cache_path, allow_pickle=True)
        keypoint_left_first = data["keypoint_left_first"]
        keypoint_left_next = data["keypoint_left_next"]
        descriptor_left_first = data["descriptor_left_first"]
        descriptor_left_next = data["descriptor_left_next"]
        matches = data["matches"]
        scores = data["scores"] if "scores" in data.files else None

        # Apply threshold filtering
        if detector == "lightglue":
            topk = np.argsort(-scores)[:threshold]
            filtered_matches = matches[topk]
            keypoint_left_first = keypoint_left_first[filtered_matches[:, 0]]
            keypoint_left_next = keypoint_left_next[filtered_matches[:, 1]]
        else:
            # For BF: use only matches below distance threshold
            filtered_matches = []
            for match1, match2 in matches:
                if match1.distance <= threshold * match2.distance:
                    filtered_matches.append(match1)
    else:
        # --- Compute matches and save to cache ---
        if idx == 0:
            print(f"Computing matches and saving to {cache_dir}")
        if detector != 'lightglue':
            keypoint_left_first, descriptor_left_first = feature_extractor(image_left, detector, mask)
            keypoint_left_next, descriptor_left_next = feature_extractor(next_image, detector, mask)

            # Brute-Force match features
            matches = BF_matching(descriptor_left_first, descriptor_left_next)
            
            # Save raw data
            np.savez(cache_path,
                     keypoint_left_first=keypoint_left_first,
                     keypoint_left_next=keypoint_left_next,
                     descriptor_left_first=descriptor_left_first,
                     descriptor_left_next=descriptor_left_next,
                     matches=matches,
                     scores=None)
        
            # Filtering out the weak features
            filtered_matches = []
            for match1, match2 in matches:
                if match1.distance <= threshold * match2.distance:
                    filtered_matches.append(match1)

        else:
            # LightGlue feature matching
            image_left = load_image(data_handler.sequence_dir + 'image_0/' + data_handler.left_camera_images[idx])
            next_image = load_image(data_handler.sequence_dir + 'image_0/' + data_handler.left_camera_images[idx + 1])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
            matcher = LightGlue(features="superpoint").eval().to(device)

            descriptor_left_first = extractor.extract(image_left.to(device))
            descriptor_left_next = extractor.extract(next_image.to(device))
            matches01 = matcher({
             "image0": descriptor_left_first, 
             "image1": descriptor_left_next
             })

            # Remove batch dimension
            descriptor_left_first, descriptor_left_next, matches01 = [
                rbd(x) for x in [descriptor_left_first, descriptor_left_next, matches01]
            ]  
            
            # Convert to numpy arrays
            keypoint_left_first = descriptor_left_first["keypoints"].cpu().numpy()
            keypoint_left_next = descriptor_left_next["keypoints"].cpu().numpy()
            matches = matches01["matches"].cpu().numpy()
            scores = matches01["scores"].cpu().detach().numpy()

            # Save raw data
            np.savez(cache_path,
                     keypoint_left_first=keypoint_left_first,
                     keypoint_left_next=keypoint_left_next,
                     descriptor_left_first=descriptor_left_first,
                     descriptor_left_next=descriptor_left_next,
                     matches=matches,
                     scores=scores)


           # Apply top-k filtering
            topk = np.argsort(-scores)[:threshold]
            filtered_matches = matches[topk]
            keypoint_left_first = keypoint_left_first[filtered_matches[:, 0]]
            keypoint_left_next = keypoint_left_next[filtered_matches[:, 1]]

    # Plot matches every 500 frames
    if not plot and idx % 500 == 0:
        save_dir = f"../datasets/predicted/matches/{name}/{detector}_{threshold}"
        os.makedirs(save_dir, exist_ok=True)
        if detector != "lightglue":
            show_matches = cv2.drawMatches(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB),
                                           keypoint_left_first,
                                           cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB),
                                           keypoint_left_next,
                                           matches, None, flags=2)
            plt.figure(figsize=(15, 5), dpi=100)
            plt.imshow(show_matches)
            plt.title(f"Matches using {detector}. Frames {idx} and {idx+1}")
            if show:
                plt.show()
            plt.savefig(os.path.join(save_dir, f"matches_{idx}.png"))
        else:
            _ = viz2d.plot_images([image_left, next_image])
            viz2d.plot_matches(keypoint_left_first, keypoint_left_next, color="lime", lw=0.2)
            # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
            plt.title(f"Matches using LightGlue. Frames {idx} and {idx+1}")
            if show:
                plt.show()
            viz2d.save_plot(os.path.join(save_dir, f"matches_{idx}.png"))
            plt.close()

    return keypoint_left_first, keypoint_left_next, filtered_matches

######################################### Motion Estimation ####################################
def motion_estimation_old(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, config):
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
        z = depth[int(v), int(u)] # From the depth map 

        # We will not consider depth greater than max_depth
        if z <= 0 or z > max_depth or np.isnan(z):
            outliers.append(indices)
            continue
        
        # Only consider the points that are in the bottom half of the image
        if v < (1/2)*720 or u < (1/8)*1280:
            outliers.append(indices)
            continue 
        # Using z we can find the x,y points in 3D coordinate (Camera coordinate system) using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    # Deleting the false depth points:
    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    image1_points_norm = cv2.undistortPoints(image1_points, intrinsic_matrix, None)
    image2_points_norm = cv2.undistortPoints(image2_points, intrinsic_matrix, None)

    # Perspective-n-Point (PnP) pose computation
    # Apply RANSAC Algorithm: matching robust to outliers and obtaing rotation and translation
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(points_3D, 
                                                image2_points, 
                                                intrinsic_matrix, 
                                                None)

    # Convert the rotation vector to a rotation matrix
    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, config):
    """
    Depth-weighted motion estimation: closer points favor translation, farther points favor rotation.
    """

    max_depth = config['parameters']['max_depth']
    detector = config['parameters']['detector']
    img_height, img_width = depth.shape

    # Camera intrinsics
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    # Get keypoint coordinates
    if detector != 'lightglue':
        image1_points = np.float32([firstImage_keypoints[m.queryIdx].pt for m in matches])
        image2_points = np.float32([secondImage_keypoints[m.trainIdx].pt for m in matches])
    else:
        image1_points = np.float32(firstImage_keypoints)
        image2_points = np.float32(secondImage_keypoints)

    pts3D, pts2D, weights = [], [], []

    for i, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

        if z <= 0 or z > max_depth or np.isnan(z):
            continue
        # if v < (3/4)*img_height or u < (1/4)*img_width:
        #     continue

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        pt3D = np.array([x, y, z])

        # Normalize weights: inverse of depth → closer points get higher weight
        weight = 1.0 / z

        pts3D.append(pt3D)
        pts2D.append(image2_points[i])
        weights.append(weight)

    if len(pts3D) < 4:
        return np.eye(3), np.zeros((3, 1)), image1_points, image2_points

    pts3D = np.array(pts3D, dtype=np.float32)
    pts2D = np.array(pts2D, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    # Normalize weights to [0.5, 2] (arbitrary range)
    weights = 0.5 + 1.5 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    # Step 1: estimate pose robustly via RANSAC (no weights yet)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3D, pts2D, intrinsic_matrix, None
    )

    if not success or inliers is None or len(inliers) < 4:
        return np.eye(3), np.zeros((3, 1)), image1_points, image2_points

    # Step 2: refine with weights (optional)
    inliers_3D = pts3D[inliers[:, 0]]
    inliers_2D = pts2D[inliers[:, 0]]
    inlier_weights = weights[inliers[:, 0]]

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            inliers_3D, inliers_2D, intrinsic_matrix, None, rvec, tvec, criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 1e-6), weights=inlier_weights
        )
    except Exception:
        pass  # Fallback if refinement fails

    rotation_matrix = cv2.Rodrigues(rvec)[0]
    return rotation_matrix, tvec, image1_points, image2_points



