import cv2
import json
import os, re
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import math


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

def generate_angles_near_deg(angle_deg, spread=5.0, num=10, wrap=True):
    """
    Generate a vector of angles (in degrees) close to a given angle.

    Args:
        angle_deg (float): Target angle in degrees.
        spread (float): Total spread around the angle (±spread/2).
        num (int): Number of angles to generate.
        wrap (bool): Whether to wrap angles to [-180, 180].

    Returns:
        np.ndarray: Array of angles near the given angle (degrees).
    """
    angles = np.linspace(angle_deg - spread / 2, angle_deg + spread / 2, num)
    if wrap:
        angles = (angles + 180) % 360 - 180  # wrap to [-180, 180]
    return angles

def rospred_to_pyrped(input_path, output_path):

    # Output list
    poses = []

    with open(input_path, 'r') as file:
        for line in file:
            # Split and convert to float
            values = list(map(float, line.strip().split()))
           
            tx, ty, tz = values[0:3]
            qx, qy, qz, qw = values[3:7]

            pose = {
                "pose": {
                    "translation": [tx, ty, tz],
                    "quaternion": [qx, qy, qz, qw]
                }
            }
            poses.append(pose)

    # Write each pose as a line to the .txt file
    with open(output_path, 'w') as out_file:
        for pose in poses:
            out_file.write(json.dumps(pose) + '\n')

    print(f"Successfully written {len(poses)} formatted poses to '{output_path}'.")

def GT_reader(seq):
    """
    Parse GT trajectory file and extract bounding box, UTM zone, and initial point.
    
    Args:
        seq (str): Sequence ID to match (e.g., "00", "01", etc.)
    
    Returns:
        tuple: containing max_lat, min_lat, max_lon, min_lon, zone_number, initial_point, and intital angle
    """

    file_path = "../datasets/BIEL/IRI_sequences_GT.txt"
    
    # Read and parse the file
    with open(file_path, 'r', encoding='utf-8-sig') as file: 
        content = file.read()

    # Find the specific sequence
    target_sequence = f"IRI_{seq}"
    
    # Split content into trajectory blocks
    blocks = re.split(r'type\s+latitude\s+longitude\s+name\s+desc', content)
    blocks = [block.strip() for block in blocks if block.strip()]
    
    # Find the block containing our target sequence
    target_block = None
    for block in blocks:
        if target_sequence in block:
            target_block = block
            break
    
    if not target_block:
        raise ValueError(f"Sequence {target_sequence}") 
                         
    # Extract all coordinate lines from the target block
    lines = target_block.strip().split('\n')
    coordinates = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('T'):
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    coordinates.append((lat, lon))
                except ValueError:
                    continue
                    
    # Calculate bounding box
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    max_lat = max(lats) + 0.0005
    min_lat = min(lats) - 0.0005
    max_lon = max(lons) + 0.0005
    min_lon = min(lons) - 0.0005
    
    # Get initial point (first coordinate) and convert to UTM
    initial_lat, initial_lon = coordinates[0]
    initial_x, initial_y, zone_number, _ = latlon_to_utm(initial_lat, initial_lon)
    
    # Calculate initial angle from first two points
    second_lat, second_lon = coordinates[1]
    second_x, second_y, _, _ = latlon_to_utm(second_lat, second_lon)
    
    # Calculate the direction vector
    dx = second_x - initial_x
    dy = second_y - initial_y
    
    # Calculate angle from North (0 = North, clockwise positive)
    # atan2(dx, dy) gives angle from North (y-axis)
    initial_angle_rad = math.atan2(dx, dy)
    
   # Convert to degrees
    initial_angle = math.degrees(initial_angle_rad)
    
    # Normalize to [0, 360] range
    if initial_angle < 0:
        initial_angle += 360
    
    return max_lat, min_lat, max_lon, min_lon, zone_number, (initial_x, initial_y), initial_angle

def gt_to_prediction_format(sequence_id="00", output_file_path=None, z_height=1.8):
    """
    Convert GT trajectory file to prediction format.
    
    Args:
        sequence_id (str): Sequence ID to convert (e.g., "00", "01", etc.)
        output_file_path (str): Output file path (optional, if None prints to console)
        z_height (float): Z coordinate (height) for all points, default 1.8 meters
    
    Returns:
        list: List of formatted prediction lines
    """
    
    # Read and parse the file
    gt_file_path =  "../datasets/BIEL/IRI_sequences_GT.txt"

    with open(gt_file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
    
    # Find the specific sequence
    target_sequence = f"IRI_{sequence_id}"
    
    # Split content into trajectory blocks
    blocks = re.split(r'type\s+latitude\s+longitude\s+name\s+desc', content)
    blocks = [block.strip() for block in blocks if block.strip()]
    
    # Find the block containing our target sequence
    target_block = None
    for block in blocks:
        if target_sequence in block:
            target_block = block
            break
    
    if not target_block:
        raise ValueError(f"Sequence {target_sequence} not found in file")
    
    # Extract all coordinate lines from the target block
    lines = target_block.strip().split('\n')
    coordinates = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('T'):
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    coordinates.append((lat, lon))
                except ValueError:
                    continue
    
    if not coordinates:
        raise ValueError(f"No valid coordinates found for sequence {target_sequence}")
    
    # Convert coordinates to UTM and format output
    prediction_lines = []
    
    # Convert all coordinates to UTM and format with constant Z height
    for lat, lon in coordinates:
        x, y, *_= latlon_to_utm(lat, lon)
        
        # Format line with high precision: X Y Z
        line = f"{x:.16f} {y:.16f} {z_height:.16f}"
        prediction_lines.append(line)
    
    # Output to file or console
    if output_file_path:
        with open(output_file_path, 'w') as f:
            for line in prediction_lines:
                f.write(line + '\n')
        print(f"Converted trajectory saved to: {output_file_path}")
    else:
        for line in prediction_lines:
            print(line)
    
    return prediction_lines

def convert_all_sequences(z_height=1.8):
    """
    Convert all sequences in the GT file to prediction format.
    
    Args:
        z_height (float): Z coordinate (height) for all points, default 1.8 meters
    """
    
    # Read file and find all sequences
    gt_file_path =  "../datasets/BIEL/IRI_sequences_GT.txt"
    with open(gt_file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
    
    # Find all sequence IDs
    sequence_pattern = r'IRI_(\d+)'
    sequences = re.findall(sequence_pattern, content)
    sequences = list(set(sequences))  # Remove duplicates
    sequences.sort()  # Sort numerically
    
    print(f"Found sequences: {sequences}")
    for seq_id in sequences:
        output_folder = f"../datasets/predicted/trajectories/{seq_id}/"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "GT.txt")

        try:

            gt_to_prediction_format(seq_id, output_file, z_height)
            print(f"Converted sequence {seq_id}")
        except Exception as e:
            print(f"Error converting sequence {seq_id}: {e}")

############################################################################################################
def motion_estimation_old(firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, config, depth):
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

if __name__ == "__main__":

    # seqs = [str(i).zfill(2) for i in range(2,23)]

    # for seq in seqs:
    #     input_path = f"../datasets/BIEL/ZED_ROS_Odom/odometry_{seq}.txt"
    #     output_path = f"../datasets/BIEL/{seq}/odometry_{seq}.txt"
    #     rospred_to_pyrped(input_path, output_path)
    convert_all_sequences()